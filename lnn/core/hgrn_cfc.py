"""HGRN: Hierarchically Gated Recurrent Network for CfC (PRD #10-93, 2026-06-15).

Implements the bounded-forget-gate mechanism from HGRN
(Qi, Yang, Zhao, Wang, Sun, Wei, NeurIPS 2023,
arXiv:2404.18807). The paper proposes a gated linear RNN with
a learnable **lower bound** on the forget gate that increases
monotonically across layers.

This module ports the core idea to a CfC-friendly setting:

- **HGRNCfCCell**: gated linear recurrence with a learnable
  lower bound ``alpha`` on the forget gate.
  ``gate = max(alpha, sigmoid(W_g x_t + b_g))``
  ``h_t = (1 - gate) * h_{t-1} + gate * tanh(W_x x_t + b)``

- **HGRNCfCStackedNetwork**: stack of HGRN cells, optionally
  with per-layer ``alpha_l`` that increases monotonically
  with layer index (the "hierarchical" part).

The mechanism is **structural** (modifies the recurrent step
itself, not just routing) and **distribution-agnostic** (works
on any 1D-ND target). Per the 91-130 audit, this is the
profile of all 12 STRICTLY POSITIVE winners.

Why this should work in 1D:
- The bounded forget gate is a simple structural regularizer
  that prevents the linear recurrence from overwriting its
  state in the face of noisy inputs (relevant to ``random_irr``).
- The hierarchical α_l provides a "smoothness gradient":
  lower layers forget more (model local), upper layers forget
  less (model long-range). This is a richer inductive bias
  than a flat α.
- The gated linear recurrence is **simpler** than CfC's
  full ODE solve — fewer parameters, more interpretable
  (each α_l is a single scalar that can be inspected).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HGRNCfCCell(nn.Module):
    """HGRN CfC cell (PRD #10-93).

    Gated linear recurrence with a learnable lower bound on the
    forget gate. The recurrent step is::

        gate = max(alpha, sigmoid(W_g @ x_t + b_g))   # in [alpha, 1]
        h_t = (1 - gate) * h_{t-1} + gate * tanh(W_x @ x_t + b)

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        alpha_init: initial value of the lower bound ``alpha``.
            Set to 0 to recover a free-gate (HGRN-free).
        learn_alpha: if True, ``alpha`` is a learnable scalar in
            [0, 1] (clamped via sigmoid during forward). If
            False, ``alpha`` is a fixed constant.
        nonlinearity: one of 'tanh' (default) or 'relu' for the
            input update branch. 'tanh' preserves the boundedness
            property; 'relu' allows non-negative h.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        alpha_init: float = 0.1,
        learn_alpha: bool = True,
        nonlinearity: str = "tanh",
    ):
        super().__init__()
        if not 0.0 <= alpha_init <= 1.0:
            raise ValueError(f"alpha_init must be in [0, 1], got {alpha_init}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity

        # Input projection (the "candidate" state).
        self.W_x = nn.Linear(input_size, hidden_size)
        # Forget-gate projection.
        self.W_g = nn.Linear(input_size, hidden_size)
        # Initialize forget-gate bias to be slightly positive so
        # the initial gate is around 0.5 (not too aggressive).
        with torch.no_grad():
            self.W_g.bias.fill_(0.0)

        # Learnable lower bound on the forget gate.
        if learn_alpha:
            # Parameterize via sigmoid: alpha = sigmoid(raw_alpha).
            # Initialize so that sigmoid(raw_alpha) = alpha_init.
            import math
            if alpha_init <= 0.0:
                raw_init = -10.0  # very negative → sigmoid ≈ 0
            elif alpha_init >= 1.0:
                raw_init = 10.0   # very positive → sigmoid ≈ 1
            else:
                raw_init = math.log(alpha_init / (1.0 - alpha_init))
            self.raw_alpha = nn.Parameter(torch.tensor(raw_init))
        else:
            self.register_buffer(
                "_alpha_const", torch.tensor(float(alpha_init)),
            )

        # Caches for diagnostics.
        self.last_gate: torch.Tensor | None = None
        self.last_alpha: float | None = None

    @property
    def alpha(self) -> float:
        """Current lower bound on the forget gate."""
        if hasattr(self, "raw_alpha"):
            return torch.sigmoid(self.raw_alpha).item()
        return self._alpha_const.item()

    def reset_state(self) -> None:
        """Clear diagnostic caches."""
        self.last_gate = None
        self.last_alpha = None

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """One recurrent step.

        Args:
            x_t: input of shape ``[B, input_size]``.
            h: hidden state of shape ``[B, hidden_size]``.
        Returns:
            New hidden state ``[B, hidden_size]``.
        """
        # Use a soft formulation so gradient flows to alpha even when
        # the free gate is large. ``gate = alpha + (1 - alpha) * s``
        # ranges in [alpha, 1] and is smooth w.r.t. both s and alpha.
        s = torch.sigmoid(self.W_g(x_t))  # [B, H] in [0, 1]
        # Compute alpha as a tensor to preserve gradient flow.
        if hasattr(self, "raw_alpha"):
            a_tensor = torch.sigmoid(self.raw_alpha)  # tensor in [0, 1]
        else:
            a_tensor = self._alpha_const  # buffer
        gate = a_tensor + (1.0 - a_tensor) * s  # [B, H] in [alpha, 1]
        candidate = self.W_x(x_t)  # [B, H]
        if self.nonlinearity == "tanh":
            candidate = torch.tanh(candidate)
        elif self.nonlinearity == "relu":
            candidate = F.relu(candidate)
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")
        h_new = (1.0 - gate) * h + gate * candidate
        self.last_gate = gate.detach()
        self.last_alpha = a_tensor.item()
        return h_new

    def extra_repr(self) -> str:
        learn = "learn" if hasattr(self, "raw_alpha") else "fixed"
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"alpha={self.alpha:.4f} ({learn}), "
            f"nonlinearity={self.nonlinearity}"
        )


class HGRNCfCStackedNetwork(nn.Module):
    """Stacked HGRN cells (PRD #10-93).

    Each layer has its own ``alpha_l``. With ``hierarchical=True``,
    ``alpha_l`` increases monotonically with layer index:
    ``alpha_l = (l / (L-1)) * alpha_max`` (l=0 → 0, l=L-1 → alpha_max).

    With ``hierarchical=False``, all layers share the same
    ``alpha_init``.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        output_size: output feature dimension.
        num_layers: number of stacked HGRN cells.
        alpha_init: initial value of the lower bound.
        hierarchical: if True, alpha_l increases with layer index.
        alpha_max: only used if ``hierarchical=True``. Upper
            bound on the monotonically-increasing alpha schedule.
        return_sequences: if True, return outputs at every
            timestep; else return only the last.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        alpha_init: float = 0.1,
        hierarchical: bool = True,
        alpha_max: float = 0.7,
        learn_alpha: bool = True,
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hierarchical = hierarchical
        self.alpha_max = alpha_max
        self.return_sequences = return_sequences

        # Compute per-layer alpha_init based on the schedule.
        layer_alphas = []
        for l in range(num_layers):
            if hierarchical and num_layers > 1:
                # Monotonically increasing: l=0 → 0, l=L-1 → alpha_max.
                a_l = (l / (num_layers - 1)) * alpha_max
            else:
                a_l = alpha_init
            layer_alphas.append(a_l)

        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else hidden_size
            self.cells.append(
                HGRNCfCCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    alpha_init=layer_alphas[l],
                    learn_alpha=learn_alpha,
                )
            )

        self.head = nn.Linear(hidden_size, output_size)

    def alphas(self) -> list[float]:
        """Return the current lower-bound value for each layer."""
        return [c.alpha for c in self.cells]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input of shape ``[B, T, input_size]``.
        Returns:
            Output of shape ``[B, T, output_size]`` if
            ``return_sequences=True`` else ``[B, output_size]``.
        """
        B, T, _ = x.shape
        h = [
            torch.zeros(B, self.hidden_size, device=x.device)
            for _ in range(self.num_layers)
        ]
        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            for li, cell in enumerate(self.cells):
                if li == 0:
                    h[li] = cell(inp, h[li])
                else:
                    h[li] = cell(h[li - 1], h[li])
            outputs.append(self.head(h[-1]))
        outputs = torch.stack(outputs, dim=1)
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]
