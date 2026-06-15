"""Hebbian Fast Weights for CfC (PRD #10-95, 2026-06-15).

Implements the fast-weight mechanism from "Using Fast Weights to
Attend to the Recent Past" (Ba, Hinton, Mnih, Romoff, Veness,
NIPS 2016, arXiv:1610.06258).

The mechanism augments the standard recurrent step with a
**Hebbian fast weight matrix** ``F_t`` that evolves at every
recurrent step::

    F_t = lambda * F_{t-1} + eta * (h_{t-1} outer h_{t-2})

where ``lambda`` is a decay factor and ``eta`` is a Hebbian
learning rate. The fast weights provide a **short-term memory**
that captures pairwise interactions between recent hidden
states.

The recurrent step then uses BOTH the slow weights (W_h) and
the fast weights (F_t)::

    h_t = cf_c_step(x, h, F_t @ h)

This module ports the fast-weights idea to a CfC-friendly setting:

- **FastWeightsCfCCell**: CfC-style update with Hebbian fast
  weights ``F_t`` maintained as a buffer across the forward pass.
  The fast-weight term ``F_t @ h`` is concatenated with the
  standard input to the gate and candidate branches.

- **FastWeightsCfCStackedNetwork**: stack of FastWeightsCfC cells.

The mechanism is **structural** (adds to the recurrent step) and
**distribution-agnostic** (works on any 1D-ND target). Per the
91-132 audit, mechanisms that ADD a useful inductive bias to the
recurrent step (rather than REPLACE it) are STRICTLY POSITIVE
(12 winners). Rounds 128-132 (oscillator, ELM, MR-MoE+dual attn,
HGRN, Antisymm) all proposed alternatives to the recurrent step
and LOSE in 1D.

Why this should work in 1D:
- The fast-weight term ``F_t @ h`` provides a learned,
  time-varying projection that captures pairwise interactions
  between recent hidden states.
- This is "additive" — it does not replace W_h or CfC's f-gate.
- The Hebbian update is unsupervised (no learning signal beyond
  the outer product), but the recurrent dynamics learn to USE
  the fast-weight term.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FastWeightsCfCCell(nn.Module):
    """Hebbian Fast Weights CfC cell (PRD #10-95).

    Maintains a fast weight matrix ``F_t`` of shape ``[H, H]`` that
    evolves at every recurrent step via the Hebbian rule::

        F_t = lambda * F_{t-1} + eta * (h_{t-1} outer h_{t-2})

    The recurrent step is::

        f = sigmoid(W_f [x, h, F_t @ h])  # gate
        g = tanh(W_g [x, h, F_t @ h])     # candidate
        h_t = (1 - f) * h + f * g

    The fast-weight term ``F_t @ h`` is concatenated with the
    standard input ``[x, h]`` to provide an additional context
    signal.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        init_lambda: initial decay factor for fast weights.
        init_eta: initial Hebbian learning rate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        init_lambda: float = 0.9,
        init_eta: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # CfC-style gates. Input is [x, h, F_t @ h] (3H + input_size).
        combined_dim = input_size + 2 * hidden_size
        self.f_gate = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.Sigmoid(),
        )
        self.g_branch = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.Tanh(),
        )
        # h_branch is the "no input" branch (just keep h).
        # This is the standard CfC form.
        # We don't need a separate h_branch because the standard
        # CfC equation `h_t = (1-f)*h + f*g` already preserves h.

        # Learnable scalars for the Hebbian update.
        # Use sigmoid to keep them in [0, 1].
        self.raw_lambda = nn.Parameter(torch.tensor(self._inv_sigmoid(init_lambda)))
        self.raw_eta = nn.Parameter(torch.tensor(self._inv_sigmoid(init_eta)))

        # Caches for diagnostics.
        self._F: torch.Tensor | None = None
        self._h_prev: torch.Tensor | None = None

    @staticmethod
    def _inv_sigmoid(p: float) -> float:
        """Inverse sigmoid for parameter init."""
        import math
        p = max(min(p, 0.999), 0.001)
        return math.log(p / (1.0 - p))

    @property
    def lam(self) -> torch.Tensor:
        """Current decay factor (tensor in [0, 1])."""
        return torch.sigmoid(self.raw_lambda)

    @property
    def eta(self) -> torch.Tensor:
        """Current Hebbian learning rate (tensor in [0, 1])."""
        return torch.sigmoid(self.raw_eta)

    def reset_state(self) -> None:
        """Reset the fast weight matrix and the previous h cache."""
        self._F = None
        self._h_prev = None

    def fast_weight_norm(self) -> float:
        """Frobenius norm of the current fast weight matrix (diagnostic)."""
        if self._F is None:
            return 0.0
        return float(self._F.norm().item())

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
        H = h.shape[1]
        device = h.device

        # Initialize F if this is the first step.
        if self._F is None:
            self._F = torch.zeros(H, H, device=device, dtype=h.dtype)

        # Compute the fast-weight term: F_t @ h.
        # F_t is [H, H], h is [B, H] -> F_t @ h.T is [H, B] -> .T is [B, H]
        Fh = torch.matmul(h, self._F.t())  # [B, H]

        # CfC step with the fast-weight term as additional context.
        combined = torch.cat([x_t, h, Fh], dim=-1)
        f = self.f_gate(combined)  # [B, H]
        g = self.g_branch(combined)  # [B, H]
        h_new = (1.0 - f) * h + f * g

        # Update F_t for the next step (Hebbian).
        # F_{t+1} = lambda * F_t + eta * (h_t outer h_{t-1})
        if self._h_prev is not None:
            # Outer product: h_new [B, H] outer h_prev [B, H] -> [B, H, H]
            # Then average over batch.
            outer = torch.einsum("bi,bj->bij", h_new.detach(), self._h_prev.detach())
            outer = outer.mean(dim=0)  # [H, H]
            # No torch.no_grad() so gradient flows to lam and eta.
            self._F = self.lam * self._F + self.eta * outer
        # Cache h_new for the next step's Hebbian update.
        self._h_prev = h_new.detach()

        return h_new


class FastWeightsCfCStackedNetwork(nn.Module):
    """Stacked Hebbian Fast Weights CfC cells (PRD #10-95).

    Each layer is a FastWeightsCfCCell. The output is the head
    projection of the last layer's hidden state.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        output_size: output feature dimension.
        num_layers: number of stacked cells.
        init_lambda: initial decay factor for fast weights.
        init_eta: initial Hebbian learning rate.
        return_sequences: if True, return outputs at every
            timestep; else return only the last.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        init_lambda: float = 0.9,
        init_eta: float = 0.1,
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else hidden_size
            self.cells.append(
                FastWeightsCfCCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    init_lambda=init_lambda,
                    init_eta=init_eta,
                )
            )
        self.head = nn.Linear(hidden_size, output_size)

    def reset_state(self) -> None:
        """Reset the fast weight matrices of all cells."""
        for cell in self.cells:
            cell.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input of shape ``[B, T, input_size]``.
        Returns:
            Output of shape ``[B, T, output_size]`` if
            ``return_sequences=True`` else ``[B, output_size]``.
        """
        # Reset fast weights for each new sequence.
        self.reset_state()
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
