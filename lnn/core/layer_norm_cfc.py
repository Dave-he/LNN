"""Layer Normalization for CfC (PRD #10-97, 2026-06-15).

Implements the Layer Normalization mechanism from
"Layer Normalization" (Ba, Kiros, Hinton, 2016, arXiv:1607.06450),
ported to the CfC recurrent cell.

The mechanism applies per-sample Layer Normalization to the
**input of the f-gate, g-branch, and h-branch** (Ba et al. 2016
§3.2 recommendation). This keeps the gate input at a consistent
scale, preventing saturation and improving gradient flow::

    combined = [x, h]                 # raw
    combined = LayerNorm(combined)    # per-sample normalize
    f = sigmoid(W_f combined)         # gate
    g = tanh(W_g combined)            # candidate
    h_out = tanh(W_h combined)        # alternative carry
    decay = sigmoid(-f * time_scale)
    h_new = decay * g + (1-decay) * h_out

Why this should work in 1D:
- LN is the most well-established "additive" mechanism for RNNs
  (Ba et al. 2016 showed 2-7× speedup on attention and RNN tasks).
- LN preserves W·h and CfC's f-gate (normalization is pre-projection).
- LN is **structural** — modifies the input to the recurrent step.
- Per the 91-134 audit, mechanisms that ADD a useful normalization
  to the recurrent step (rather than REPLACE it) are STRICTLY
  POSITIVE (13 winners).

This module contains:

- **LayerNormCfCCell**: standard 3-branch CfC cell with LN applied
  to the combined [x, h] input.
- **LayerNormCfCStackedNetwork**: stack of LN-CfC cells.

Risks:
- LN might be redundant with CfC's own normalization (time_scale).
- LN might over-constrain the representation, especially on smooth
  data where the gate is already well-behaved.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LayerNormCfCCell(nn.Module):
    """Layer Normalization CfC cell (PRD #10-97).

    Standard 3-branch CfC recurrent step with LN applied to the
    combined [x, h] input BEFORE the linear projections.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        time_scale_init: initial value of CfC's time scale parameter.
        ln_eps: epsilon for Layer Norm numerical stability.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_scale_init: float = 1.0,
        ln_eps: float = 1e-5,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ln_eps = ln_eps

        combined_dim = input_size + hidden_size

        # Layer Norm applied to the combined input [x, h].
        # Per-sample normalization over the feature dimension.
        self.layer_norm = nn.LayerNorm(combined_dim, eps=ln_eps)

        # Standard 3-branch CfC step (operates on LN-normalized input).
        self.f_gate = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.Sigmoid(),
        )
        self.g_branch = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.Tanh(),
        )
        self.h_branch = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.Tanh(),
        )
        self.time_scale = nn.Parameter(torch.full((hidden_size,), float(time_scale_init)))

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One recurrent step with Layer Normalization.

        Args:
            x_t: input of shape ``[B, input_size]``.
            h: hidden state of shape ``[B, hidden_size]``.
            dt: scalar time delta.
        Returns:
            New hidden state ``[B, hidden_size]``.
        """
        # Concatenate [x, h].
        combined = torch.cat([x_t, h], dim=-1)
        # Apply Layer Norm.
        combined = self.layer_norm(combined)
        # Standard 3-branch CfC step.
        f = self.f_gate(combined)
        g = self.g_branch(combined)
        h_out = self.h_branch(combined)
        decay = torch.sigmoid(-f * self.time_scale * dt)
        h_new = decay * g + (1.0 - decay) * h_out
        return h_new


class LayerNormCfCStackedNetwork(nn.Module):
    """Stacked Layer Norm CfC cells (PRD #10-97).

    Each layer is a ``LayerNormCfCCell``. The output is the head
    projection of the last layer's hidden state.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        output_size: output feature dimension.
        num_layers: number of stacked cells.
        time_scale_init: initial value of CfC's time scale.
        ln_eps: epsilon for Layer Norm.
        return_sequences: if True, return outputs at every
            timestep; else return only the last.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        time_scale_init: float = 1.0,
        ln_eps: float = 1e-5,
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
                LayerNormCfCCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    time_scale_init=time_scale_init,
                    ln_eps=ln_eps,
                )
            )
        self.head = nn.Linear(hidden_size, output_size)

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
            torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
            for _ in range(self.num_layers)
        ]
        outputs = []
        for t in range(T):
            inp = torch.nan_to_num(x[:, t, :], nan=0.0)
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
