"""Gated Input Skip for CfC (PRD #10-96, 2026-06-15).

Implements the Highway-style gated skip connection from
"Highway Networks" (Srivastava, Greff, Schmidhuber, 2015,
arXiv:1505.00387), ported to the CfC recurrent cell.

The mechanism adds a learnable input skip path to the standard
CfC recurrent step::

    h_new_cfc = cf_c_step(x, h)    # standard 3-branch CfC update
    skip  = W_skip @ x            # linear projection of input
    gate  = sigmoid(W_gate @ [x, h])   # input-conditional gate
    h_t   = h_new_cfc + gate * skip   # gated skip additive

The skip provides a DIRECT path from input to hidden state that
bypasses the recurrent dynamics. The gate lets the model decide
when to use this leak (per-step, per-dim).

Why this should work in 1D:
- The skip provides a low-pass filter on the input that can bypass
  noisy recurrent dynamics.
- Per the 91-133 audit, mechanisms that ADD to the recurrent step
  (11/12 MoE winners) win. Mechanisms that REPLACE the recurrent
  step (HGRN, Antisymm, FastWeights) lose.
- Gated Input Skip is **additive** — it preserves W·h and CfC's
  f-gate AND adds a useful structure (the skip path).

This module contains:

- **GatedInputSkipCfCCell**: CfC-style 3-branch recurrent step
  plus a gated input skip.
- **GatedInputSkipCfCStackedNetwork**: stack of GIS cells.

Risks:
- The skip might add high-frequency noise (like FastWeights).
- CfC's f-gate might already capture most of the information the
  skip provides, making the skip redundant.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GatedInputSkipCfCCell(nn.Module):
    """Gated Input Skip CfC cell (PRD #10-96).

    Standard CfC recurrent step (3-branch form with time scale)::

        combined = [x, h]
        f = sigmoid(W_f combined)         # gate
        g = tanh(W_g combined)            # candidate
        h_out = tanh(W_h combined)        # alternative carry
        decay = sigmoid(-f * time_scale)
        h_new_cfc = decay * g + (1-decay) * h_out

    Plus a gated input skip::

        skip = W_skip @ x
        gate = sigmoid(W_gate @ [x, h])
        h_final = h_new_cfc + gate * skip

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        skip_init_scale: std of W_skip initial weights (default 0.1).
        gate_init_bias: initial bias for the gate (sigmoid(b) ≈
            initial gate value). Default 0.0 (gate ≈ 0.5).
        time_scale_init: initial value of CfC's time scale parameter.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        skip_init_scale: float = 0.1,
        gate_init_bias: float = 0.0,
        time_scale_init: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.skip_init_scale = skip_init_scale
        self.gate_init_bias = gate_init_bias

        combined_dim = input_size + hidden_size

        # Standard 3-branch CfC step.
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

        # Input skip: W_skip: input_size -> hidden_size.
        self.skip_proj = nn.Linear(input_size, hidden_size, bias=False)
        # Init small so the skip starts close to zero (don't disrupt training).
        with torch.no_grad():
            self.skip_proj.weight.normal_(mean=0.0, std=skip_init_scale)

        # Gate: input-conditional, sigmoid -> [0, 1].
        self.gate_proj = nn.Linear(combined_dim, hidden_size)
        # Init bias so gate starts at sigmoid(gate_init_bias).
        with torch.no_grad():
            self.gate_proj.bias.fill_(gate_init_bias)

        # Diagnostic: latest gate activation (per-batch mean).
        self._last_gate_mean: float = 0.0

    def skip_norm(self) -> float:
        """Frobenius norm of W_skip (diagnostic)."""
        return float(self.skip_proj.weight.norm().item())

    def gate_bias(self) -> float:
        """Current bias of the gate projection."""
        return float(self.gate_proj.bias.mean().item())

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One recurrent step with gated input skip.

        Args:
            x_t: input of shape ``[B, input_size]``.
            h: hidden state of shape ``[B, hidden_size]``.
            dt: scalar time delta.
        Returns:
            New hidden state ``[B, hidden_size]``.
        """
        # Standard 3-branch CfC step.
        combined = torch.cat([x_t, h], dim=-1)
        f = self.f_gate(combined)
        g = self.g_branch(combined)
        h_out = self.h_branch(combined)
        decay = torch.sigmoid(-f * self.time_scale * dt)
        h_new_cfc = decay * g + (1.0 - decay) * h_out

        # Gated input skip.
        skip = self.skip_proj(x_t)               # [B, H]
        gate = torch.sigmoid(self.gate_proj(combined))   # [B, H]
        h_final = h_new_cfc + gate * skip

        # Diagnostic: track gate activation.
        with torch.no_grad():
            self._last_gate_mean = float(gate.mean().item())

        return h_final


class GatedInputSkipCfCStackedNetwork(nn.Module):
    """Stacked Gated Input Skip CfC cells (PRD #10-96).

    Each layer is a ``GatedInputSkipCfCCell``. The output is the
    head projection of the last layer's hidden state.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        output_size: output feature dimension.
        num_layers: number of stacked cells.
        skip_init_scale: std of W_skip initial weights.
        gate_init_bias: initial bias for the gate.
        time_scale_init: initial value of CfC's time scale.
        return_sequences: if True, return outputs at every
            timestep; else return only the last.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        skip_init_scale: float = 0.1,
        gate_init_bias: float = 0.0,
        time_scale_init: float = 1.0,
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
                GatedInputSkipCfCCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    skip_init_scale=skip_init_scale,
                    gate_init_bias=gate_init_bias,
                    time_scale_init=time_scale_init,
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
