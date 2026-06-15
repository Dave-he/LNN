"""Gated Linear Unit (GLU) Input Modulation for CfC (PRD #10-101, 2026-06-15).

Adds a per-feature input gate (LSTM-style) to the input fed to the
CfC recurrent step. Inspired by GLU (Dauphin et al. 2017) and LSTM
input gates (Hochreiter & Schmidhuber 1997).

The mechanism::

    x_gate = sigmoid(W_gate x_t)      # [B, D_in] in [0, 1]
    x_gated = x_gate * x_t            # [B, D_in] modulated input
    h_t = cf_c_step(x_gated, h_{t-1})  # standard 3-branch CfC

Why this should work in 1D:
- CfC's f-gate is per-hidden-dim scalar over [x, h] — it can't
  selectively modulate INDIVIDUAL input features.
- A separate per-feature input gate (GLU) gives the cell finer
  control: it can decide per-feature whether to let x through.
- Per the 91-138 audit, mechanisms that ADD useful input-side
  processing to the recurrent step (rather than REPLACE it) are
  STRICTLY POSITIVE (13 winners including QuITE 102, GIS 134).

This module contains:

- **GatedLinearUnitCfCCell**: standard 3-branch CfC cell with GLU
  input modulation.
- **GatedLinearUnitCfCStackedNetwork**: stack of GLU-CfC cells.

Risks:
- GLU adds parameters that could overfit on noisy data.
- The f-gate might already provide sufficient modulation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GatedLinearUnitCfCCell(nn.Module):
    """CfC cell with GLU input modulation.

    Standard 3-branch CfC recurrent step preceded by a GLU input
    modulation::

        x_gate = sigmoid(W_gate x_t)         # [B, D_in] in [0, 1]
        x_gated = x_gate * x_t               # [B, D_in] modulated input
        combined = [x_gated, h]
        f = sigmoid(W_f combined)
        g = tanh(W_g combined)
        h_out = tanh(W_h combined)
        decay = sigmoid(-f * time_scale)
        h_new = decay * g + (1-decay) * h_out

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        time_scale_init: initial value of CfC's time scale parameter.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_scale_init: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # GLU input gate: per-feature sigmoid.
        self.input_gate = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid(),
        )

        # Standard 3-branch CfC step (operates on gated input).
        combined_dim = input_size + hidden_size
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

    def gate(self, x_t: torch.Tensor) -> torch.Tensor:
        """Compute the input gate values in [0, 1]."""
        return self.input_gate(x_t)

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One recurrent step with GLU input modulation.

        Args:
            x_t: input at current timestep, [B, input_size].
            h: hidden state, [B, hidden_size].
            dt: scalar time delta.
        Returns:
            New hidden state, [B, hidden_size].
        """
        # GLU input modulation.
        x_gate = self.input_gate(x_t)  # [B, D_in] in [0, 1]
        x_gated = x_gate * x_t          # [B, D_in] modulated

        # Standard 3-branch CfC step.
        combined = torch.cat([x_gated, h], dim=-1)
        f = self.f_gate(combined)
        g = self.g_branch(combined)
        h_out = self.h_branch(combined)
        decay = torch.sigmoid(-f * self.time_scale * dt)
        h_new = decay * g + (1.0 - decay) * h_out
        return h_new


class GatedLinearUnitCfCStackedNetwork(nn.Module):
    """Stacked GLU-CfC cells (PRD #10-101).

    Each layer is a ``GatedLinearUnitCfCCell``. The GLU modulation
    is applied per layer to the input. The output is the head
    projection of the last layer's hidden state.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        output_size: output feature dimension.
        num_layers: number of stacked cells.
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
                GatedLinearUnitCfCCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
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
                h_new = cell(inp, h[li])
                h[li] = h_new
                # For subsequent layers, the input is the previous layer's hidden.
                inp = h[li]
            outputs.append(self.head(h[-1]))
        outputs = torch.stack(outputs, dim=1)
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]
