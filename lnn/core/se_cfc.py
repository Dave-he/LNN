"""Squeeze-and-Excitation (SE) Channel Attention for CfC (PRD #10-102, 2026-06-15).

Adds per-channel input attention to the input fed to the CfC
recurrent step. Inspired by SE-Net (Hu et al. 2017, CVPR 2018
winner).

The mechanism::

    score = sigmoid(W_score [x_t, h])   # [B, D_in] in [0, 1]
    x_se = score * x_t                  # [B, D_in] recalibrated
    h_t = cf_c_step(x_se, h_{t-1})      # standard 3-branch CfC

Why this should work in 1D:
- CfC's f-gate is per-hidden-dim scalar over [x, h] — different
  dimension from input.
- SE is per-input-channel (matches input dim) and uses BOTH x and
  h to compute the score (cross-attention style).
- Per the 91-139 audit, input-side processing that adds structure
  the f-gate doesn't provide is a winner.

This module contains:

- **SECfCCell**: standard 3-branch CfC cell with SE channel
  attention.
- **SECfCStackedNetwork**: stack of SE-CfC cells.

Risks:
- SE adds parameters that could overfit on noisy data.
- Score=0 can zero out input (information bottleneck).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SECfCCell(nn.Module):
    """CfC cell with Squeeze-and-Excitation channel attention.

    Standard 3-branch CfC recurrent step preceded by SE channel
    attention applied to the input::

        score = sigmoid(W_score [x_t, h])    # [B, D_in] in [0, 1]
        x_se = score * x_t                   # [B, D_in] recalibrated
        combined = [x_se, h]
        f = sigmoid(W_f combined)
        g = tanh(W_g combined)
        h_out = tanh(W_h combined)
        decay = sigmoid(-f * time_scale)
        h_new = decay * g + (1-decay) * h_out

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        mode: how to compute the SE score:
            - "concat": from concat [x, h] (default, cross-attention)
            - "input": from input only (like GLU but with h context)
            - "hidden": from hidden only
        time_scale_init: initial value of CfC's time scale parameter.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mode: str = "concat",
        time_scale_init: float = 1.0,
    ):
        super().__init__()
        assert mode in ("concat", "input", "hidden"), f"mode must be in (concat, input, hidden), got {mode}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode

        # SE score network.
        if mode == "concat":
            score_in = input_size + hidden_size
        elif mode == "input":
            score_in = input_size
        else:  # hidden
            score_in = hidden_size
        self.se_score = nn.Sequential(
            nn.Linear(score_in, input_size),
            nn.Sigmoid(),
        )

        # Standard 3-branch CfC step (operates on SE-recalibrated input).
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

    def score(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute the SE channel attention score in [0, 1]."""
        if self.mode == "concat":
            inp = torch.cat([x_t, h], dim=-1)
        elif self.mode == "input":
            inp = x_t
        else:  # hidden
            inp = h
        return self.se_score(inp)

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One recurrent step with SE channel attention.

        Args:
            x_t: input at current timestep, [B, input_size].
            h: hidden state, [B, hidden_size].
            dt: scalar time delta.
        Returns:
            New hidden state, [B, hidden_size].
        """
        # SE channel attention.
        score = self.score(x_t, h)  # [B, D_in] in [0, 1]
        x_se = score * x_t           # [B, D_in] recalibrated

        # Standard 3-branch CfC step.
        combined = torch.cat([x_se, h], dim=-1)
        f = self.f_gate(combined)
        g = self.g_branch(combined)
        h_out = self.h_branch(combined)
        decay = torch.sigmoid(-f * self.time_scale * dt)
        h_new = decay * g + (1.0 - decay) * h_out
        return h_new


class SECfCStackedNetwork(nn.Module):
    """Stacked SE-CfC cells (PRD #10-102).

    Each layer is a ``SECfCCell``. The SE attention is applied per
    layer to the input. The output is the head projection of the
    last layer's hidden state.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        output_size: output feature dimension.
        num_layers: number of stacked cells.
        mode: SE mode ("concat", "input", or "hidden").
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
        mode: str = "concat",
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
                SECfCCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    mode=mode,
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
