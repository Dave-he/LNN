"""Sinusoidal Time Embedding for CfC (PRD #10-100, 2026-06-15).

Adds an explicit time embedding to the input fed to the CfC
recurrent step. Inspired by Transformer positional encoding
(Vaswani et al. 2017).

The mechanism::

    t_emb = sinusoidal_encoding(t/T)     # [B, D_te]
    x_aug = concat([x_t, t_emb])         # add D_te to input dim
    h_t = cf_c_step(x_aug, h_{t-1})

Why this should work in 1D:
- CfC has no explicit knowledge of which timestep it's at.
- Sinusoidal time embedding gives the cell a "clock" that helps
  detect regime switches and time-varying patterns.
- Per the 91-137 audit, mechanisms that ADD useful input-side
  processing to the recurrent step (rather than REPLACE it) are
  STRICTLY POSITIVE (13 winners including QuITE round 102, GIS 134).

This module contains:

- **SinusoidalTimeEmbCfCCell**: standard 3-branch CfC cell with
  sinusoidal time embedding applied to the input.
- **SinusoidalTimeEmbCfCStackedNetwork**: stack of TE-CfC cells.

Risks:
- The f-gate might already implicitly learn the time information
  if the input has time-correlated structure.
- The time embedding is parameter-free (sinusoidal) so it cannot
  adapt to data — but this is also its strength (no overfitting).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_time_embedding(
    t: torch.Tensor,
    dim: int = 4,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """Compute sinusoidal time embedding (Vaswani et al. 2017).

    Args:
        t: time steps, any shape.
        dim: embedding dimension (even).
        max_period: max period for the lowest frequency.
    Returns:
        Time embedding of shape t.shape + (dim,).
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
    )  # [half], decreasing from 1 to 1/max_period
    # Broadcast: t * freqs
    # t shape: [...], freqs shape: [half]
    # args shape: [..., half]
    args = t.float().unsqueeze(-1) * freqs  # broadcast over last dim
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [..., dim]
    return emb


class SinusoidalTimeEmbCfCCell(nn.Module):
    """CfC cell with sinusoidal time embedding applied to the input.

    Standard 3-branch CfC recurrent step preceded by a sinusoidal
    time embedding concatenated to the input::

        t_emb = sinusoidal_encoding(t/T)     # [B, D_te]
        x_aug = concat([x_t, t_emb])         # [B, D_in + D_te]
        combined = [x_aug, h]
        f = sigmoid(W_f combined)
        g = tanh(W_g combined)
        h_out = tanh(W_h combined)
        decay = sigmoid(-f * time_scale)
        h_new = decay * g + (1-decay) * h_out

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        time_emb_dim: sinusoidal time embedding dimension (default 4).
        max_period: max period for the lowest frequency
            (default 10000.0, Vaswani 2017).
        time_scale_init: initial value of CfC's time scale parameter.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_emb_dim: int = 4,
        max_period: float = 10000.0,
        time_scale_init: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_emb_dim = time_emb_dim
        self.max_period = max_period

        # The input fed to the recurrent step is x_t concatenated with t_emb.
        augmented_input_size = input_size + time_emb_dim
        self.augmented_input_size = augmented_input_size

        # Standard 3-branch CfC step (operates on augmented input).
        combined_dim = augmented_input_size + hidden_size
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

    def time_embedding(self, t_norm: torch.Tensor) -> torch.Tensor:
        """Compute the time embedding for normalized t in [0, 1]."""
        return sinusoidal_time_embedding(
            t_norm, dim=self.time_emb_dim, max_period=self.max_period,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        t_norm: torch.Tensor | None = None,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One recurrent step with time embedding.

        Args:
            x_t: input at current timestep, [B, input_size].
            h: hidden state, [B, hidden_size].
            t_norm: normalized time in [0, 1], scalar or [B] or [B, 1].
                If None, treated as 0.0.
            dt: scalar time delta.
        Returns:
            New hidden state, [B, hidden_size].
        """
        B = x_t.shape[0]
        if t_norm is None:
            t_norm = torch.zeros(B, device=x_t.device, dtype=x_t.dtype)
        elif t_norm.dim() == 0:
            t_norm = t_norm.expand(B)
        # Compute time embedding.
        t_emb = self.time_embedding(t_norm)  # [B, time_emb_dim]
        # Concat with input.
        x_aug = torch.cat([x_t, t_emb], dim=-1)  # [B, augmented_input_size]

        # Standard 3-branch CfC step.
        combined = torch.cat([x_aug, h], dim=-1)
        f = self.f_gate(combined)
        g = self.g_branch(combined)
        h_out = self.h_branch(combined)
        decay = torch.sigmoid(-f * self.time_scale * dt)
        h_new = decay * g + (1.0 - decay) * h_out
        return h_new


class SinusoidalTimeEmbCfCStackedNetwork(nn.Module):
    """Stacked TE-CfC cells (PRD #10-100).

    Each layer is a ``SinusoidalTimeEmbCfCCell``. The time embedding
    is applied per layer to the input. The output is the head
    projection of the last layer's hidden state.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        output_size: output feature dimension.
        num_layers: number of stacked cells.
        time_emb_dim: sinusoidal time embedding dimension.
        max_period: max period for the lowest frequency.
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
        time_emb_dim: int = 4,
        max_period: float = 10000.0,
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
        self.time_emb_dim = time_emb_dim
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else hidden_size
            self.cells.append(
                SinusoidalTimeEmbCfCCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    time_emb_dim=time_emb_dim,
                    max_period=max_period,
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
        # Normalized time in [0, 1] for each timestep.
        t_norm_seq = torch.linspace(0.0, 1.0, T, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            inp = torch.nan_to_num(x[:, t, :], nan=0.0)
            t_norm = t_norm_seq[t]
            for li, cell in enumerate(self.cells):
                h_new = cell(inp, h[li], t_norm=t_norm)
                h[li] = h_new
                # For subsequent layers, the input is the previous layer's hidden.
                inp = h[li]
            outputs.append(self.head(h[-1]))
        outputs = torch.stack(outputs, dim=1)
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]
