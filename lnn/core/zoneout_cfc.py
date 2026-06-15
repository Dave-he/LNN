"""Zoneout for CfC (PRD #10-98, 2026-06-15).

Implements the Zoneout regularizer from "Zoneout: Regularizing RNNs
by Preserving Hidden States" (Krueger, Maharaj, Kratz, Ramalho,
Ballas, 2016, arXiv:1606.01305, ICLR 2017).

The mechanism randomly preserves the previous hidden state with
probability p_zoneout::

    h_t = h_{t-1}                with prob p_zoneout  (zone out)
    h_t = cf_c_step(x, h_{t-1})  with prob 1 - p_zoneout

Unlike dropout (which DROPS units by setting to 0), Zoneout PRESERVES
units (replaces with the previous state). This is a form of stochastic
depth applied to recurrent cells.

Why this should work in 1D:
- Zoneout is a clean regularizer that prevents overfitting and
  stabilizes gradients.
- Zoneout preserves the recurrent step architecture (the cell output
  is still computed the same way; we just sometimes discard it).
- Per the 91-135 audit, mechanisms that ADD a useful regularizer
  to the recurrent step (rather than REPLACE it) are STRICTLY
  POSITIVE (13 winners).

This module contains:

- **ZoneoutCfCCell**: standard 3-branch CfC cell with Zoneout
  applied to the new hidden state during training.
- **ZoneoutCfCStackedNetwork**: stack of Zoneout-CfC cells.

Risks:
- Zoneout might not be useful for short 1D sequences (T=32) where
  overfitting is less of a problem.
- The 30-epoch training on small batches may not overfit much, so
  Zoneout's regularizing effect might be wasted.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ZoneoutCfCCell(nn.Module):
    """Zoneout CfC cell (PRD #10-98).

    Standard 3-branch CfC recurrent step with Zoneout applied to
    the new hidden state during training::

        h_new_cfc = cf_c_step(x, h)        # standard 3-branch CfC
        if self.training:
            mask = bernoulli(p_zoneout).expand_as(h_new_cfc)
            h_final = mask * h + (1 - mask) * h_new_cfc
        else:
            h_final = h_new_cfc

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        p_zoneout: probability of preserving the previous hidden
            state (0.0 = no Zoneout, 1.0 = always preserve).
        time_scale_init: initial value of CfC's time scale parameter.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p_zoneout: float = 0.1,
        time_scale_init: float = 1.0,
    ):
        super().__init__()
        if not 0.0 <= p_zoneout < 1.0:
            raise ValueError(f"p_zoneout must be in [0.0, 1.0), got {p_zoneout}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.p_zoneout = p_zoneout

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

        # Diagnostic: track Zoneout rate.
        self._last_zoneout_rate: float = 0.0

    def zoneout_rate(self) -> float:
        """Current Zoneout rate (diagnostic)."""
        return self.p_zoneout

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One recurrent step with Zoneout.

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

        if self.training and self.p_zoneout > 0.0:
            # Per-cell (per-neuron) Zoneout mask. Bernoulli(p_zoneout)
            # means 1 = keep h (zone out), 0 = use h_new_cfc.
            mask = torch.bernoulli(
                torch.full((1, self.hidden_size), self.p_zoneout, device=h.device, dtype=h.dtype)
            )
            h_final = mask * h + (1.0 - mask) * h_new_cfc
            # Track diagnostic.
            with torch.no_grad():
                self._last_zoneout_rate = float(mask.mean().item())
        else:
            h_final = h_new_cfc
            with torch.no_grad():
                self._last_zoneout_rate = 0.0

        return h_final


class ZoneoutCfCStackedNetwork(nn.Module):
    """Stacked Zoneout CfC cells (PRD #10-98).

    Each layer is a ``ZoneoutCfCCell``. The output is the head
    projection of the last layer's hidden state.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        output_size: output feature dimension.
        num_layers: number of stacked cells.
        p_zoneout: probability of Zoneout per cell.
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
        p_zoneout: float = 0.1,
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
                ZoneoutCfCCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    p_zoneout=p_zoneout,
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
