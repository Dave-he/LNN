"""MONO-CfC (Monotonic Activation CfC) (PRD #10-116, Round 154, 2026-06-15).

Replaces the Tanh activations in CfC's g_branch and h_branch with
monotonic alternatives (Softplus). Inspired by monotonic networks
(Chilinski & Silva 2020, "Neural Likelihoods for Continuous-Time
Markov Chains").

Standard CfC uses Tanh in both g_branch and h_branch::

    h_t = σ(-f · τ) · g(x,h) + (1 - σ(-f · τ)) · h_branch(x,h)

where g and h_branch are::

    g_branch = Tanh(W_g · [x, h] + b_g)
    h_branch = Tanh(W_h · [x, h] + b_h)

Tanh is NOT monotonic in the output direction — negative input
gives negative output, positive input gives positive output.

This module provides 4 variants:

- **mono_g**: replace Tanh in g_branch with Softplus (positive
  monotonic).
- **mono_h**: replace Tanh in h_branch with Softplus.
- **mono_both**: replace Tanh in BOTH g_branch and h_branch.
- **mono_sig**: replace Tanh with Sigmoid (control, bounded [0,1]).

Audit context (91-153): 14 strictly positive + 10 target-dep +
22 negatives = 46 mechanism classes. Round 153 FiLM-CfC tested
multiplicative modulation. Round 154 tests monotonic activation.

Risks:
- Softplus outputs are positive, breaking bidirectional flow.
- May cause mode collapse on oscillating data (sin).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.sequence_utils import select_step_delta, select_step_mask


class MonoCfCCell(nn.Module):
    """Mono-CfC cell: monotonic activation CfC.

    Args:
        input_size: input feature dimension D.
        hidden_size: hidden state dimension.
        mono_mode: 'g_only', 'h_only', 'both', 'sigmoid'.
            Determines which Tanh activations are replaced.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mono_mode: str = "g_only",
    ):
        super().__init__()
        assert mono_mode in ("g_only", "h_only", "both", "sigmoid"), \
            f"mono_mode must be one of 'g_only', 'h_only', 'both', 'sigmoid', got {mono_mode}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mono_mode = mono_mode

        # f_gate: Sigmoid (unchanged across all variants).
        self.f_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )

        # g_branch and h_branch: Tanh or Softplus depending on mode.
        if mono_mode == "g_only":
            g_act = nn.Softplus()
            h_act = nn.Tanh()
        elif mono_mode == "h_only":
            g_act = nn.Tanh()
            h_act = nn.Softplus()
        elif mono_mode == "both":
            g_act = nn.Softplus()
            h_act = nn.Softplus()
        else:  # sigmoid
            g_act = nn.Sigmoid()
            h_act = nn.Sigmoid()

        self.g_branch = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            g_act,
        )
        self.h_branch = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            h_act,
        )
        self.time_scale = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One step of MONO-CfC.

        Args:
            x: input at this step [B, input_size].
            h: previous hidden state [B, hidden_size].
            dt: time delta (scalar or [B]).

        Returns:
            New hidden state [B, hidden_size].
        """
        x = torch.nan_to_num(x, nan=0.0)
        z = torch.cat([x, h], dim=-1)

        # Closed-form CfC solution (unchanged across mono modes).
        f = self.f_gate(z)
        g = self.g_branch(z)
        h_branch = self.h_branch(z)

        # Effective time constant: f * dt / tau
        if isinstance(dt, torch.Tensor):
            dt_b = dt.unsqueeze(-1) if dt.dim() < 1 else dt
            if dt_b.dim() == 1:
                dt_b = dt_b.unsqueeze(-1)
            tau_eff = torch.exp(-f * dt_b / torch.abs(self.time_scale))
        else:
            tau_eff = torch.exp(-f * float(dt) / torch.abs(self.time_scale))

        # h_t = tau * g + (1 - tau) * h_branch
        h_new = tau_eff * g + (1.0 - tau_eff) * h_branch
        return h_new


class MonoCfCStackedNetwork(nn.Module):
    """Stacked Mono-CfC network (PRD #10-116).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden dimension (per layer).
        output_size: output feature dimension.
        num_layers: number of stacked Mono-CfC cells.
        mono_mode: 'g_only', 'h_only', 'both', 'sigmoid' (per layer).
        return_sequences: if True, return per-timestep outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        mono_mode: str = "g_only",
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.mono_mode = mono_mode
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(MonoCfCCell(in_dim, hidden_size, mono_mode=mono_mode))

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass over the sequence.

        Args:
            x: input [B, T, input_size].
            h0: optional initial hidden state [num_layers, B, hidden].
            dt: optional per-step time deltas.
            mask: optional observed-feature mask.

        Returns:
            Output [B, T, output_size] if return_sequences else [B, output_size].
        """
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype,
            )

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            for t in range(seq_len):
                dt_t = select_step_delta(dt, t, batch_size, seq_len, x.device, x.dtype)
                input_mask, update_mask = select_step_mask(
                    mask, t, batch_size, seq_len, self.input_size, x.device, x.dtype
                )
                x_t = torch.nan_to_num(layer_input[:, t, :])
                if i == 0 and input_mask is not None:
                    x_t = x_t * input_mask
                h_candidate = cell(x_t, h_i, dt=dt_t)
                h_i = h_candidate if update_mask is None else update_mask * h_candidate + (1.0 - update_mask) * h_i
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat(
                [h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)],
                dim=0,
            )

        if self.return_sequences:
            return self.output_proj(layer_input)
        return self.output_proj(layer_input[:, -1, :])
