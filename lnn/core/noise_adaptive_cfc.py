"""Noise-Adaptive Decay CfC (CfC-NAD).

Implements a CfC variant whose mixing dynamics are conditioned on a running
estimate of input noise. Motivation comes from the 2026-05-26 paper "Comparative
Analysis of Liquid Neural Networks and LSTM for Sequential Pattern Recognition:
Robustness, Efficiency, and Clinical Utility" which reports that vanilla CfC
already beats LSTM under noise, but uses a single time-scale gate that ignores
local heteroscedasticity.

The cell maintains a streaming EMA of the squared first-difference of the
input as a cheap online noise score. A learnable noise gate then biases the
fresh CfC candidate toward the previous hidden state when the local noise is
high — effectively a heteroscedastic low-pass on h.

Memory footprint stays O(1): the only added state per sample is one EMA
vector of shape [batch, input_size] (the noise score) and one previous-input
vector for the first-difference proxy.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.sequence_utils import select_step_delta, select_step_mask


class NoiseAdaptiveCfCCell(nn.Module):
    """Closed-form Continuous-time cell with heteroscedastic gating.

    The base mixing follows the standard CfC formulation:
        h_cfc = decay * g(x, h) + (1 - decay) * h_branch(x, h)
        decay = sigmoid(-f_gate(x, h, noise) * time_scale * dt)

    A separate noise gate then interpolates between h_cfc and the previous
    hidden state:
        h_new = (1 - noise_gate) * h_cfc + noise_gate * h_prev

    where ``noise_gate = sigmoid(W_noise @ noise_score)`` reads off the streaming
    noise EMA of the input. Under low noise the cell reduces to vanilla CfC;
    under high noise it holds the prior state, suppressing spurious updates.
    """

    def __init__(self, input_size: int, hidden_size: int, noise_beta: float = 0.9) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # EMA smoothing factor for the noise score (closer to 1 = smoother).
        self.noise_beta = float(noise_beta)

        # f_gate receives [x, h, noise_score] so the gate can be modulated
        # directly by the per-feature noise estimate.
        gate_in = input_size + hidden_size + input_size
        self.f_gate = nn.Sequential(
            nn.Linear(gate_in, hidden_size),
            nn.Sigmoid(),
        )
        self.g_branch = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.h_branch = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        # Per-channel learnable time scale (matches the baseline CfC layout).
        self.time_scale = nn.Parameter(torch.ones(hidden_size))
        # Maps per-feature noise scores to a per-channel retention gate.
        self.noise_gate_proj = nn.Linear(input_size, hidden_size)
        # Initialise noise gate near zero — start as vanilla CfC; let training
        # decide how much heteroscedastic retention to use.
        nn.init.zeros_(self.noise_gate_proj.weight)
        nn.init.zeros_(self.noise_gate_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        noise_score: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        gate_in = torch.cat([x_t, h, noise_score], dim=-1)
        f = self.f_gate(gate_in)
        combined = torch.cat([x_t, h], dim=-1)
        g = self.g_branch(combined)
        h_branch = self.h_branch(combined)
        decay = torch.sigmoid(-f * self.time_scale * dt)
        h_cfc = decay * g + (1.0 - decay) * h_branch
        # Heteroscedastic blend: pull toward h under high noise.
        noise_gate = torch.sigmoid(self.noise_gate_proj(noise_score))
        h_new = (1.0 - noise_gate) * h_cfc + noise_gate * h
        return h_new


class NoiseAdaptiveCfCNetwork(nn.Module):
    """Stacked Noise-Adaptive CfC network with the standard (x, dt, mask) API.

    The forward signature matches :class:`lnn.core.cfc.CfCNetwork` so the network
    is a drop-in replacement in :class:`lnn.core.trainer.Trainer`.

    Args:
        input_size: Feature dimension of x.
        hidden_size: Hidden state width per layer.
        output_size: Output projection dimension.
        num_layers: Number of stacked cells.
        return_sequences: If False, returns only the last step.
        noise_beta: EMA smoothing factor for the noise score (default 0.9).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
        noise_beta: float = 0.9,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(NoiseAdaptiveCfCCell(in_dim, hidden_size, noise_beta=noise_beta))

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs: list[torch.Tensor] = []
            h_i = h[i]
            # Streaming noise estimate for this layer's input stream.
            prev_x = torch.zeros_like(layer_input[:, 0, :])
            noise_ema = torch.zeros_like(layer_input[:, 0, :])
            beta = cell.noise_beta if hasattr(cell, "noise_beta") else 0.9
            for t in range(seq_len):
                dt_t = select_step_delta(dt, t, batch_size, seq_len, x.device, x.dtype)
                input_mask, update_mask = select_step_mask(
                    mask, t, batch_size, seq_len, cell.input_size, x.device, x.dtype
                )
                x_t = torch.nan_to_num(layer_input[:, t, :])
                if i == 0 and input_mask is not None:
                    x_t = x_t * input_mask

                # Update the streaming noise score (squared first difference EMA).
                diff_sq = (x_t - prev_x) ** 2 if t > 0 else torch.zeros_like(x_t)
                noise_ema = beta * noise_ema + (1.0 - beta) * diff_sq
                prev_x = x_t

                h_candidate = cell(x_t, h_i, noise_score=noise_ema, dt=dt_t)
                if update_mask is None:
                    h_i = h_candidate
                else:
                    h_i = update_mask * h_candidate + (1.0 - update_mask) * h_i
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat(
                [
                    h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0)
                    for j in range(self.num_layers)
                ],
                dim=0,
            )

        if self.return_sequences:
            return self.output_proj(layer_input)
        return self.output_proj(layer_input[:, -1, :])


__all__ = ["NoiseAdaptiveCfCCell", "NoiseAdaptiveCfCNetwork"]
