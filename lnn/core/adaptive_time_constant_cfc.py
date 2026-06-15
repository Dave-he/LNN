"""Adaptive Time-Constant CfC (PRD #10-103, 2026-06-15).

Makes the per-neuron time constant input-conditional instead of
fixed. Inspired by Adaptive Computation Time (Graves 2016) and
multi-timescale RNNs.

The mechanism::

    tau = softplus(W_tau [x_t, h]) + 1.0     # [B, hidden_size]
    f = sigmoid(W_f [x_t, h])                # [B, hidden_size]
    g = tanh(W_g [x_t, h])
    h_out = tanh(W_h [x_t, h])
    decay = sigmoid(-f * tau)                # adaptive decay
    h_new = decay * g + (1-decay) * h_out

Why this should work in 1D:
- CfC's time_scale is per-neuron but FIXED across timesteps.
- An input-conditional tau lets the cell adapt its time constant
  per timestep (fast on regime switches, slow on smooth data).
- Per the 91-140 audit, mechanisms that ADD useful structure to
  the recurrent step (rather than REPLACE it) are STRICTLY
  POSITIVE (13 winners).

This module contains:

- **AdaptiveTimeConstantCfCCell**: standard 3-branch CfC cell with
  input-conditional time constant.
- **AdaptiveTimeConstantCfCStackedNetwork**: stack of ATC-CfC cells.

Risks:
- tau computation adds parameters that could overfit on noisy data.
- The f-gate might already provide enough adaptation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveTimeConstantCfCCell(nn.Module):
    """CfC cell with input-conditional time constant.

    Standard 3-branch CfC recurrent step with an input-conditional
    time constant::

        tau = softplus(W_tau [x_t, h]) + 1.0     # [B, hidden_size]
        f = sigmoid(W_f [x_t, h])                # [B, hidden_size]
        g = tanh(W_g [x_t, h])
        h_out = tanh(W_h [x_t, h])
        decay = sigmoid(-f * tau)                # adaptive decay
        h_new = decay * g + (1-decay) * h_out

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        mode: how to compute the time constant:
            - "concat": from concat [x, h] (default)
            - "input": from input only
        time_scale_init: initial value of the bias for tau (the
            linear transformation is initialized to output ~0 so
            softplus(0) + 1 = log(2) + 1 ≈ 1.69 initially).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mode: str = "concat",
        time_scale_init: float = 1.0,
    ):
        super().__init__()
        assert mode in ("concat", "input"), f"mode must be in (concat, input), got {mode}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode

        # Time-constant network.
        if mode == "concat":
            tau_in = input_size + hidden_size
        else:  # input
            tau_in = input_size
        self.tau_net = nn.Linear(tau_in, hidden_size)
        # Initialize so softplus(W * x + b) + 1 ≈ time_scale_init at init.
        # softplus(z) = log(1 + exp(z)).
        # If we set bias = softplus_inv(time_scale_init - 1.0) and weights
        # to 0, then tau = softplus(bias) + 1 = time_scale_init.
        with torch.no_grad():
            # softplus_inv(y) = log(exp(y) - 1). For y=0, this is -inf, so
            # we clamp the input to expm1 to a minimum to avoid -inf.
            inv_softplus = torch.log(
                torch.expm1(torch.tensor(float(time_scale_init) - 1.0)).clamp(min=1e-3)
            )
            self.tau_net.bias.fill_(inv_softplus.item())
            self.tau_net.weight.zero_()

        # Standard 3-branch CfC step.
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

    def tau(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute the time constant, always >= 1.0."""
        if self.mode == "concat":
            inp = torch.cat([x_t, h], dim=-1)
        else:  # input
            inp = x_t
        return torch.nn.functional.softplus(self.tau_net(inp)) + 1.0

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One recurrent step with adaptive time constant.

        Args:
            x_t: input at current timestep, [B, input_size].
            h: hidden state, [B, hidden_size].
            dt: scalar time delta.
        Returns:
            New hidden state, [B, hidden_size].
        """
        # Compute adaptive time constant.
        tau = self.tau(x_t, h)  # [B, hidden_size], >= 1.0

        # Standard 3-branch CfC step.
        combined = torch.cat([x_t, h], dim=-1)
        f = self.f_gate(combined)
        g = self.g_branch(combined)
        h_out = self.h_branch(combined)
        decay = torch.sigmoid(-f * tau * dt)
        h_new = decay * g + (1.0 - decay) * h_out
        return h_new


class AdaptiveTimeConstantCfCStackedNetwork(nn.Module):
    """Stacked ATC-CfC cells (PRD #10-103).

    Each layer is an ``AdaptiveTimeConstantCfCCell``. The ATC is
    applied per layer. The output is the head projection of the
    last layer's hidden state.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        output_size: output feature dimension.
        num_layers: number of stacked cells.
        mode: ATC mode ("concat" or "input").
        time_scale_init: initial value of the time constant.
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
                AdaptiveTimeConstantCfCCell(
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
