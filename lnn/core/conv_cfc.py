"""1D Convolutional Input Preprocessing for CfC (PRD #10-99, 2026-06-15).

Implements a 1D causal convolution preprocessing step applied to
the input BEFORE the CfC recurrent step. Inspired by ConvLSTM (Shi
et al. 2015) and similar convolutional-recurrent architectures.

The mechanism::

    x_conv = Conv1d_causal(x)        # [B, T, D_in] -> [B, T, D_conv]
    h_t = cf_c_step(x_conv[t], h_{t-1})

The 1D causal convolution has kernel size k and uses causal padding
so the output at time t only depends on inputs at times ≤ t.

Why this should work in 1D:
- Captures local temporal patterns before the recurrent step.
- Preserves the CfC recurrent step entirely (input-side only).
- Per the 91-136 audit, mechanisms that ADD useful input-side
  processing to the recurrent step (rather than REPLACE it) are
  STRICTLY POSITIVE (13 winners including QuITE+MoE round 103).

This module contains:

- **ConvCfCCell**: standard 3-branch CfC cell with 1D conv
  preprocessing of the input.
- **ConvCfCStackedNetwork**: stack of Conv-CfC cells.

Risks:
- 1D conv might be redundant with CfC's W·h (which already captures
  local patterns).
- 1D conv might be redundant with QuITE (which already does input
  processing in round 103).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvCfCCell(nn.Module):
    """Conv-CfC cell (PRD #10-99).

    Standard 3-branch CfC recurrent step preceded by a 1D causal
    convolution applied to the input::

        x_conv = Conv1d_causal(x)         # 1D conv preprocessing
        combined = [x_conv, h]            # standard CfC
        f = sigmoid(W_f combined)
        g = tanh(W_g combined)
        h_out = tanh(W_h combined)
        decay = sigmoid(-f * time_scale)
        h_new = decay * g + (1-decay) * h_out

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        kernel_size: 1D conv kernel size (default 3).
        conv_init_scale: std of conv weight init (default 0.1).
        time_scale_init: initial value of CfC's time scale parameter.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int = 3,
        conv_init_scale: float = 0.1,
        time_scale_init: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv_init_scale = conv_init_scale

        # 1D causal conv preprocessing.
        # Causal padding: pad (kernel_size - 1) on the left.
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,   # same number of channels
            kernel_size=kernel_size,
            padding=0,                 # we'll do manual causal padding
            bias=False,
        )
        # Init small to start close to identity.
        with torch.no_grad():
            self.conv.weight.normal_(mean=0.0, std=conv_init_scale)
            # Make the conv start as identity (last position = eye, others = 0).
            # For causal conv, the last position of the window is the current timestep.
            # We want conv(x) at the last position to give x_t, so use an identity matrix
            # in the last kernel slice.
            last = kernel_size - 1
            self.conv.weight.zero_()
            self.conv.weight[:, :, last] = torch.eye(input_size)

        # Standard 3-branch CfC step (operates on conv'd input).
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

    def conv_norm(self) -> float:
        """Frobenius norm of conv weights (diagnostic)."""
        return float(self.conv.weight.norm().item())

    def _causal_conv(self, x_t_prev: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Apply 1D causal conv to a window of (kernel_size) timesteps.

        Args:
            x_t_prev: previous (kernel_size - 1) timesteps, [B, kernel_size-1, D].
            x_t: current timestep, [B, D].
        Returns:
            conv output at current timestep, [B, D].
        """
        # Stack previous and current: [B, kernel_size, D]
        # Need to convert to [B, D, kernel_size] for Conv1d.
        if x_t_prev is not None and x_t_prev.shape[1] > 0:
            window = torch.cat([x_t_prev, x_t.unsqueeze(1)], dim=1)  # [B, k, D]
        else:
            window = x_t.unsqueeze(1)  # [B, 1, D]
        # Conv1d expects [B, C, L]. We treat D as C and kernel_size as L.
        window = window.transpose(1, 2)  # [B, D, kernel_size]
        out = self.conv(window)  # [B, D, 1]
        return out.squeeze(-1)  # [B, D]

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        x_t_prev: torch.Tensor | None = None,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One recurrent step with 1D conv preprocessing.

        Args:
            x_t: input at current timestep, [B, input_size].
            h: hidden state, [B, hidden_size].
            x_t_prev: previous (kernel_size - 1) timesteps, [B, kernel_size-1, input_size].
                If None, treated as zeros.
            dt: scalar time delta.
        Returns:
            New hidden state, [B, hidden_size].
        """
        # Apply 1D causal conv.
        if self.kernel_size > 1:
            if x_t_prev is None:
                x_t_prev = torch.zeros(
                    x_t.shape[0], self.kernel_size - 1, self.input_size,
                    device=x_t.device, dtype=x_t.dtype,
                )
            x_conv = self._causal_conv(x_t_prev, x_t)
        else:
            x_conv = x_t

        # Standard 3-branch CfC step.
        combined = torch.cat([x_conv, h], dim=-1)
        f = self.f_gate(combined)
        g = self.g_branch(combined)
        h_out = self.h_branch(combined)
        decay = torch.sigmoid(-f * self.time_scale * dt)
        h_new = decay * g + (1.0 - decay) * h_out
        return h_new


class ConvCfCStackedNetwork(nn.Module):
    """Stacked Conv-CfC cells (PRD #10-99).

    Each layer is a ``ConvCfCCell``. The conv preprocessing is
    applied per layer to the input. The output is the head
    projection of the last layer's hidden state.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        output_size: output feature dimension.
        num_layers: number of stacked cells.
        kernel_size: 1D conv kernel size.
        conv_init_scale: std of conv weight init.
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
        kernel_size: int = 3,
        conv_init_scale: float = 0.1,
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
        self.kernel_size = kernel_size
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else hidden_size
            self.cells.append(
                ConvCfCCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                    conv_init_scale=conv_init_scale,
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
        # Keep a window of (kernel_size - 1) previous inputs per layer.
        k = self.kernel_size
        windows = [
            torch.zeros(B, k - 1, self.cells[li].input_size, device=x.device, dtype=x.dtype)
            for li in range(self.num_layers)
        ]
        outputs = []
        for t in range(T):
            inp = torch.nan_to_num(x[:, t, :], nan=0.0)
            for li, cell in enumerate(self.cells):
                # Apply 1D causal conv preprocessing + CfC step.
                h_new = cell(inp, h[li], x_t_prev=windows[li] if k > 1 else None)
                h[li] = h_new
                # Update window: shift left, append current input.
                if k > 1:
                    windows[li] = torch.cat([windows[li][:, 1:, :], inp.unsqueeze(1)], dim=1)
                # For subsequent layers, the input is the previous layer's hidden.
                inp = h[li]
            outputs.append(self.head(h[-1]))
        outputs = torch.stack(outputs, dim=1)
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]
