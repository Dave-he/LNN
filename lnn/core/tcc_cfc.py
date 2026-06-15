"""Temporal Conv Concat CfC (PRD #10-111, Round 149, 2026-06-15).

Implements a parallel 1D temporal convolution stream concatenated
with the input to the CfC cell. The conv provides a local context
window (size K) at each step, and the CfC sees the augmented input.

The key idea: at each step t, the conv computes c_t from the past
K observations, and the CfC's input is concat(x_t, c_t).

This is structurally different from:
- **Conv preprocessing 137 (target-dep)**: 137 REPLACES x with the
  conv output. TCC PRESERVES x and ADDS c as a parallel stream.
- **QuITE 102 (strictly positive)**: QuITE uses attention to embed
  irregular TS. TCC uses simple 1D conv.
- **Gated Input Skip 134 (strictly positive 13th)**: GIS is a
  single-step skip (skip = 1). TCC uses multi-step kernel (K=3, 5, etc.).

Risks:
- Doubles the input dimension (D → 2D), so the first linear layer
  has more parameters. May overfit on small data.
- Conv smoothing might hurt high-frequency information.

Audit context (91-148):
- 13 strictly positive (preserves recurrent step + adds structure)
- 7 target-dep (input-side processing that preserves x OR bidi OR
  SCRN α=0.5 OR Time-Decay γ=0.5)
- 20 negatives (per-step modifications, alternatives, regularizers,
  bottlenecks, redundant info, replacements, long-α SCRN, Clockwork)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell


class TemporalConvConcatCfCCell(nn.Module):
    """TCC-CfC cell: 1D conv parallel stream concatenated with x as input.

    Args:
        input_size: input feature dimension D.
        hidden_size: hidden state dimension.
        kernel_size: K — number of past observations the conv sees.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        assert kernel_size >= 1, f"kernel_size must be >= 1, got {kernel_size}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        # Causal 1D conv over the time axis.
        # Left-pad with (K-1, 0) so position t sees only x_{t-K+1..t}.
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            kernel_size=kernel_size,
            stride=1,
            padding=0,  # we pad manually for causality
            bias=True,
        )

        # The CfC cell takes the augmented input [x, c] (dim 2*D).
        self.cfc = CfCCell(input_size * 2, hidden_size, n_tau=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over the full sequence.

        Args:
            x: input of shape [B, T, input_size].
        Returns:
            Hidden states of shape [B, T, hidden_size].
        """
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        # NaN handling: zero-fill input per step.
        x_clean = torch.nan_to_num(x, nan=0.0)

        # Causal 1D conv: x_clean is [B, T, D] -> [B, D, T] for Conv1d.
        x_conv = x_clean.transpose(1, 2)  # [B, D, T]
        # Left-pad with (K-1, 0) zeros.
        x_padded = nn.functional.pad(x_conv, (self.kernel_size - 1, 0))
        # Apply conv: [B, D, T].
        c = self.conv(x_padded)  # [B, D, T]
        c = c.transpose(1, 2)  # [B, T, D]

        # Concatenate with x: aug_x = [x, c] of shape [B, T, 2D].
        aug_x = torch.cat([x_clean, c], dim=-1)  # [B, T, 2D]

        # Standard CfC step.
        h = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)
        outputs = []
        for t in range(T):
            aug_x_t = aug_x[:, t, :]
            h = self.cfc(aug_x_t, h)
            outputs.append(h)
        return torch.stack(outputs, dim=1)  # [B, T, hidden_size]


class TemporalConvConcatCfCStackedNetwork(nn.Module):
    """Stacked TCC-CfC network (PRD #10-111).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden dimension (per layer).
        output_size: output feature dimension.
        num_layers: number of stacked TCC-CfC cells.
        kernel_size: K — conv kernel size (same for all layers).
        return_sequences: if True, return per-timestep outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        kernel_size: int = 3,
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
        for li in range(num_layers):
            if li == 0:
                in_size = input_size
            else:
                in_size = hidden_size  # output of previous layer
            self.cells.append(
                TemporalConvConcatCfCCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                )
            )

        # Final head.
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input of shape [B, T, input_size].
        Returns:
            Output of shape [B, T, output_size] if return_sequences else
            [B, output_size].
        """
        layer_input = x
        for cell in self.cells:
            layer_input = cell(layer_input)
        out = self.head(layer_input)
        if self.return_sequences:
            return out
        return out[:, -1, :]
