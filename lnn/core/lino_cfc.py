"""Linear-Nonlinear CfC (LiNo-CfC) (PRD #10-112, Round 150, 2026-06-15).

Implements a parallel linear stream + nonlinear CfC stream, summed
(or optionally concatenated) at the output. Inspired by the LiNo
framework (PKU/HK PolyU, Jan 2025) and DLinear (Zeng 2022 AAAI)
that separate linear and nonlinear modes for sequence modeling.

The key idea: a simple linear projection of x captures the smooth
linear trend, while the CfC captures the nonlinear residual.
Combined, they cover both modes::

    # Linear stream: per-step linear projection (no recurrence)
    h_lin = x @ W_lin + b_lin  # [B, T, hidden_size]

    # Nonlinear stream: standard CfC
    h_nl = CfCStackedNetwork(x)  # [B, T, hidden_size]

    # Combine: sum (LiNo spirit) or concat
    h = h_lin + h_nl  # or concat

This is structurally different from:
- **Conv preprocessing 137 (target-dep)**: 137 REPLACES x with
  the conv output. LiNo PRESERVES both linear and nonlinear streams.
- **TCC 149 (target-dep)**: TCC concats x with conv. LiNo sums
  linear projection with CfC.
- **Gated Input Skip 134 (strictly positive 13th)**: GIS is a
  single-step skip. LiNo is a parallel stream architecture.
- **Bidirectional CfC 144 (target-dep 5th)**: bidi processes the
  same input forward + backward. LiNo processes with two different
  model classes (linear vs nonlinear).

Risks:
- Linear stream's projection may be too weak (too few params) or
  too strong (essentially linear model). Need to match dimensions.
- Sum requires matching hidden sizes between linear and CfC streams.

Audit context (91-149):
- 13 strictly positive (preserves recurrent step + adds structure)
- 8 target-dep (input-side processing, bidi, SCRN, Time-Decay, TCC)
- 20 negatives (per-step modifications, alternatives, regularizers,
  bottlenecks, redundant info, replacements, long-α SCRN, Clockwork)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell


class LinearNonlinearCfCCell(nn.Module):
    """LiNo-CfC cell: parallel linear + nonlinear (CfC) streams.

    Args:
        input_size: input feature dimension D.
        hidden_size: hidden state dimension.
        mode: 'sum' (LiNo original) or 'concat'.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mode: str = "sum",
    ):
        super().__init__()
        assert mode in ("sum", "concat"), f"mode must be 'sum' or 'concat', got {mode}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode

        # Linear stream: per-step linear projection (no recurrence).
        self.linear_proj = nn.Linear(input_size, hidden_size)

        # Nonlinear stream: standard CfC cell.
        if mode == "sum":
            cfc_output_size = hidden_size
        else:  # concat
            cfc_output_size = hidden_size  # CfC outputs hidden_size
        self.cfc = CfCCell(input_size, cfc_output_size, n_tau=1)

        if mode == "concat":
            # When concatenating, we have 2 * hidden_size total. The final
            # head will project to output_size later, so we just store
            # both.
            self.concat_size = hidden_size * 2
        else:
            self.concat_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over the full sequence.

        Args:
            x: input of shape [B, T, input_size].
        Returns:
            Combined hidden states of shape [B, T, hidden_size] (sum)
            or [B, T, 2*hidden_size] (concat).
        """
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        # NaN handling: zero-fill input.
        x_clean = torch.nan_to_num(x, nan=0.0)

        # Linear stream: per-step linear projection.
        h_lin = self.linear_proj(x_clean)  # [B, T, hidden_size]

        # Nonlinear stream: standard CfC step.
        h_nl = torch.zeros(B, self.cfc.hidden_size, device=device, dtype=dtype)
        outputs_nl = []
        for t in range(T):
            x_t = x_clean[:, t, :]
            h_nl = self.cfc(x_t, h_nl)
            outputs_nl.append(h_nl)
        h_nl_seq = torch.stack(outputs_nl, dim=1)  # [B, T, hidden_size]

        # Combine.
        if self.mode == "sum":
            h = h_lin + h_nl_seq  # [B, T, hidden_size]
        else:  # concat
            h = torch.cat([h_lin, h_nl_seq], dim=-1)  # [B, T, 2*hidden_size]
        return h


class LinearNonlinearCfCStackedNetwork(nn.Module):
    """Stacked LiNo-CfC network (PRD #10-112).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden dimension (per layer).
        output_size: output feature dimension.
        num_layers: number of stacked LiNo-CfC cells.
        mode: 'sum' or 'concat' (per layer).
        return_sequences: if True, return per-timestep outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        mode: str = "sum",
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.mode = mode
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for li in range(num_layers):
            if li == 0:
                in_size = input_size
            else:
                # Output of previous layer: hidden_size if sum, 2*hidden if concat.
                in_size = hidden_size if mode == "sum" else 2 * hidden_size
            self.cells.append(
                LinearNonlinearCfCCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    mode=mode,
                )
            )

        # Final head.
        if mode == "sum":
            head_in = hidden_size
        else:
            head_in = 2 * hidden_size
        self.head = nn.Linear(head_in, output_size)

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
