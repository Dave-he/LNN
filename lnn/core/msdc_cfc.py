"""Multi-Scale Dilated Conv CfC (MSDC-CfC) (PRD #10-113, Round 151, 2026-06-15).

Implements a parallel multi-scale 1D dilated conv stream, summed
(or optionally concatenated) and then concatenated with x as
input to CfC. Inspired by WaveNet (Oord 2016), TCN (Bai 2018),
and Inception (Szegedy 2015) — multiple parallel filters at
different temporal scales capture different patterns simultaneously.

The key idea: a 1D conv with kernel=2 and dilation=d sees a
receptive field of (1 + d) steps. Running 3 parallel convs with
dilations 1/2/4 gives receptive fields 1, 3, 5 — covering local,
medium, and longer-range context in a single pass::

    # Three parallel 1D convs (kernel=2, dilations 1/2/4)
    c1 = Conv1D_d1(x_padded)  # [B, D, T], receptive field 1
    c2 = Conv1D_d2(x_padded)  # receptive field 3
    c3 = Conv1D_d4(x_padded)  # receptive field 5
    # Sum (or concat) to form context
    c = c1 + c2 + c3  # [B, D, T]
    # Concatenate with x
    aug_x = concat([x, c], dim=-1)  # [B, T, 2D]
    # Standard CfC with augmented input
    h = CfCCell(aug_x, h)

This is structurally different from:
- **TCC 149 (target-dep 8th)**: TCC uses single K (3/5/7), no
  dilation. MSDC uses three dilations, summed.
- **Conv preprocessing 137 (target-dep)**: 137 REPLACES x with
  conv. MSDC PRESERVES x (concats).
- **LiNo 150 (target-dep 9th)**: LiNo uses linear projection (no
  receptive field). MSDC uses conv (has receptive field).
- **WaveNet (Oord 2016)**: WaveNet stacks dilated convs SERIALLY
  with residual connections. MSDC runs them in PARALLEL and sums.

Risks:
- Three convs add params, may overfit on T=32 data.
- Conv smoothing is bad for noise (TCC 149 lost on random_irr).
- Multi-scale may not add value over single best scale.

Audit context (91-150):
- 13 strictly positive (preserves recurrent step + adds structure)
- 9 target-dep (input-side processing, bidi, SCRN, Time-Decay, TCC,
  LiNo)
- 20 negatives (per-step mods, alternatives, regularizers,
  bottlenecks, redundant info, replacements, long-α SCRN, Clockwork)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell


class MultiScaleDilatedConvCfCCell(nn.Module):
    """MSDC-CfC cell: parallel multi-scale dilated conv + CfC.

    Three parallel 1D convs with kernel=2, dilations 1/2/4, summed
    (or optionally concatenated) to form a context vector, then
    concatenated with x as input to CfC.

    Args:
        input_size: input feature dimension D.
        hidden_size: hidden state dimension.
        dilations: list of dilation values (default [1, 2, 4]).
        combine: 'sum' (sum dilations) or 'concat' (concat dilations).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dilations: list[int] | None = None,
        combine: str = "sum",
    ):
        super().__init__()
        assert combine in ("sum", "concat"), f"combine must be 'sum' or 'concat', got {combine}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dilations = dilations if dilations is not None else [1, 2, 4]
        self.combine = combine

        # Parallel 1D convs, one per dilation. kernel_size=2.
        self.convs = nn.ModuleList()
        for d in self.dilations:
            self.convs.append(
                nn.Conv1d(
                    in_channels=input_size,
                    out_channels=input_size,
                    kernel_size=2,
                    stride=1,
                    padding=0,
                    dilation=d,
                    bias=True,
                )
            )

        # Output dim of context: input_size (sum) or len(dilations) * input_size (concat).
        if combine == "sum":
            context_dim = input_size
        else:
            context_dim = input_size * len(self.dilations)
        self.context_dim = context_dim

        # CfC cell takes aug_x = concat([x, c]) → 2 * input_size.
        aug_input_size = input_size + context_dim
        self.cfc = CfCCell(aug_input_size, hidden_size, n_tau=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over the full sequence.

        Args:
            x: input of shape [B, T, input_size].
        Returns:
            Hidden states of shape [B, T, hidden_size].
        """
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        # NaN handling: zero-fill input.
        x_clean = torch.nan_to_num(x, nan=0.0)

        # Transpose to [B, D, T] for Conv1d.
        x_conv = x_clean.transpose(1, 2)

        # Apply each dilated conv.
        conv_outputs = []
        for d, conv in zip(self.dilations, self.convs):
            # Causal left-pad: pad (d, 0) so position t sees x_{t-d..t}.
            x_padded = F.pad(x_conv, (d, 0))
            c = conv(x_padded)  # [B, D, T]
            conv_outputs.append(c)

        # Combine (sum or concat).
        if self.combine == "sum":
            c = torch.stack(conv_outputs, dim=0).sum(dim=0)  # [B, D, T]
        else:
            c = torch.cat(conv_outputs, dim=1)  # [B, len(dilations) * D, T]
        c = c.transpose(1, 2)  # [B, T, context_dim]

        # Concat with x.
        aug_x = torch.cat([x_clean, c], dim=-1)  # [B, T, 2D or 4D]

        # Standard CfC with augmented input.
        h = torch.zeros(B, self.cfc.hidden_size, device=device, dtype=dtype)
        outputs = []
        for t in range(T):
            aug_x_t = aug_x[:, t, :]
            h = self.cfc(aug_x_t, h)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class MultiScaleDilatedConvCfCStackedNetwork(nn.Module):
    """Stacked MSDC-CfC network (PRD #10-113).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden dimension (per layer).
        output_size: output feature dimension.
        num_layers: number of stacked MSDC-CfC cells.
        dilations: list of dilation values (default [1, 2, 4]).
        combine: 'sum' or 'concat' (per layer).
        return_sequences: if True, return per-timestep outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dilations: list[int] | None = None,
        combine: str = "sum",
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dilations = dilations if dilations is not None else [1, 2, 4]
        self.combine = combine
        self.return_sequences = return_sequences

        # Determine input size for each layer.
        # Layer 0: input_size (raw).
        # Layer i>0: input_size (since output of CfC is hidden_size,
        # but the MSDC cell needs raw x at every step... no wait,
        # we feed the cell's output as input to the next layer).
        # Actually the design: each MSDC cell takes x (from previous
        # layer or input) and computes its own context. So layer i
        # input is layer i-1's output (which is hidden_size).
        layer_in_sizes = [input_size] + [hidden_size] * (num_layers - 1)

        self.cells = nn.ModuleList()
        for li in range(num_layers):
            self.cells.append(
                MultiScaleDilatedConvCfCCell(
                    input_size=layer_in_sizes[li],
                    hidden_size=hidden_size,
                    dilations=self.dilations,
                    combine=combine,
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
