"""Clockwork CfC (PRD #10-109, Round 147, 2026-06-15).

Implements the Clockwork RNN idea from Koutnik et al. 2014
("A Clockwork RNN", ICML 2014 / arXiv:1402.3558) applied to CfC.

The original CW-RNN partitions the hidden state into K modules,
each updating only at its assigned period (2^k). The
motivation is to capture multi-timescale dynamics:

  - Module 0: period 1 (every step)        — "fast"  / local
  - Module 1: period 2 (every 2 steps)     — "medium" / transition
  - Module 2: period 4 (every 4 steps)     — "slow"  / regime
  - Module 3: period 8 (every 8 steps)     — "very slow" / global

When a module does NOT update on step t, it **carries forward** its
previous h (preserves h). All modules' outputs are concatenated to
form the full hidden state.

This is a STRUCTURAL PARTITION (not a per-step modification) and
PRESERVES both x and h, per the 91-146 audit pattern:
- 6 of 6 target-dep winners (LN 135, conv 137, GLU+skip 139,
  decoupled/IndRNN 143, bidi_concat 144, scrn_05 146) preserve x.
- 19 negatives include input-side REPLACEMENTS (diff_only 145,
  long-α SCRN 146) and per-step modifications (ATC 141, MI 142,
  zoneout 136, etc.).
- Different from SCRN 146 (parallel stream) and ELM 129 (per-step
  multi-timescale, NEGATIVE).

Risks:
- Training instability for slow modules: K=4 with period 8 gives
  only 4 gradient updates per T=32 sequence.
- Carry-forward h creates discrete "phase changes" that may not
  align with smooth data.
- NaN handling: zero-fill input per step.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell


class ClockworkCfCCell(nn.Module):
    """Clockwork CfC cell: hidden partitioned into K modules, each with period 2^k.

    Args:
        input_size: input feature dimension D.
        hidden_size: total hidden dim (sum of all module sizes).
        num_modules: K (number of partitions).
        module_sizes: explicit sizes for each module. If None, auto-equal.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_modules: int = 3,
        module_sizes: list[int] | None = None,
    ):
        super().__init__()
        assert num_modules >= 1, f"num_modules must be >= 1, got {num_modules}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_modules = num_modules

        if module_sizes is None:
            # Auto-equal split: each module gets hidden_size // K (last `rem` modules absorb the remainder).
            base = hidden_size // num_modules
            rem = hidden_size - base * num_modules
            self.module_sizes = [base] * num_modules
            # Distribute remainder: add 1 to the last `rem` modules.
            for i in range(rem):
                self.module_sizes[num_modules - 1 - i] += 1
        else:
            assert sum(module_sizes) == hidden_size, f"module_sizes must sum to hidden_size, got {sum(module_sizes)}"
            self.module_sizes = list(module_sizes)
        # Periods: 2^0, 2^1, ..., 2^{K-1}
        self.periods = [2 ** k for k in range(num_modules)]

        # Each module has its own CfC cell.
        self.cells = nn.ModuleList([
            CfCCell(input_size, module_size, n_tau=1)
            for module_size in self.module_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over the full sequence.

        Args:
            x: input of shape [B, T, input_size].
        Returns:
            Concatenated module hidden states of shape [B, T, hidden_size].
        """
        B, T, _ = x.shape
        # Initialize per-module hidden state.
        hs = [
            torch.zeros(B, module_size, device=x.device, dtype=x.dtype)
            for module_size in self.module_sizes
        ]
        outputs = []
        for t in range(T):
            x_t = torch.nan_to_num(x[:, t, :], nan=0.0)
            module_outs = []
            for k, cell in enumerate(self.cells):
                if t % self.periods[k] == 0:
                    # Module k updates on this step.
                    hs[k] = cell(x_t, hs[k])
                # else: carry forward h (preserves h)
                module_outs.append(hs[k])
            outputs.append(torch.cat(module_outs, dim=-1))
        return torch.stack(outputs, dim=1)  # [B, T, hidden]


class ClockworkCfCStackedNetwork(nn.Module):
    """Stacked Clockwork CfC network (PRD #10-109).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden dimension (per layer, partitioned into K modules).
        output_size: output feature dimension.
        num_layers: number of stacked Clockwork cells.
        num_modules: K partitions per layer.
        module_sizes: explicit module sizes (default: auto-equal).
        return_sequences: if True, return per-timestep outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        num_modules: int = 3,
        module_sizes: list[int] | None = None,
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_modules = num_modules
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for li in range(num_layers):
            if li == 0:
                in_size = input_size
            else:
                in_size = hidden_size  # output of previous layer (concatenated modules)
            self.cells.append(
                ClockworkCfCCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    num_modules=num_modules,
                    module_sizes=module_sizes,
                )
            )

        # Final head.
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input of shape [B, T, input_size].
        Returns:
            Output of shape [B, T, output_size] if return_sequences else [B, output_size].
        """
        layer_input = x
        for cell in self.cells:
            layer_input = cell(layer_input)
        out = self.head(layer_input)
        if self.return_sequences:
            return out
        return out[:, -1, :]
