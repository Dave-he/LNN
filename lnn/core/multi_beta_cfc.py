"""MultiBeta-CfC (Multi-Scale EMA Augmentation) (PRD #10-120, Round 158, 2026-06-15).

Augments CfC input with MULTIPLE parallel EMAs at different β
values, providing temporal context at multiple time-scales
simultaneously.

Mechanism::

    # Initialize: ema_k = x_0 for each k
    # At step t:
    ema_k,t[d] = beta_k * ema_k,t-1[d] + (1 - beta_k) * x_t[d]
    aug_x_t = f_concat(x_t, ema_1,t, ema_2,t, ..., ema_K,t)

This is structurally different from:
- Round 156 (EMA-X-CfC): K=1, scalar beta=0.9 — single-scale.
- Round 157 (LearnedBeta-CfC): K=1, per-feature learnable beta — single-scale.
- Round 129 (Multi-timescale ELM): multi-timescale with ELM, NEGATIVE.
- Round 76 (n_tau): multi-timescale tau in CfC recurrence, 7th winner.

MultiBeta extends single-beta EMA to multi-scale EMA, providing
temporal context at multiple time-scales simultaneously.

Variants:
- K=2: beta in {0.7, 0.95} (short, long)
- K=3: beta in {0.5, 0.9, 0.99} (short, medium, long)
- Modes: 'diff' (high-pass: ema-x) or 'concat' (low-pass: ema)

Audit context (91-157): 18 strictly positive + 13 target-dep +
28 negatives = 59 mechanism classes.

Risks:
- More EMAs = more parameters (3D or 4D input).
- High-pass with multiple cutoffs may be redundant.
- Long-window EMA may lag regime changes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.sequence_utils import select_step_delta, select_step_mask


class MultiBetaCfCCell(nn.Module):
    """MultiBeta-CfC cell: CfC with multi-scale EMA augmentation.

    Args:
        input_size: input feature dimension D.
        hidden_size: hidden state dimension.
        betas: list of β values for parallel EMAs.
        mode: 'diff' (aug_x = [x, ema_1-x, ..., ema_K-x]) or
            'concat' (aug_x = [x, ema_1, ..., ema_K]).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        betas: list[float],
        mode: str = "diff",
    ):
        super().__init__()
        assert mode in ("diff", "concat"), \
            f"mode must be 'diff' or 'concat', got {mode}"
        assert len(betas) >= 1, f"betas must have at least 1 element"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.betas = [float(b) for b in betas]
        self.K = len(betas)
        self.mode = mode

        # Aug input size: D (original) + K * D (EMAs) = (K+1) * D
        aug_input_size = (self.K + 1) * input_size

        # Standard CfC with augmented input.
        self.f_gate = nn.Sequential(
            nn.Linear(aug_input_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.g_branch = nn.Sequential(
            nn.Linear(aug_input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.h_branch = nn.Sequential(
            nn.Linear(aug_input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.time_scale = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        emas: list[torch.Tensor],
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """One step of MultiBeta-CfC.

        Args:
            x: input at this step [B, input_size].
            h: previous hidden state [B, hidden_size].
            emas: list of K previous EMA states, each [B, input_size].
            dt: time delta.

        Returns:
            (h_new, emas_new) tuple.
        """
        x = torch.nan_to_num(x, nan=0.0)
        emas = [torch.nan_to_num(e, nan=0.0) for e in emas]

        # Update all EMAs.
        emas_new = [
            beta * e + (1.0 - beta) * x
            for beta, e in zip(self.betas, emas)
        ]

        # Build augmented input.
        if self.mode == "concat":
            aug_x = torch.cat([x] + emas_new, dim=-1)
        else:  # diff
            aug_x = torch.cat([x] + [e - x for e in emas_new], dim=-1)

        z = torch.cat([aug_x, h], dim=-1)

        # Closed-form CfC solution.
        f = self.f_gate(z)
        g = self.g_branch(z)
        h_branch = self.h_branch(z)

        if isinstance(dt, torch.Tensor):
            dt_b = dt.unsqueeze(-1) if dt.dim() < 1 else dt
            if dt_b.dim() == 1:
                dt_b = dt_b.unsqueeze(-1)
            tau_eff = torch.exp(-f * dt_b / torch.abs(self.time_scale))
        else:
            tau_eff = torch.exp(-f * float(dt) / torch.abs(self.time_scale))

        h_new = tau_eff * g + (1.0 - tau_eff) * h_branch

        return h_new, emas_new


class MultiBetaCfCStackedNetwork(nn.Module):
    """Stacked MultiBeta-CfC network (PRD #10-120).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden dimension (per layer).
        output_size: output feature dimension.
        num_layers: number of stacked MultiBeta-CfC cells.
        betas: list of β values for parallel EMAs.
        mode: 'diff' or 'concat'.
        return_sequences: if True, return per-timestep outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        betas: list[float] = (0.7, 0.95),
        mode: str = "diff",
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.betas = [float(b) for b in betas]
        self.K = len(betas)
        self.mode = mode
        self.return_sequences = return_sequences

        # The first cell's input is the augmented original input.
        # Subsequent cells receive the previous layer's h output (no
        # multi-EMA augmentation on hidden state — that would be a
        # different signal).
        layer_in_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        self.cells = nn.ModuleList()
        for li in range(num_layers):
            self.cells.append(
                MultiBetaCfCCell(
                    layer_in_sizes[li], hidden_size,
                    betas=betas, mode=mode,
                )
            )

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        emas0: list[torch.Tensor] | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass over the sequence.

        Args:
            x: input [B, T, input_size].
            h0: optional initial hidden state.
            emas0: optional initial EMA states (one per β).
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
        if emas0 is None:
            # Initialize EMAs to first non-NaN x.
            x_first = torch.nan_to_num(x[:, 0, :], nan=0.0)
            emas0 = [x_first.clone() for _ in self.betas]

        h = h0
        emas_state = emas0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            # EMA state dim must match the cell's input dim.
            emas_i = [
                torch.zeros(
                    batch_size, cell.input_size, device=x.device, dtype=x.dtype,
                )
                for _ in self.betas
            ]
            if i == 0:
                emas_i = emas_state
            for t in range(seq_len):
                dt_t = select_step_delta(dt, t, batch_size, seq_len, x.device, x.dtype)
                input_mask, update_mask = select_step_mask(
                    mask, t, batch_size, seq_len, self.input_size, x.device, x.dtype
                )
                x_t = torch.nan_to_num(layer_input[:, t, :])
                if i == 0 and input_mask is not None:
                    x_t = x_t * input_mask
                h_new, emas_new = cell(x_t, h_i, emas_i, dt=dt_t)
                h_i = h_new if update_mask is None else update_mask * h_new + (1.0 - update_mask) * h_i
                outputs.append(h_i)
                emas_i = emas_new
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat(
                [h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)],
                dim=0,
            )

        if self.return_sequences:
            return self.output_proj(layer_input)
        return self.output_proj(layer_input[:, -1, :])
