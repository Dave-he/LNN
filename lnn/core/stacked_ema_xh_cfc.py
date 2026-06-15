"""Stacked-EMA-XH-CfC (Input + Hidden State EMA) (PRD #10-123, Round 161, 2026-06-15).

Combines input-side EMA (rounds 156-158) with hidden-state EMA
(rounds 159-160) into a single CfC cell. Augments BOTH x and h
with multi-scale EMAs at different β values.

Mechanism::

    # Input-side EMAs:
    ema_x_k,t = beta_x_k * ema_x_k,t-1 + (1 - beta_x_k) * x_t
    aug_x_t = [x_t, ema_x_1,t, ..., ema_x_Kx,t]

    # Hidden-state EMAs:
    ema_h_k,t = beta_h_k * ema_h_k,t-1 + (1 - beta_h_k) * h_t
    aug_h_t = [h_t, ema_h_1,t, ..., ema_h_Kh,t]

    # Combined:
    z_t = cat(aug_x_t, aug_h_t)

This is the FULL STACK of all 5 EMA mechanisms from rounds 156-160.
The goal is BEST of BOTH worlds:
- x-side: best for structured (-65% round 158)
- h-side: best for structured (-77% round 159) and sin (-32% round 160)

Combining them may give -32% sin AND -77% structured simultaneously.

Variants (4 conds):
- sx_xh_diff_1_1:   Kx=1 (β=0.9) + Kh=1 (β=0.9), both diff
- sx_xh_diff_3_2:   Kx=3 (β ∈ {0.5, 0.9, 0.99}) + Kh=2 (β ∈ {0.7, 0.95}), both diff
- sx_xh_concat_2_2: Kx=2 (β ∈ {0.7, 0.95}) + Kh=2 (β ∈ {0.7, 0.95}), both concat
- sx_xh_best:       Kx=3 diff (round 158 best) + Kh=2 diff (round 160 best)

Audit context (91-160): 22 strictly positive + 17 target-dep +
30 negatives = 69 mechanism classes.

Risks:
- Combined X+H has 4D-5D+ input (vs 2D-3D before) — more parameters.
- Both mechanisms may overfit on small data.
- Diminishing returns from combining 2 positive mechanisms.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.sequence_utils import select_step_delta, select_step_mask


class StackedEMAXHCfCCell(nn.Module):
    """Stacked-EMA-XH-CfC cell: CfC with BOTH x-side and h-side EMA augmentation.

    Args:
        input_size: input feature dimension D (for this layer).
        hidden_size: hidden state dimension.
        betas_x: list of β values for x-side EMAs.
        betas_h: list of β values for h-side EMAs.
        mode_x: 'diff' or 'concat' for x-side.
        mode_h: 'diff' or 'concat' for h-side.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        betas_x: list[float],
        betas_h: list[float],
        mode_x: str = "diff",
        mode_h: str = "diff",
    ):
        super().__init__()
        assert mode_x in ("diff", "concat"), f"mode_x must be 'diff' or 'concat', got {mode_x}"
        assert mode_h in ("diff", "concat"), f"mode_h must be 'diff' or 'concat', got {mode_h}"
        assert len(betas_x) >= 1, f"betas_x must have at least 1 element"
        assert len(betas_h) >= 1, f"betas_h must have at least 1 element"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.betas_x = [float(b) for b in betas_x]
        self.betas_h = [float(b) for b in betas_h]
        self.Kx = len(betas_x)
        self.Kh = len(betas_h)
        self.mode_x = mode_x
        self.mode_h = mode_h

        # Aug input size: D (original) + Kx * D (x-EMAs) = (Kx+1) * D
        aug_input_size = (self.Kx + 1) * input_size
        # Aug hidden size: H (original) + Kh * H (h-EMAs) = (Kh+1) * H
        aug_hidden_size = (self.Kh + 1) * hidden_size

        # Standard CfC with augmented input and hidden state.
        self.f_gate = nn.Sequential(
            nn.Linear(aug_input_size + aug_hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.g_branch = nn.Sequential(
            nn.Linear(aug_input_size + aug_hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.h_branch = nn.Sequential(
            nn.Linear(aug_input_size + aug_hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.time_scale = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        emas_x: list[torch.Tensor],
        emas_h: list[torch.Tensor],
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """One step of Stacked-EMA-XH-CfC.

        Args:
            x: input at this step [B, input_size].
            h: previous hidden state [B, hidden_size].
            emas_x: list of Kx previous x-EMAs, each [B, input_size].
            emas_h: list of Kh previous h-EMAs, each [B, hidden_size].
            dt: time delta.

        Returns:
            (h_new, emas_x_new, emas_h_new) tuple.
        """
        x = torch.nan_to_num(x, nan=0.0)
        h = torch.nan_to_num(h, nan=0.0)
        emas_x = [torch.nan_to_num(e, nan=0.0) for e in emas_x]
        emas_h = [torch.nan_to_num(e, nan=0.0) for e in emas_h]

        # Update x-side EMAs.
        emas_x_new = [
            beta * e + (1.0 - beta) * x
            for beta, e in zip(self.betas_x, emas_x)
        ]
        # Update h-side EMAs.
        emas_h_new = [
            beta * e + (1.0 - beta) * h
            for beta, e in zip(self.betas_h, emas_h)
        ]

        # Build augmented x and h.
        if self.mode_x == "concat":
            aug_x = torch.cat([x] + emas_x_new, dim=-1)
        else:  # diff
            aug_x = torch.cat([x] + [e - x for e in emas_x_new], dim=-1)

        if self.mode_h == "concat":
            aug_h = torch.cat([h] + emas_h_new, dim=-1)
        else:  # diff
            aug_h = torch.cat([h] + [e - h for e in emas_h_new], dim=-1)

        z = torch.cat([aug_x, aug_h], dim=-1)

        # Closed-form CfC solution.
        f = self.f_gate(z)
        g = self.g_branch(z)
        h_branch = self.h_branch(z)

        if isinstance(dt, torch.Tensor):
            dt_b = dt
            if dt_b.dim() < 2:
                dt_b = dt_b.unsqueeze(-1)
            tau_eff = torch.exp(-f * dt_b / torch.abs(self.time_scale))
        else:
            tau_eff = torch.exp(-f * float(dt) / torch.abs(self.time_scale))

        h_new = tau_eff * g + (1.0 - tau_eff) * h_branch

        return h_new, emas_x_new, emas_h_new


class StackedEMAXHCfCStackedNetwork(nn.Module):
    """Stacked Stacked-EMA-XH-CfC network (PRD #10-123).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden dimension (per layer).
        output_size: output feature dimension.
        num_layers: number of stacked Stacked-EMA-XH-CfC cells.
        betas_x: list of β values for x-side EMAs.
        betas_h: list of β values for h-side EMAs.
        mode_x: 'diff' or 'concat' for x-side.
        mode_h: 'diff' or 'concat' for h-side.
        return_sequences: if True, return per-timestep outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        betas_x: list[float] = (0.7, 0.95),
        betas_h: list[float] = (0.7, 0.95),
        mode_x: str = "diff",
        mode_h: str = "diff",
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.betas_x = [float(b) for b in betas_x]
        self.betas_h = [float(b) for b in betas_h]
        self.Kx = len(betas_x)
        self.Kh = len(betas_h)
        self.mode_x = mode_x
        self.mode_h = mode_h
        self.return_sequences = return_sequences

        # Layer 0 receives input_size, layer 1+ receives hidden_size.
        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_input_size = input_size if layer_idx == 0 else hidden_size
            self.cells.append(
                StackedEMAXHCfCCell(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    betas_x=betas_x,
                    betas_h=betas_h,
                    mode_x=mode_x,
                    mode_h=mode_h,
                )
            )

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        emas_x0: list[torch.Tensor] | None = None,
        emas_h0: list[torch.Tensor] | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass over the sequence.

        Args:
            x: input [B, T, input_size].
            h0: optional initial hidden state.
            emas_x0: optional initial x-EMAs.
            emas_h0: optional initial h-EMAs.
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
        if emas_x0 is None:
            x_first = torch.nan_to_num(x[:, 0, :], nan=0.0)
            emas_x0 = [x_first.clone() for _ in self.betas_x]
        if emas_h0 is None:
            emas_h0 = [
                torch.zeros(
                    self.num_layers, batch_size, self.hidden_size,
                    device=x.device, dtype=x.dtype,
                )
                for _ in self.betas_h
            ]

        h_state = h0
        emas_x_state = emas_x0
        emas_h_state = emas_h0
        current_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h_state[i]
            # Re-initialize x-side EMAs to match the current layer's input size.
            if i == 0:
                # Layer 0: use the user-provided emas_x0 (which match input_size).
                emas_x_i = emas_x_state
            else:
                # Layer 1+: re-init EMAs to zeros matching current layer's
                # input size (hidden_size).
                first_input = torch.nan_to_num(current_input[:, 0, :])
                emas_x_i = [first_input.clone() for _ in self.betas_x]
            emas_h_i = [emas_h_state[k][i] for k in range(self.Kh)]
            for t in range(seq_len):
                dt_t = select_step_delta(dt, t, batch_size, seq_len, x.device, x.dtype)
                input_mask, update_mask = select_step_mask(
                    mask, t, batch_size, seq_len, current_input.shape[-1], x.device, x.dtype
                )
                x_t = torch.nan_to_num(current_input[:, t, :])
                if input_mask is not None:
                    x_t = x_t * input_mask
                h_new, emas_x_new, emas_h_new = cell(x_t, h_i, emas_x_i, emas_h_i, dt=dt_t)
                if update_mask is None:
                    h_i = h_new
                else:
                    h_i = update_mask * h_new + (1.0 - update_mask) * h_i
                emas_x_i = emas_x_new
                emas_h_i = emas_h_new
                outputs.append(h_i)
            layer_output = torch.stack(outputs, dim=1)

            # Update per-layer state tensors.
            new_h_state = []
            for j in range(self.num_layers):
                if j == i:
                    new_h_state.append(h_i)
                else:
                    new_h_state.append(h_state[j])
            h_state = torch.stack(new_h_state, dim=0)
            emas_x_state = emas_x_i  # x-side EMAs are layer-independent

            new_emas_h_state = []
            for k in range(self.Kh):
                layer_emas = []
                for j in range(self.num_layers):
                    if j == i:
                        layer_emas.append(emas_h_i[k])
                    else:
                        layer_emas.append(emas_h_state[k][j])
                new_emas_h_state.append(torch.stack(layer_emas, dim=0))
            emas_h_state = new_emas_h_state

            # The next layer's input is the current layer's hidden state.
            current_input = layer_output  # (B, T, hidden_size)

        if self.return_sequences:
            return self.output_proj(current_input)
        return self.output_proj(current_input[:, -1, :])
