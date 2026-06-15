"""MultiBeta-H-CfC (Multi-Scale Hidden State EMA Augmentation) (PRD #10-122, Round 160, 2026-06-15).

Augments CfC hidden state h with MULTIPLE parallel EMAs at
different β values, providing temporal context at multiple
time-scales simultaneously — but applied to the hidden state,
not the input.

Mechanism::

    # At step t, for each k in 0..K-1:
    ema_h_k,t = beta_k * ema_h_k,t-1 + (1 - beta_k) * h_t
    # Build augmented hidden state:
    aug_h_t = f_concat(h_t, ema_h_1,t, ..., ema_h_K,t)  # variants

This is the CROSS of:
- Round 158 (MultiBeta-CfC): K=2/K=3 fixed β on input x.
- Round 159 (EMA-H-CfC): K=1 scalar β on hidden state h.

It tests whether multi-scale h-side EMA strictly improves over
single-scale h-side EMA.

Variants:
- K=2: beta in {0.7, 0.95} (short, long)
- K=3: beta in {0.5, 0.9, 0.99} (short, medium, long)
- Modes: 'diff' (high-pass: ema_h - h) or 'concat' (low-pass: ema_h)

Audit context (91-159): 21 strictly positive + 16 target-dep +
28 negatives = 65 mechanism classes.

Risks:
- More EMAs = more parameters (3H or 4H hidden state).
- High-pass with multiple cutoffs may be redundant on h-space.
- Long-window h-EMA may lag regime changes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.sequence_utils import select_step_delta, select_step_mask


class MultiBetaHCfCCell(nn.Module):
    """MultiBeta-H-CfC cell: CfC with multi-scale hidden-state EMA augmentation.

    Args:
        input_size: input feature dimension D (for this layer).
        hidden_size: hidden state dimension.
        betas: list of β values for parallel EMAs on h.
        mode: 'diff' (aug_h = [h, ema_1-h, ..., ema_K-h]) or
            'concat' (aug_h = [h, ema_1, ..., ema_K]).
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

        # Aug hidden state size: H (original) + K * H (EMAs) = (K+1) * H
        aug_hidden_size = (self.K + 1) * hidden_size

        # Standard CfC with augmented hidden state.
        self.f_gate = nn.Sequential(
            nn.Linear(input_size + aug_hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.g_branch = nn.Sequential(
            nn.Linear(input_size + aug_hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.h_branch = nn.Sequential(
            nn.Linear(input_size + aug_hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.time_scale = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        emas_h: list[torch.Tensor],
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """One step of MultiBeta-H-CfC.

        Args:
            x: input at this step [B, input_size].
            h: previous hidden state [B, hidden_size].
            emas_h: list of K previous EMA-hidden states, each [B, hidden_size].
            dt: time delta.

        Returns:
            (h_new, emas_h_new) tuple.
        """
        x = torch.nan_to_num(x, nan=0.0)
        h = torch.nan_to_num(h, nan=0.0)
        emas_h = [torch.nan_to_num(e, nan=0.0) for e in emas_h]

        # Update all EMAs of hidden state.
        emas_h_new = [
            beta * e + (1.0 - beta) * h
            for beta, e in zip(self.betas, emas_h)
        ]

        # Build augmented hidden state.
        if self.mode == "concat":
            aug_h = torch.cat([h] + emas_h_new, dim=-1)
        else:  # diff
            aug_h = torch.cat([h] + [e - h for e in emas_h_new], dim=-1)

        z = torch.cat([x, aug_h], dim=-1)

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

        return h_new, emas_h_new


class MultiBetaHCfCStackedNetwork(nn.Module):
    """Stacked MultiBeta-H-CfC network (PRD #10-122).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden dimension (per layer).
        output_size: output feature dimension.
        num_layers: number of stacked MultiBeta-H-CfC cells.
        betas: list of β values for parallel EMAs on h.
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

        # Layer 0 receives input_size, layer 1+ receives hidden_size.
        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_input_size = input_size if layer_idx == 0 else hidden_size
            self.cells.append(
                MultiBetaHCfCCell(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    betas=betas,
                    mode=mode,
                )
            )

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        emas_h0: list[torch.Tensor] | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass over the sequence.

        Args:
            x: input [B, T, input_size].
            h0: optional initial hidden state.
            emas_h0: optional initial EMA-hidden states (one per β).
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
        if emas_h0 is None:
            emas_h0 = [
                torch.zeros(
                    self.num_layers, batch_size, self.hidden_size,
                    device=x.device, dtype=x.dtype,
                )
                for _ in self.betas
            ]

        h_state = h0
        emas_h_state = emas_h0
        current_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h_state[i]
            emas_h_i = [emas_h_state[k][i] for k in range(self.K)]
            for t in range(seq_len):
                dt_t = select_step_delta(dt, t, batch_size, seq_len, x.device, x.dtype)
                input_mask, update_mask = select_step_mask(
                    mask, t, batch_size, seq_len, current_input.shape[-1], x.device, x.dtype
                )
                x_t = torch.nan_to_num(current_input[:, t, :])
                if input_mask is not None:
                    x_t = x_t * input_mask
                h_new, emas_h_new = cell(x_t, h_i, emas_h_i, dt=dt_t)
                if update_mask is None:
                    h_i = h_new
                else:
                    h_i = update_mask * h_new + (1.0 - update_mask) * h_i
                # Update all EMAs independently.
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

            new_emas_h_state = []
            for k in range(self.K):
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
