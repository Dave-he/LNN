"""EMA-H-CfC (Hidden State EMA Augmentation) (PRD #10-121, Round 159, 2026-06-15).

Augments CfC hidden state h with an Exponential Moving Average
(EMA) of the hidden state, providing explicit access to a
smoothed / low-pass-filtered version of h.

Mechanism::

    # At step t:
    ema_h_t = beta * ema_h_{t-1} + (1 - beta) * h_t
    aug_h_t = f_concat(h_t, ema_h_t)  # 4 variants

This is structurally different from rounds 155-158 (which
augment input x). It tests whether the multi-scale EMA pattern
transfers to a different signal (h instead of x).

Variants (mirror round 156 for direct comparability):
- eh_concat:    aug_h = [h_t, ema_h_t], input dim = 2H
- eh_gate:      aug_h = alpha * h_t + (1 - alpha) * ema_h_t, dim = H
- eh_diff:      aug_h = [h_t, ema_h_t - h_t], input dim = 2H
- eh_ema_only:  aug_h = ema_h_t only (control, replace h)

Audit context (91-158): 20 strictly positive + 15 target-dep +
28 negatives = 63 mechanism classes.

Risks:
- Interior augmentation may be too disruptive (affects g_branch
  and h_branch within the cell).
- β = 0.9 may not be optimal for h (h has different statistics
  than x).
- Could be redundant with input augmentation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.sequence_utils import select_step_delta, select_step_mask


class EMAHCfCCell(nn.Module):
    """EMA-H-CfC cell: CfC with hidden state EMA augmentation.

    Args:
        input_size: input feature dimension D (for this layer).
        hidden_size: hidden state dimension.
        ema_mode: 'concat' (2H), 'gate' (H), 'diff' (2H),
            'ema_only' (H).
        beta: EMA decay rate (fixed hyperparameter).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        ema_mode: str = "concat",
        beta: float = 0.9,
    ):
        super().__init__()
        assert ema_mode in ("concat", "gate", "diff", "ema_only"), \
            f"ema_mode must be one of 'concat', 'gate', 'diff', 'ema_only', got {ema_mode}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ema_mode = ema_mode
        self.beta = float(beta)

        # Determine the augmented hidden state size.
        if ema_mode in ("concat", "diff"):
            aug_hidden_size = 2 * hidden_size
        else:
            aug_hidden_size = hidden_size

        # For 'gate' mode, we need a learned alpha (initialized to
        # give a 50/50 mix of h and ema_h at start).
        if ema_mode == "gate":
            self.gate_alpha = nn.Parameter(torch.tensor(0.5))
        else:
            self.gate_alpha = None

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
        ema_h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step of EMA-H-CfC.

        Args:
            x: input at this step [B, input_size].
            h: previous hidden state [B, hidden_size].
            ema_h: previous EMA-hidden state [B, hidden_size].
            dt: time delta.

        Returns:
            (h_new, ema_h_new) tuple.
        """
        # Replace any NaN values with 0.
        x = torch.nan_to_num(x)

        # Update EMA of hidden state.
        ema_h_new = self.beta * ema_h + (1.0 - self.beta) * h

        # Build augmented hidden state.
        if self.ema_mode == "concat":
            aug_h = torch.cat([h, ema_h_new], dim=-1)
        elif self.ema_mode == "diff":
            aug_h = torch.cat([h, ema_h_new - h], dim=-1)
        elif self.ema_mode == "ema_only":
            aug_h = ema_h_new
        else:  # gate
            alpha = torch.sigmoid(self.gate_alpha)
            aug_h = alpha * h + (1.0 - alpha) * ema_h_new

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

        return h_new, ema_h_new


class EMAHCfCStackedNetwork(nn.Module):
    """Stacked EMA-H-CfC network (PRD #10-121).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden dimension (per layer).
        output_size: output feature dimension.
        num_layers: number of stacked EMA-H-CfC cells.
        ema_mode: 'concat', 'gate', 'diff', 'ema_only'.
        beta: EMA decay rate (fixed hyperparameter).
        return_sequences: if True, return per-timestep outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        ema_mode: str = "concat",
        beta: float = 0.9,
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.ema_mode = ema_mode
        self.beta = float(beta)
        self.return_sequences = return_sequences

        # Build cells with the correct input size per layer:
        # layer 0 receives input_size, layer 1+ receives hidden_size.
        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_input_size = input_size if layer_idx == 0 else hidden_size
            self.cells.append(
                EMAHCfCCell(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    ema_mode=ema_mode,
                    beta=beta,
                )
            )

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        ema_h0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass over the sequence.

        Args:
            x: input [B, T, input_size].
            h0: optional initial hidden state.
            ema_h0: optional initial EMA-hidden state.
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
        if ema_h0 is None:
            ema_h0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype,
            )

        h_state = h0
        ema_h_state = ema_h0
        current_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h_state[i]
            ema_h_i = ema_h_state[i]
            for t in range(seq_len):
                dt_t = select_step_delta(dt, t, batch_size, seq_len, x.device, x.dtype)
                input_mask, update_mask = select_step_mask(
                    mask, t, batch_size, seq_len, current_input.shape[-1], x.device, x.dtype
                )
                x_t = torch.nan_to_num(current_input[:, t, :])
                if input_mask is not None:
                    x_t = x_t * input_mask
                h_new, ema_h_new = cell(x_t, h_i, ema_h_i, dt=dt_t)
                if update_mask is None:
                    h_i = h_new
                else:
                    h_i = update_mask * h_new + (1.0 - update_mask) * h_i
                # ema_h_i is updated independently
                ema_h_i = ema_h_new
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

            new_ema_h_state = []
            for j in range(self.num_layers):
                if j == i:
                    new_ema_h_state.append(ema_h_i)
                else:
                    new_ema_h_state.append(ema_h_state[j])
            ema_h_state = torch.stack(new_ema_h_state, dim=0)

            # The next layer's input is the current layer's hidden state.
            current_input = layer_output  # (B, T, hidden_size)

        if self.return_sequences:
            return self.output_proj(current_input)
        return self.output_proj(current_input[:, -1, :])
