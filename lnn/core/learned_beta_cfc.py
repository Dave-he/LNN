"""LearnedBeta-CfC (per-feature learnable beta EMA) (PRD #10-119, Round 157, 2026-06-15).

Augments CfC input with an Exponential Moving Average (EMA) of
the input, where the smoothing factor β is per-feature and
LEARNABLE (initialized to 0.9 via sigmoid parameterization).

Mechanism::

    # At step t:
    beta = sigmoid(beta_raw)  # per-feature, in (0, 1)
    ema_t[d] = beta[d] * ema_{t-1}[d] + (1 - beta[d]) * x_t[d]
    aug_x_t = f_concat(x_t, ema_t)  # 4 variants

This is the natural extension of round 156 (EMA-X-CfC with
scalar β=0.9). The hypothesis is that different features need
different smoothing:
- A slow trend feature benefits from high β (long EMA window).
- A noisy feature benefits from low β (short EMA window).
- Per-feature β lets the model learn this automatically.

Variants (mirror round 156 for direct comparability):
- lb_concat:    aug_x = [x_t, ema_t], input dim = 2D
- lb_gate:      aug_x = alpha * x_t + (1 - alpha) * ema_t, dim = D
- lb_diff:      aug_x = [x_t, ema_t - x_t], input dim = 2D
- lb_ema_only:  aug_x = ema_t only (control, replace x)

Audit context (91-156): 17 strictly positive + 12 target-dep +
26 negatives = 55 mechanism classes.

Risks:
- Per-feature β may be hard to learn (55 → 5.5 effective per
  round 156 finding that gate is hard to learn).
- Sigmoid parameterization keeps β in (0, 1) but gradient may
  saturate.
- 2D input variant doubles parameter count.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.sequence_utils import select_step_delta, select_step_mask


class LearnedBetaCfCCell(nn.Module):
    """LearnedBeta-CfC cell: CfC with per-feature learnable β EMA.

    Args:
        input_size: input feature dimension D.
        hidden_size: hidden state dimension.
        ema_mode: 'concat' (2D), 'gate' (D), 'diff' (2D),
            'ema_only' (D).
        beta_init: initial value for β (raw, before sigmoid).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        ema_mode: str = "concat",
        beta_init: float = 2.197,  # sigmoid(2.197) ≈ 0.9
    ):
        super().__init__()
        assert ema_mode in ("concat", "gate", "diff", "ema_only"), \
            f"ema_mode must be one of 'concat', 'gate', 'diff', 'ema_only', got {ema_mode}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ema_mode = ema_mode
        self.beta_init = float(beta_init)

        # Determine the augmented input size.
        if ema_mode in ("concat", "diff"):
            aug_input_size = 2 * input_size
        else:
            aug_input_size = input_size

        # Per-feature learnable β (raw, sigmoid-parameterized).
        # Initialized so that sigmoid(beta_init) ≈ 0.9 by default.
        self.beta_raw = nn.Parameter(torch.full((input_size,), beta_init))

        # For 'gate' mode, we need a learned alpha (initialized to
        # give a 50/50 mix of x and ema at start).
        if ema_mode == "gate":
            self.gate_alpha = nn.Parameter(torch.tensor(0.5))
        else:
            self.gate_alpha = None

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

    @property
    def beta(self) -> torch.Tensor:
        """Per-feature β in (0, 1)."""
        return torch.sigmoid(self.beta_raw)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        ema: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step of LearnedBeta-CfC.

        Args:
            x: input at this step [B, input_size].
            h: previous hidden state [B, hidden_size].
            ema: previous EMA state [B, input_size].
            dt: time delta.

        Returns:
            (h_new, ema_new) tuple.
        """
        x = torch.nan_to_num(x, nan=0.0)
        ema = torch.nan_to_num(ema, nan=0.0)

        # Update EMA with per-feature learnable β.
        beta = self.beta  # [D]
        ema_new = beta * ema + (1.0 - beta) * x

        # Build augmented input.
        if self.ema_mode == "concat":
            aug_x = torch.cat([x, ema_new], dim=-1)
        elif self.ema_mode == "diff":
            aug_x = torch.cat([x, ema_new - x], dim=-1)
        elif self.ema_mode == "ema_only":
            aug_x = ema_new
        else:  # gate
            alpha = torch.sigmoid(self.gate_alpha)
            aug_x = alpha * x + (1.0 - alpha) * ema_new

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

        return h_new, ema_new


class LearnedBetaCfCStackedNetwork(nn.Module):
    """Stacked LearnedBeta-CfC network (PRD #10-119).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden dimension (per layer).
        output_size: output feature dimension.
        num_layers: number of stacked LearnedBeta-CfC cells.
        ema_mode: 'concat', 'gate', 'diff', 'ema_only'.
        beta_init: initial value for β (raw, before sigmoid).
        return_sequences: if True, return per-timestep outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        ema_mode: str = "concat",
        beta_init: float = 2.197,
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
        self.beta_init = float(beta_init)
        self.return_sequences = return_sequences

        # The first cell's input is the augmented original input.
        # Subsequent cells receive the previous layer's h output (no
        # EMA augmentation on hidden state — that would be a different
        # signal).
        layer_in_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        self.cells = nn.ModuleList()
        for li in range(num_layers):
            self.cells.append(
                LearnedBetaCfCCell(
                    layer_in_sizes[li], hidden_size,
                    ema_mode=ema_mode, beta_init=beta_init,
                )
            )

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        ema0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass over the sequence.

        Args:
            x: input [B, T, input_size].
            h0: optional initial hidden state.
            ema0: optional initial EMA state.
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
        if ema0 is None:
            # Initialize EMA to first non-NaN x.
            x_first = torch.nan_to_num(x[:, 0, :], nan=0.0)
            ema0 = x_first

        h = h0
        ema_state = ema0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            # EMA state dim must match the cell's input dim.
            ema_i = torch.zeros(
                batch_size, cell.input_size, device=x.device, dtype=x.dtype,
            )
            if i == 0:
                ema_i = ema_state
            for t in range(seq_len):
                dt_t = select_step_delta(dt, t, batch_size, seq_len, x.device, x.dtype)
                input_mask, update_mask = select_step_mask(
                    mask, t, batch_size, seq_len, self.input_size, x.device, x.dtype
                )
                x_t = torch.nan_to_num(layer_input[:, t, :])
                if i == 0 and input_mask is not None:
                    x_t = x_t * input_mask
                h_new, ema_new = cell(x_t, h_i, ema_i, dt=dt_t)
                h_i = h_new if update_mask is None else update_mask * h_new + (1.0 - update_mask) * h_i
                outputs.append(h_i)
                ema_i = ema_new
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat(
                [h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)],
                dim=0,
            )

        if self.return_sequences:
            return self.output_proj(layer_input)
        return self.output_proj(layer_input[:, -1, :])
