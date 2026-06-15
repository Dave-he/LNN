"""LearnedBeta-XH-CfC (Per-Feature β on Stacked XH) (PRD #10-124, Round 162, 2026-06-15).

Combines per-feature learned β (round 157) with stacked x-side +
h-side multi-scale EMAs (round 161). The ULTIMATE CROSS-PRODUCT
of all winners from rounds 156-161.

Mechanism::

    # Per-feature learned β (round 157):
    beta_x_k,d = sigmoid(beta_x_k_raw[d])  # shape [Kx, D]
    beta_h_k,d = sigmoid(beta_h_k_raw[d])  # shape [Kh, H]

    # Input-side EMAs (round 158):
    ema_x_k,t[d] = beta_x_k,d * ema_x_k,t-1[d] + (1 - beta_x_k,d) * x_t[d]
    aug_x_t = [x_t, ema_x_1,t - x_t, ..., ema_x_Kx,t - x_t]

    # Hidden-state EMAs (round 160):
    ema_h_k,t[d] = beta_h_k,d * ema_h_k,t-1[d] + (1 - beta_h_k,d) * h_t[d]
    aug_h_t = [h_t, ema_h_1,t - h_t, ..., ema_h_Kh,t - h_t]

    # Combined:
    z_t = cat(aug_x_t, aug_h_t)

This is the FULL CROSS-PRODUCT of rounds 156-161:
- Multi-scale input (round 158 K=3)
- Multi-scale hidden (round 160 K=2)
- Per-feature learned β (round 157)
- Stacked X+H (round 161)

Variants (4 conds):
- lb_xh_diff_1_1:   Kx=1, Kh=1, per-feature learned β, both diff
- lb_xh_diff_3_2:   Kx=3, Kh=2, per-feature learned β, both diff
- lb_xh_concat_2_2: Kx=2, Kh=2, per-feature learned β, both concat
- lb_xh_best:       Kx=3, Kh=2, per-feature learned β, both diff

Audit context (91-161): 24 strictly positive + 17 target-dep +
33 negatives = 74 mechanism classes.

Risks:
- Per-feature β on multi-scale = many parameters to learn.
- Could overfit on small data.
- Diminishing returns from combining 2 positive mechanisms.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.sequence_utils import select_step_delta, select_step_mask


class LearnedBetaXHCfCCell(nn.Module):
    """LearnedBeta-XH-CfC cell: CfC with per-feature learned β
    on BOTH x-side and h-side multi-scale EMAs.

    Args:
        input_size: input feature dimension D (for this layer).
        hidden_size: hidden state dimension.
        Kx: number of x-side EMA channels.
        Kh: number of h-side EMA channels.
        mode_x: 'diff' or 'concat' for x-side.
        mode_h: 'diff' or 'concat' for h-side.
        beta_init: initial value for β (raw, before sigmoid).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        Kx: int = 1,
        Kh: int = 1,
        mode_x: str = "diff",
        mode_h: str = "diff",
        beta_init: float = 2.197,  # sigmoid(2.197) ≈ 0.9
    ):
        super().__init__()
        assert mode_x in ("diff", "concat"), f"mode_x must be 'diff' or 'concat', got {mode_x}"
        assert mode_h in ("diff", "concat"), f"mode_h must be 'diff' or 'concat', got {mode_h}"
        assert Kx >= 1, f"Kx must be >= 1, got {Kx}"
        assert Kh >= 1, f"Kh must be >= 1, got {Kh}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Kx = Kx
        self.Kh = Kh
        self.mode_x = mode_x
        self.mode_h = mode_h
        self.beta_init = float(beta_init)

        # Aug input size: D (original) + Kx * D (x-EMAs) = (Kx+1) * D
        aug_input_size = (self.Kx + 1) * input_size
        # Aug hidden size: H (original) + Kh * H (h-EMAs) = (Kh+1) * H
        aug_hidden_size = (self.Kh + 1) * hidden_size

        # Per-feature learned β for x-side (Kx channels × D features).
        self.beta_x_raw = nn.Parameter(torch.full((Kx, input_size), beta_init))
        # Per-feature learned β for h-side (Kh channels × H features).
        self.beta_h_raw = nn.Parameter(torch.full((Kh, hidden_size), beta_init))

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

    @property
    def beta_x(self) -> torch.Tensor:
        """Per-feature β in (0, 1) for x-side. Shape [Kx, D]."""
        return torch.sigmoid(self.beta_x_raw)

    @property
    def beta_h(self) -> torch.Tensor:
        """Per-feature β in (0, 1) for h-side. Shape [Kh, H]."""
        return torch.sigmoid(self.beta_h_raw)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        emas_x: list[torch.Tensor],
        emas_h: list[torch.Tensor],
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """One step of LearnedBeta-XH-CfC.

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

        # Update x-side EMAs with per-feature learned β.
        # beta_x shape [Kx, D], expand to [Kx, B, D] for broadcasting.
        beta_x = self.beta_x.unsqueeze(1)  # [Kx, 1, D]
        emas_x_new = [
            beta_x[k] * emas_x[k] + (1.0 - beta_x[k]) * x
            for k in range(self.Kx)
        ]
        # Update h-side EMAs with per-feature learned β.
        beta_h = self.beta_h.unsqueeze(1)  # [Kh, 1, H]
        emas_h_new = [
            beta_h[k] * emas_h[k] + (1.0 - beta_h[k]) * h
            for k in range(self.Kh)
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


class LearnedBetaXHCfCStackedNetwork(nn.Module):
    """Stacked LearnedBeta-XH-CfC network (PRD #10-124).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden dimension (per layer).
        output_size: output feature dimension.
        num_layers: number of stacked LearnedBeta-XH-CfC cells.
        Kx: number of x-side EMA channels.
        Kh: number of h-side EMA channels.
        mode_x: 'diff' or 'concat' for x-side.
        mode_h: 'diff' or 'concat' for h-side.
        beta_init: initial value for β (raw, before sigmoid).
        return_sequences: if True, return per-timestep outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        Kx: int = 1,
        Kh: int = 1,
        mode_x: str = "diff",
        mode_h: str = "diff",
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
        self.Kx = Kx
        self.Kh = Kh
        self.mode_x = mode_x
        self.mode_h = mode_h
        self.beta_init = float(beta_init)
        self.return_sequences = return_sequences

        # Layer 0 receives input_size, layer 1+ receives hidden_size.
        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_input_size = input_size if layer_idx == 0 else hidden_size
            self.cells.append(
                LearnedBetaXHCfCCell(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    Kx=Kx,
                    Kh=Kh,
                    mode_x=mode_x,
                    mode_h=mode_h,
                    beta_init=beta_init,
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
            emas_x0 = [x_first.clone() for _ in range(self.Kx)]
        if emas_h0 is None:
            emas_h0 = [
                torch.zeros(
                    self.num_layers, batch_size, self.hidden_size,
                    device=x.device, dtype=x.dtype,
                )
                for _ in range(self.Kh)
            ]

        h_state = h0
        emas_h_state = emas_h0
        current_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h_state[i]
            # Re-initialize x-side EMAs to match the current layer's input size.
            if i == 0:
                first_input = torch.nan_to_num(current_input[:, 0, :])
            else:
                first_input = torch.nan_to_num(current_input[:, 0, :])
            emas_x_i = [first_input.clone() for _ in range(self.Kx)]
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
