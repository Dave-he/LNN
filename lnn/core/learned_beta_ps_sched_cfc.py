"""LearnedPerScaleBeta+Schedule-CfC (Learnable β + Per-Layer Schedule) (PRD #10-134, Round 172, 2026-06-15).

Combination of round 171 (per-scale learnable β) and round 167
(per-layer β schedule). Each layer has its own learnable β values
with a schedule across layers.

Round 171: lb_ps_h2_75 wins sin (-76%), lb_ps_h5_75 wins structured
(-92%). Round 167: per-layer schedule (constant/linear/reverse)
modulates β across layers.

This round tests if combining learnable β with per-layer schedule
beats either alone.

Hypothesis:
- H1 (positive): schedule + learnable β compounds wins
- H2 (negative): schedule constrains β so it can't adapt
- H3 (mixed): schedule helps structured, learnable helps sin

Mechanism::

    For each layer l:
        # Per-scale learnable β (round 171):
        beta_x_k_raw, beta_h_k_raw (nn.Parameter)
        # Per-layer schedule (round 167):
        beta_x_k_eff = schedule_func(beta_x_k, layer=l, mode=mode)
        beta_h_k_eff = schedule_func(beta_h_k, layer=l, mode=mode)
        # EMAs:
        ema_x_k,t[b,d] = beta_x_k_eff * ema_x_k,t-1[b,d] + (1 - beta_x_k_eff) * x_t[b,d]
        ema_h_k,t[b,h] = beta_h_k_eff * ema_h_k,t-1[b,h] + (1 - beta_h_k_eff) * h_t[b,h]
        z_t = cat(aug_x_t, aug_h_t)
        h_t = CfC(z_t)

Audit context (91-171): 43 strictly positive + 17 target-dep +
35 negatives = 95 mechanism classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def make_layer_beta_schedule(betas_h, num_layers, mode):
    """Build per-layer β schedule (same as round 167).

    Args:
        betas_h: list of Kh base β values (per layer, will be shared with x)
        num_layers: number of layers
        mode: 'constant', 'linear', or 'reverse'

    Returns:
        list of num_layers lists, each of Kh β values
    """
    if mode == "constant":
        return [list(betas_h) for _ in range(num_layers)]
    if num_layers == 1:
        return [list(betas_h)]
    beta_min = min(betas_h)
    beta_max = max(betas_h)
    schedule = []
    for l in range(num_layers):
        if mode == "linear":
            frac = l / (num_layers - 1)
        elif mode == "reverse":
            frac = 1.0 - l / (num_layers - 1)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        layer_betas = [beta_min + frac * (beta_max - beta_min) for _ in betas_h]
        schedule.append(layer_betas)
    return schedule


# ---------------------------------------------------------------------------
# LearnedPerScaleBeta+Schedule-CfC cell (single layer)
# ---------------------------------------------------------------------------


class LearnedBetaPSSchedCfCCell(nn.Module):
    """Single CfC cell with per-scale learnable β + per-layer schedule."""

    def __init__(self, input_size, hidden_size, Kx, Kh,
                 mode_x="diff", mode_h="diff",
                 beta_x_init=0.75, beta_h_init=0.75,
                 schedule_mode="constant", layer_idx=0, num_layers=1):
        """Initialize LearnedBetaPSSchedCfCCell.

        Args:
            input_size: number of input features.
            hidden_size: number of hidden units.
            Kx: number of input-side EMA scales.
            Kh: number of hidden-side EMA scales.
            mode_x: 'diff' or 'concat' for x-side.
            mode_h: 'diff' or 'concat' for h-side.
            beta_x_init: initial scalar β value for x-side.
            beta_h_init: initial scalar β value for h-side.
            schedule_mode: 'constant', 'linear', or 'reverse'.
            layer_idx: which layer this is (for schedule).
            num_layers: total number of layers (for schedule).
        """
        super().__init__()
        assert mode_x in ("diff", "concat")
        assert mode_h in ("diff", "concat")
        assert Kx >= 1
        assert Kh >= 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Kx = Kx
        self.Kh = Kh
        self.mode_x = mode_x
        self.mode_h = mode_h
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.schedule_mode = schedule_mode

        # Augmented sizes.
        aug_input_size = (Kx + 1) * input_size
        aug_hidden_size = (Kh + 1) * hidden_size

        # Convert init value to raw (logit) for sigmoid parameterization.
        def logit(p):
            return float(torch.log(torch.tensor(p / (1.0 - p))).item())

        # Per-scale learned β (round 171 base).
        self.beta_x_raw = nn.Parameter(torch.full((Kx,), logit(beta_x_init)))
        self.beta_h_raw = nn.Parameter(torch.full((Kh,), logit(beta_h_init)))

        # CfC closed-form components.
        aug_total = aug_input_size + aug_hidden_size
        self.f_gate = nn.Sequential(
            nn.Linear(aug_total, hidden_size),
            nn.Sigmoid(),
        )
        self.g_branch = nn.Sequential(
            nn.Linear(aug_total, hidden_size),
            nn.Tanh(),
        )
        self.h_branch = nn.Sequential(
            nn.Linear(aug_total, hidden_size),
            nn.Tanh(),
        )
        self.time_scale = nn.Parameter(torch.ones(hidden_size))

    @property
    def beta_x(self):
        """Per-scale β with schedule applied. Shape [Kx]."""
        base = torch.sigmoid(self.beta_x_raw)  # [Kx]
        if self.schedule_mode == "constant" or self.num_layers == 1:
            return base
        # Compute schedule factor for this layer (0..1).
        if self.schedule_mode == "linear":
            frac = self.layer_idx / (self.num_layers - 1)
        elif self.schedule_mode == "reverse":
            frac = 1.0 - self.layer_idx / (self.num_layers - 1)
        else:
            return base
        # Scale base β by (0.5 + 0.5 * frac). Layer 0 (linear): scale=0.5, layer
        # N-1 (linear): scale=1.0. Reverse inverts.
        scale = 0.5 + 0.5 * frac
        return base * scale

    @property
    def beta_h(self):
        """Per-scale β with schedule applied. Shape [Kh]."""
        base = torch.sigmoid(self.beta_h_raw)  # [Kh]
        if self.schedule_mode == "constant" or self.num_layers == 1:
            return base
        if self.schedule_mode == "linear":
            frac = self.layer_idx / (self.num_layers - 1)
        elif self.schedule_mode == "reverse":
            frac = 1.0 - self.layer_idx / (self.num_layers - 1)
        else:
            return base
        scale = 0.5 + 0.5 * frac
        return base * scale

    def forward(self, x_t, h_t, emas_x, emas_h, dt=1.0):
        """One step of LearnedPerScaleBeta+Schedule-CfC.

        Args:
            x_t: input at this step [B, input_size].
            h_t: previous hidden state [B, hidden_size].
            emas_x: list of Kx previous x-EMAs, each [B, input_size].
            emas_h: list of Kh previous h-EMAs, each [B, hidden_size].
            dt: time delta.

        Returns:
            (h_new, emas_x_new, emas_h_new) tuple.
        """
        x_t = torch.nan_to_num(x_t, nan=0.0)
        h_t = torch.nan_to_num(h_t, nan=0.0)
        emas_x = [torch.nan_to_num(e, nan=0.0) for e in emas_x]
        emas_h = [torch.nan_to_num(e, nan=0.0) for e in emas_h]

        # Update x-side EMAs (per-scale learned β with schedule).
        beta_x = self.beta_x  # [Kx]
        emas_x_new = [
            beta_x[k] * emas_x[k] + (1.0 - beta_x[k]) * x_t
            for k in range(self.Kx)
        ]
        # Update h-side EMAs (per-scale learned β with schedule).
        beta_h = self.beta_h  # [Kh]
        emas_h_new = [
            beta_h[k] * emas_h[k] + (1.0 - beta_h[k]) * h_t
            for k in range(self.Kh)
        ]

        # Build augmented x and h.
        if self.mode_x == "concat":
            aug_x = torch.cat([x_t] + emas_x_new, dim=-1)
        else:
            aug_x = torch.cat([x_t] + [e - x_t for e in emas_x_new], dim=-1)

        if self.mode_h == "concat":
            aug_h = torch.cat([h_t] + emas_h_new, dim=-1)
        else:
            aug_h = torch.cat([h_t] + [e - h_t for e in emas_h_new], dim=-1)

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


# ---------------------------------------------------------------------------
# Stacked network
# ---------------------------------------------------------------------------


class LearnedBetaPSSchedCfCStackedNetwork(nn.Module):
    """Stacked LearnedPerScaleBeta+Schedule-CfC with per-layer schedule."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=3,
        Kx=5,
        Kh=3,
        mode_x="diff",
        mode_h="diff",
        beta_x_init=0.75,
        beta_h_init=0.75,
        schedule_mode="constant",
        return_sequences=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.Kx = Kx
        self.Kh = Kh
        self.return_sequences = return_sequences

        # Build cells with per-layer schedule.
        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else hidden_size
            self.cells.append(
                LearnedBetaPSSchedCfCCell(
                    in_size, hidden_size, Kx, Kh,
                    mode_x=mode_x, mode_h=mode_h,
                    beta_x_init=beta_x_init,
                    beta_h_init=beta_h_init,
                    schedule_mode=schedule_mode,
                    layer_idx=l,
                    num_layers=num_layers,
                ),
            )

        # Output head.
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward a full sequence.

        Args:
            x: [B, T, D] input sequence
        Returns:
            y: [B, T, output_size] if return_sequences else [B, output_size]
        """
        B, T, _ = x.shape
        device = x.device
        hs = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.num_layers)]
        emas_x = [
            [torch.zeros(B, self.cells[l].input_size, device=device) for _ in range(self.Kx)]
            for l in range(self.num_layers)
        ]
        emas_h = [
            [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.Kh)]
            for l in range(self.num_layers)
        ]
        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            for l, cell in enumerate(self.cells):
                hs[l], emas_x[l], emas_h[l] = cell(
                    inp, hs[l], emas_x[l], emas_h[l],
                )
                inp = hs[l]
            outputs.append(self.head(hs[-1]))
        outputs = torch.stack(outputs, dim=1)
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def make_lbps_h3_75_const(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=3, learnable β, schedule=constant (round 171 baseline)."""
    return LearnedBetaPSSchedCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=3,
        beta_x_init=0.75, beta_h_init=0.75,
        schedule_mode="constant", return_sequences=True,
    )


def make_lbps_h3_75_linear(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=3, learnable β, schedule=linear."""
    return LearnedBetaPSSchedCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=3,
        beta_x_init=0.75, beta_h_init=0.75,
        schedule_mode="linear", return_sequences=True,
    )


def make_lbps_h3_75_reverse(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=3, learnable β, schedule=reverse (round 167 winner)."""
    return LearnedBetaPSSchedCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=3,
        beta_x_init=0.75, beta_h_init=0.75,
        schedule_mode="reverse", return_sequences=True,
    )


def make_lbps_h2_75_const(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=2, learnable β, schedule=constant (round 171 winner)."""
    return LearnedBetaPSSchedCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=2,
        beta_x_init=0.75, beta_h_init=0.75,
        schedule_mode="constant", return_sequences=True,
    )


def make_lbps_h2_75_reverse(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=2, learnable β, schedule=reverse."""
    return LearnedBetaPSSchedCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=2,
        beta_x_init=0.75, beta_h_init=0.75,
        schedule_mode="reverse", return_sequences=True,
    )


def make_lbps_h5_75_const(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=5, learnable β, schedule=constant (round 171 winner)."""
    return LearnedBetaPSSchedCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=5,
        beta_x_init=0.75, beta_h_init=0.75,
        schedule_mode="constant", return_sequences=True,
    )


def make_lbps_h5_75_reverse(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=5, learnable β, schedule=reverse."""
    return LearnedBetaPSSchedCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=5,
        beta_x_init=0.75, beta_h_init=0.75,
        schedule_mode="reverse", return_sequences=True,
    )


__all__ = [
    "LearnedBetaPSSchedCfCCell",
    "LearnedBetaPSSchedCfCStackedNetwork",
    "make_lbps_h3_75_const",
    "make_lbps_h3_75_linear",
    "make_lbps_h3_75_reverse",
    "make_lbps_h2_75_const",
    "make_lbps_h2_75_reverse",
    "make_lbps_h5_75_const",
    "make_lbps_h5_75_reverse",
]
