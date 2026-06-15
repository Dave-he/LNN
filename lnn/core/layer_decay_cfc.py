"""LayerDecay-CfC (Per-Layer β Schedule) (PRD #10-129, Round 167, 2026-06-15).

Variant of round 165's HybridBeta-XH-Deep-HighK with **per-layer β
schedule** for the h-side EMA. Layer 0 uses fast β (low-level,
short-term), deeper layers use slow β (high-level, abstractions).

Architecture follows round 163's HybridBeta-XH-CfC exactly,
except the h-side β values are PER-LAYER (not a single scalar
list shared across all layers).

Mechanism::

    For layer l in 0..L-1:
        # Per-layer β schedule for h-side:
        beta_h_l_k = schedule(l, beta_min, beta_max, K_h, mode)
        # Same as round 163:
        beta_x_k,d = sigmoid(beta_x_k_raw[d])  # shape [Kx, D]
        # Input-side EMAs (per-feature, per-sample):
        ema_x_k,t[b,d] = beta_x_k,d * ema_x_k,t-1[b,d] + (1 - beta_x_k,d) * x_t[b,d]
        # Per-layer hidden-state EMAs:
        ema_h_k,t[b,d] = beta_h_l_k * ema_h_k,t-1[b,d] + (1 - beta_h_l_k) * h_t[b,d]
        z_t = cat(aug_x_t, aug_h_t)
        h_t = CfC(z_t)  # closed-form

Schedule modes:
- "constant": all layers use betas_h (round 165 baseline)
- "linear":   β_l_k = β_min_k + l * (β_max_k - β_min_k) / (L-1)
- "reverse":  β_l_k = β_max_k - l * (β_max_k - β_min_k) / (L-1)

Audit context (91-166): 39 strictly positive + 17 target-dep +
35 negatives = 91 mechanism classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Per-layer β schedule
# ---------------------------------------------------------------------------


def make_layer_beta_schedule(betas_h, num_layers, mode):
    """Compute per-layer β schedule for h-side EMAs.

    Args:
        betas_h: list of K_h scalar β values for the h-side
        num_layers: number of stacked cells
        mode: one of "constant", "linear", "reverse"

    Returns:
        per_layer_betas: list of length num_layers, each a list
            of length K_h with β values for that layer.
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
            # Fast at low layers, slow at high layers.
            frac = l / (num_layers - 1)
        elif mode == "reverse":
            # Slow at low layers, fast at high layers.
            frac = 1.0 - l / (num_layers - 1)
        else:
            raise ValueError(f"Unknown mode {mode!r}")
        layer_betas = [
            beta_min + frac * (beta_max - beta_min) for _ in betas_h
        ]
        schedule.append(layer_betas)
    return schedule


# ---------------------------------------------------------------------------
# LayerDecay-CfC cell (single layer)
# ---------------------------------------------------------------------------


class LayerDecayCfCCell(nn.Module):
    """Single CfC cell with hybrid β on x + per-layer β on h.

    Same closed-form CfC as round 163, but h-side β values come
    from the layer-specific schedule (not a single shared list).
    """

    def __init__(self, input_size, hidden_size, Kx, betas_h,
                 mode_x="diff", mode_h="diff", beta_init=0.0):
        """Initialize LayerDecayCfCCell.

        Args:
            input_size: number of input features
            hidden_size: number of hidden units
            Kx: number of input-side EMA scales (K_h = len(betas_h))
            betas_h: list of K_h scalar β values for h-side (fixed)
            mode_x: 'diff' or 'concat' for x-side
            mode_h: 'diff' or 'concat' for h-side
            beta_init: initial value for x-side β (raw, before sigmoid)
        """
        super().__init__()
        assert mode_x in ("diff", "concat")
        assert mode_h in ("diff", "concat")
        assert Kx >= 1
        assert len(betas_h) >= 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Kx = Kx
        self.Kh = len(betas_h)
        self.betas_h = [float(b) for b in betas_h]
        self.mode_x = mode_x
        self.mode_h = mode_h

        # Augmented sizes (same as round 163).
        aug_input_size = (Kx + 1) * input_size
        aug_hidden_size = (self.Kh + 1) * hidden_size

        # Per-feature learnable β for x-side.
        self.beta_x_raw = nn.Parameter(torch.full((Kx, input_size), beta_init))

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
    def beta_x(self) -> torch.Tensor:
        """Per-feature β in (0, 1) for x-side. Shape [Kx, D]."""
        return torch.sigmoid(self.beta_x_raw)

    def forward(self, x_t, h_t, emas_x, emas_h, layer_betas_h, dt=1.0):
        """One step of LayerDecay-CfC.

        Args:
            x_t: input at this step [B, input_size].
            h_t: previous hidden state [B, hidden_size].
            emas_x: list of Kx previous x-EMAs, each [B, input_size].
            emas_h: list of Kh previous h-EMAs, each [B, hidden_size].
            layer_betas_h: list of Kh β values for THIS layer
                (overrides cell.betas_h for per-layer schedule).
            dt: time delta.

        Returns:
            (h_new, emas_x_new, emas_h_new) tuple.
        """
        x_t = torch.nan_to_num(x_t, nan=0.0)
        h_t = torch.nan_to_num(h_t, nan=0.0)
        emas_x = [torch.nan_to_num(e, nan=0.0) for e in emas_x]
        emas_h = [torch.nan_to_num(e, nan=0.0) for e in emas_h]

        # Update x-side EMAs (per-feature).
        beta_x = self.beta_x.unsqueeze(1)  # [Kx, 1, D]
        emas_x_new = [
            beta_x[k] * emas_x[k] + (1.0 - beta_x[k]) * x_t
            for k in range(self.Kx)
        ]
        # Update h-side EMAs (per-layer scalar, NOT cell.betas_h).
        emas_h_new = [
            layer_betas_h[k] * emas_h[k] + (1.0 - layer_betas_h[k]) * h_t
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

        # Closed-form CfC solution (same as round 163).
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
# LayerDecay-CfC stacked network
# ---------------------------------------------------------------------------


class LayerDecayCfCStackedNetwork(nn.Module):
    """Stacked LayerDecay-CfC with per-layer β schedule."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=3,
        Kx=5,
        betas_h=None,
        mode="linear",
        mode_x="diff",
        mode_h="diff",
        return_sequences=True,
    ):
        super().__init__()
        if betas_h is None:
            betas_h = [0.7, 0.95]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.Kx = Kx
        self.Kh = len(betas_h)
        self.betas_h = list(betas_h)
        self.mode = mode
        self.mode_x = mode_x
        self.mode_h = mode_h
        self.return_sequences = return_sequences

        # Compute per-layer β schedule.
        self.layer_betas_h = make_layer_beta_schedule(betas_h, num_layers, mode)

        # Build cells.
        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else hidden_size
            self.cells.append(
                LayerDecayCfCCell(
                    in_size, hidden_size, Kx, betas_h,
                    mode_x=mode_x, mode_h=mode_h,
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
        # Initialize hidden states and EMAs (per-sample, per-cell).
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
                layer_betas = self.layer_betas_h[l]
                hs[l], emas_x[l], emas_h[l] = cell(
                    inp, hs[l], emas_x[l], emas_h[l], layer_betas,
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


def make_ld_constant(input_size, hidden_size, output_size, num_layers=3):
    """Round 165 baseline: constant β ∈ {0.7, 0.95} on h-side."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.7, 0.95], mode="constant",
        return_sequences=True,
    )


def make_ld_linear_k5(input_size, hidden_size, output_size, num_layers=3):
    """Linear β schedule ∈ [0.5, 0.99], fast at low layers."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.5, 0.99], mode="linear",
        return_sequences=True,
    )


def make_ld_reverse_k5(input_size, hidden_size, output_size, num_layers=3):
    """Reverse linear β schedule ∈ [0.99, 0.5], slow at low layers."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.99, 0.5], mode="reverse",
        return_sequences=True,
    )


def make_ld_linear_slow(input_size, hidden_size, output_size, num_layers=3):
    """Linear β schedule ∈ [0.7, 0.99]."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.7, 0.99], mode="linear",
        return_sequences=True,
    )


def make_ld_linear_fast(input_size, hidden_size, output_size, num_layers=3):
    """Linear β schedule ∈ [0.3, 0.9]."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.3, 0.9], mode="linear",
        return_sequences=True,
    )


def make_ld_linear_relu(input_size, hidden_size, output_size, num_layers=3):
    """Linear β schedule ∈ [0.7, 0.95] (round 165 range)."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.7, 0.95], mode="linear",
        return_sequences=True,
    )


__all__ = [
    "LayerDecayCfCCell",
    "LayerDecayCfCStackedNetwork",
    "make_layer_beta_schedule",
    "make_ld_constant",
    "make_ld_linear_k5",
    "make_ld_reverse_k5",
    "make_ld_linear_slow",
    "make_ld_linear_fast",
    "make_ld_linear_relu",
]
