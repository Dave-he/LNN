"""PerFeatureH-CfC (Per-Feature β on H-Side) (PRD #10-132, Round 170, 2026-06-15).

Variant of round 167's LayerDecay-CfC with **per-feature learned β**
on the h-side EMAs. Each hidden unit h gets its own β value for
each EMA scale k.

Round 162 tested per-feature β on h-side with Kh=2 and saw regression
(sin -15% vs -33%). Round 169 established that Kh=3 is the sweet
spot for the hybrid β family. This round tests if Kh=3 fixes the
per-feature regression.

Mechanism::

    For each layer:
        # Per-feature learned β on h-side (NEW):
        beta_h_k,h = sigmoid(beta_h_raw[k, h])  # shape [Kh, H]
        # Per-feature learned β on x-side (round 163+):
        beta_x_k,d = sigmoid(beta_x_k_raw[k, d])  # shape [Kx, D]
        # Per-sample EMAs:
        ema_x_k,t[b,d] = beta_x_k,d * ema_x_k,t-1[b,d] + (1 - beta_x_k,d) * x_t[b,d]
        ema_h_k,t[b,h] = beta_h_k,h * ema_h_k,t-1[b,h] + (1 - beta_h_k,h) * h_t[b,h]
        z_t = cat(aug_x_t, aug_h_t)
        h_t = CfC(z_t)

Audit context (91-169): 42 strictly positive + 17 target-dep +
35 negatives = 94 mechanism classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# PerFeatureH-CfC cell (single layer)
# ---------------------------------------------------------------------------


class PerFeatureHCfCCell(nn.Module):
    """Single CfC cell with per-feature learned β on BOTH x and h sides."""

    def __init__(self, input_size, hidden_size, Kx, Kh,
                 mode_x="diff", mode_h="diff",
                 beta_x_init=0.0, beta_h_init=0.0):
        """Initialize PerFeatureHCfCCell.

        Args:
            input_size: number of input features
            hidden_size: number of hidden units
            Kx: number of input-side EMA scales
            Kh: number of hidden-side EMA scales
            mode_x: 'diff' or 'concat' for x-side
            mode_h: 'diff' or 'concat' for h-side
            beta_x_init: initial value for x-side β (raw, before sigmoid)
            beta_h_init: initial value for h-side β (raw, before sigmoid)
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

        # Augmented sizes (same as round 163).
        aug_input_size = (Kx + 1) * input_size
        aug_hidden_size = (Kh + 1) * hidden_size

        # Per-feature learned β on x-side (round 163+).
        self.beta_x_raw = nn.Parameter(torch.full((Kx, input_size), beta_x_init))
        # Per-feature learned β on h-side (NEW — round 170).
        self.beta_h_raw = nn.Parameter(torch.full((Kh, hidden_size), beta_h_init))

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
        """Per-feature β in (0, 1) for x-side. Shape [Kx, D]."""
        return torch.sigmoid(self.beta_x_raw)

    @property
    def beta_h(self):
        """Per-feature β in (0, 1) for h-side. Shape [Kh, H]."""
        return torch.sigmoid(self.beta_h_raw)

    def forward(self, x_t, h_t, emas_x, emas_h, dt=1.0):
        """One step of PerFeatureH-CfC.

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

        # Update x-side EMAs (per-feature).
        beta_x = self.beta_x.unsqueeze(1)  # [Kx, 1, D]
        emas_x_new = [
            beta_x[k] * emas_x[k] + (1.0 - beta_x[k]) * x_t
            for k in range(self.Kx)
        ]
        # Update h-side EMAs (per-feature — NEW).
        beta_h = self.beta_h.unsqueeze(1)  # [Kh, 1, H]
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
# PerFeatureH-CfC stacked network
# ---------------------------------------------------------------------------


class PerFeatureHCfCStackedNetwork(nn.Module):
    """Stacked PerFeatureH-CfC with per-feature learned β on both x and h."""

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

        # Build cells.
        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else hidden_size
            self.cells.append(
                PerFeatureHCfCCell(
                    in_size, hidden_size, Kx, Kh,
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


def make_pfh_h3_finer(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=3, per-feature β on h-side, init raw=0 (β=0.5)."""
    return PerFeatureHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, Kh=3,
        return_sequences=True,
    )


def make_pfh_h3_k6(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=6, Kh=3, per-feature β on h-side."""
    return PerFeatureHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=6, Kh=3,
        return_sequences=True,
    )


def make_pfh_h4_wide(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=4, per-feature β on h-side."""
    return PerFeatureHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, Kh=4,
        return_sequences=True,
    )


def make_pfh_h2_const(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=2, per-feature β on h-side (round 162 control)."""
    return PerFeatureHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, Kh=2,
        return_sequences=True,
    )


def make_pfh_h5(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=5, per-feature β on h-side."""
    return PerFeatureHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, Kh=5,
        return_sequences=True,
    )


__all__ = [
    "PerFeatureHCfCCell",
    "PerFeatureHCfCStackedNetwork",
    "make_pfh_h3_finer",
    "make_pfh_h3_k6",
    "make_pfh_h4_wide",
    "make_pfh_h2_const",
    "make_pfh_h5",
]
