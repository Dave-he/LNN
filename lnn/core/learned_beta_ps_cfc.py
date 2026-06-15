"""LearnedPerScaleBeta-CfC (Per-Scale Trainable β) (PRD #10-133, Round 171, 2026-06-15).

Variant of round 167's LayerDecay-CfC with **learnable per-scale β**
on both x-side and h-side EMAs. Each scale has ONE scalar β value
(not per-feature) that is trained via gradient descent.

Round 169 established that β ∈ {0.75, 0.85, 0.95} (Kh=3) is the sweet
spot for hand-tuned β. Round 171 tests if data-driven β (gradient-
trained scalar β per scale) beats hand-tuned β.

Hypothesis:
- H1 (positive): data-driven β finds better values than {0.75, 0.85, 0.95}
- H2 (negative): {0.75, 0.85, 0.95} is already optimal for the bench
- H3 (mixed): learned β on x-side only helps (x-side has more diversity)

Mechanism::

    For each layer:
        # Per-scale learned β (NEW — round 171):
        beta_x_k = sigmoid(beta_x_k_raw[k])  # shape [Kx]
        beta_h_k = sigmoid(beta_h_k_raw[k])  # shape [Kh]
        # Per-sample EMAs:
        ema_x_k,t[b,d] = beta_x_k * ema_x_k,t-1[b,d] + (1 - beta_x_k) * x_t[b,d]
        ema_h_k,t[b,h] = beta_h_k * ema_h_k,t-1[b,h] + (1 - beta_h_k) * h_t[b,h]
        z_t = cat(aug_x_t, aug_h_t)
        h_t = CfC(z_t)

Audit context (91-170): 42 strictly positive + 17 target-dep +
35 negatives = 94 mechanism classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LearnedPerScaleBeta-CfC cell (single layer)
# ---------------------------------------------------------------------------


class LearnedBetaPSCfCCell(nn.Module):
    """Single CfC cell with per-scale learned β on BOTH x and h sides."""

    def __init__(self, input_size, hidden_size, Kx, Kh,
                 mode_x="diff", mode_h="diff",
                 beta_x_init=0.75, beta_h_init=0.75):
        """Initialize LearnedBetaPSCfCCell.

        Args:
            input_size: number of input features.
            hidden_size: number of hidden units.
            Kx: number of input-side EMA scales.
            Kh: number of hidden-side EMA scales.
            mode_x: 'diff' or 'concat' for x-side.
            mode_h: 'diff' or 'concat' for h-side.
            beta_x_init: initial scalar β value for x-side.
            beta_h_init: initial scalar β value for h-side.
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

        # Augmented sizes.
        aug_input_size = (Kx + 1) * input_size
        aug_hidden_size = (Kh + 1) * hidden_size

        # Convert init value to raw (logit) for sigmoid parameterization.
        def logit(p):
            return float(torch.log(torch.tensor(p / (1.0 - p))).item())

        # Per-scale learned β (NEW — round 171).
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
        """Per-scale β in (0, 1) for x-side. Shape [Kx]."""
        return torch.sigmoid(self.beta_x_raw)

    @property
    def beta_h(self):
        """Per-scale β in (0, 1) for h-side. Shape [Kh]."""
        return torch.sigmoid(self.beta_h_raw)

    def forward(self, x_t, h_t, emas_x, emas_h, dt=1.0):
        """One step of LearnedPerScaleBeta-CfC.

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

        # Update x-side EMAs (per-scale learned β).
        beta_x = self.beta_x  # [Kx]
        emas_x_new = [
            beta_x[k] * emas_x[k] + (1.0 - beta_x[k]) * x_t
            for k in range(self.Kx)
        ]
        # Update h-side EMAs (per-scale learned β).
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
# LearnedBetaPS-CfC stacked network
# ---------------------------------------------------------------------------


class LearnedBetaPSCfCStackedNetwork(nn.Module):
    """Stacked LearnedPerScaleBeta-CfC with per-scale learned β on both x and h."""

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
                LearnedBetaPSCfCCell(
                    in_size, hidden_size, Kx, Kh,
                    mode_x=mode_x, mode_h=mode_h,
                    beta_x_init=beta_x_init,
                    beta_h_init=beta_h_init,
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
        # Initialize hidden states and EMAs.
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


def make_lb_ps_h3_75(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=3, init β=0.75 (uniform start)."""
    return LearnedBetaPSCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, Kh=3,
        beta_x_init=0.75,
        beta_h_init=0.75,
        return_sequences=True,
    )


def make_lb_ps_h3_50(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=3, init β=0.5 (mid-range)."""
    return LearnedBetaPSCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, Kh=3,
        beta_x_init=0.5,
        beta_h_init=0.5,
        return_sequences=True,
    )


def make_lb_ps_h3_90(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=3, init β=0.9 (slow start)."""
    return LearnedBetaPSCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, Kh=3,
        beta_x_init=0.9,
        beta_h_init=0.9,
        return_sequences=True,
    )


def make_lb_ps_h2_75(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=2, init β=0.75."""
    return LearnedBetaPSCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, Kh=2,
        beta_x_init=0.75,
        beta_h_init=0.75,
        return_sequences=True,
    )


def make_lb_ps_h4_75(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=4, init β=0.75."""
    return LearnedBetaPSCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, Kh=4,
        beta_x_init=0.75,
        beta_h_init=0.75,
        return_sequences=True,
    )


def make_lb_ps_h5_75(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=5, init β=0.75."""
    return LearnedBetaPSCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, Kh=5,
        beta_x_init=0.75,
        beta_h_init=0.75,
        return_sequences=True,
    )


__all__ = [
    "LearnedBetaPSCfCCell",
    "LearnedBetaPSCfCStackedNetwork",
    "make_lb_ps_h3_75",
    "make_lb_ps_h3_50",
    "make_lb_ps_h3_90",
    "make_lb_ps_h2_75",
    "make_lb_ps_h4_75",
    "make_lb_ps_h5_75",
]
