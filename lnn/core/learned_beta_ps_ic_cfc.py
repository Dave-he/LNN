"""LearnedBetaPS+IC-CfC (Per-Scale Learnable β + Input-Conditioned) (PRD #10-140, Round 178, 2026-06-16).

Variant of round 171's LearnedPerScaleBeta-CfC where **β is
produced by a small MLP applied to the input** — β is no longer
a static learned parameter but **data-dependent**.

Round 171 (static learnable β) is SOTA at sin 0.0064 (-76%).
This round tests if **dynamic β** (β_t = MLP(x_t)) can beat
static learnable β.

Hypothesis:
- H1 (positive): input-conditioned β beats static β on
  multi-regime data (structured)
- H2 (positive): per-sample, per-scale β reduces overfitting
- H3 (negative): too many parameters → overfit

Audit context (91-177): 43 strictly positive + 18 target-dep +
40 negatives = 101 mechanism classes.

Mechanism::

    For each timestep t:
        # Input-conditioned β:
        beta_x[b, k] = sigmoid(W_x · x_t[b] + b_x[k])
        beta_h[b, k] = sigmoid(W_h · h_{t-1}[b] + b_h[k])
        # Per-sample, per-scale EMA update:
        ema_x_k[b] = beta_x[b, k] * ema_x_k[b] + (1 - beta_x[b, k]) * x_t[b]
        ema_h_k[b] = beta_h[b, k] * ema_h_k[b] + (1 - beta_h[b, k]) * h_{t-1}[b]
        # Closed-form CfC:
        h_t = CfC(z_t)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _logit(p):
    return float(torch.log(torch.tensor(p / (1.0 - p))).item())


# ---------------------------------------------------------------------------
# Single cell with input-conditioned β
# ---------------------------------------------------------------------------


class LearnedBetaPSICfCCell(nn.Module):
    """Single CfC cell with input-conditioned β on BOTH x and h sides."""

    def __init__(
        self,
        input_size,
        hidden_size,
        Kx,
        Kh,
        mode_x="diff",
        mode_h="diff",
        beta_x_init=0.75,
        beta_h_init=0.75,
        beta_x_hidden=None,  # default: input_size
        beta_h_hidden=None,  # default: hidden_size
    ):
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

        # Input-conditioned β (NEW — round 178).
        # Linear(x → Kx) with bias init to logit(beta_x_init).
        x_hidden = beta_x_hidden if beta_x_hidden is not None else input_size
        h_hidden = beta_h_hidden if beta_h_hidden is not None else hidden_size
        self.beta_x_proj = nn.Linear(input_size, Kx)
        nn.init.zeros_(self.beta_x_proj.weight)
        nn.init.constant_(self.beta_x_proj.bias, _logit(beta_x_init))
        self.beta_h_proj = nn.Linear(hidden_size, Kh)
        nn.init.zeros_(self.beta_h_proj.weight)
        nn.init.constant_(self.beta_h_proj.bias, _logit(beta_h_init))

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

    def forward(self, x_t, h_t, emas_x, emas_h, dt=1.0):
        """One step of LearnedBetaPS+IC-CfC.

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

        # Input-conditioned β.
        beta_x = torch.sigmoid(self.beta_x_proj(x_t))  # [B, Kx]
        beta_h = torch.sigmoid(self.beta_h_proj(h_t))  # [B, Kh]

        # Per-sample, per-scale EMA updates.
        emas_x_new = []
        for k in range(self.Kx):
            beta_x_k = beta_x[:, k].unsqueeze(-1)  # [B, 1]
            emas_x_new.append(
                beta_x_k * emas_x[k] + (1.0 - beta_x_k) * x_t,
            )
        emas_h_new = []
        for k in range(self.Kh):
            beta_h_k = beta_h[:, k].unsqueeze(-1)  # [B, 1]
            emas_h_new.append(
                beta_h_k * emas_h[k] + (1.0 - beta_h_k) * h_t,
            )

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


class LearnedBetaPSICfCStackedNetwork(nn.Module):
    """Stacked LearnedBetaPS+IC-CfC with input-conditioned β."""

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

        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else hidden_size
            self.cells.append(
                LearnedBetaPSICfCCell(
                    in_size, hidden_size, Kx, Kh,
                    mode_x=mode_x, mode_h=mode_h,
                    beta_x_init=beta_x_init,
                    beta_h_init=beta_h_init,
                ),
            )

        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
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


def make_lbps_ic_h3_75(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=3, β_init=0.75, IC β (round 178 control)."""
    return LearnedBetaPSICfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=3,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_ic_h2_75(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=2, IC β (sin-favoring from round 171)."""
    return LearnedBetaPSICfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=2,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_ic_h5_75(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=5, IC β (structured-favoring from round 171)."""
    return LearnedBetaPSICfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_ic_h3_50(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=3, β_init=0.50, IC β (fast EMA)."""
    return LearnedBetaPSICfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=3,
        beta_x_init=0.50, beta_h_init=0.50, return_sequences=True,
    )


def make_lbps_ic_h3_90(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=3, β_init=0.90, IC β (slow EMA)."""
    return LearnedBetaPSICfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=3,
        beta_x_init=0.90, beta_h_init=0.90, return_sequences=True,
    )


__all__ = [
    "LearnedBetaPSICfCCell",
    "LearnedBetaPSICfCStackedNetwork",
    "make_lbps_ic_h3_75",
    "make_lbps_ic_h2_75",
    "make_lbps_ic_h5_75",
    "make_lbps_ic_h3_50",
    "make_lbps_ic_h3_90",
]
