"""LearnedBetaPS+LNout-CfC (Per-Scale Learnable β + Output LayerNorm) (PRD #10-145, Round 183, 2026-06-16).

Variant of round 179's LearnedBetaPS+LN-CfC but with **LayerNorm
applied to h_new (output of CfC closed-form)** instead of z
(input to CfC linear projections).

Hypothesis:
- H1 (positive): post-LN normalizes h_new magnitude → stable
  gradients across timesteps
- H2 (mixed): post-LN differs from pre-LN — capture different
  signal
- H3 (negative): post-LN redundant with pre-LN

Audit context (91-182): 45 strictly positive + 18 target-dep +
43 negatives = 106 mechanism classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _logit(p):
    return float(torch.log(torch.tensor(p / (1.0 - p))).item())


class LearnedBetaPSLNOUTCfCCell(nn.Module):
    """Single CfC cell with per-scale learned β AND Output LayerNorm."""

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
        ln_eps=1e-5,
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

        aug_input_size = (Kx + 1) * input_size
        aug_hidden_size = (Kh + 1) * hidden_size

        # Per-scale learned β (round 171).
        self.beta_x_raw = nn.Parameter(torch.full((Kx,), _logit(beta_x_init)))
        self.beta_h_raw = nn.Parameter(torch.full((Kh,), _logit(beta_h_init)))

        # NEW (round 183): LayerNorm at OUTPUT (h_new), not input.
        self.layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)

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
        return torch.sigmoid(self.beta_x_raw)

    @property
    def beta_h(self):
        return torch.sigmoid(self.beta_h_raw)

    def forward(self, x_t, h_t, emas_x, emas_h, dt=1.0):
        x_t = torch.nan_to_num(x_t, nan=0.0)
        h_t = torch.nan_to_num(h_t, nan=0.0)
        emas_x = [torch.nan_to_num(e, nan=0.0) for e in emas_x]
        emas_h = [torch.nan_to_num(e, nan=0.0) for e in emas_h]

        beta_x = self.beta_x
        emas_x_new = [
            beta_x[k] * emas_x[k] + (1.0 - beta_x[k]) * x_t
            for k in range(self.Kx)
        ]
        beta_h = self.beta_h
        emas_h_new = [
            beta_h[k] * emas_h[k] + (1.0 - beta_h[k]) * h_t
            for k in range(self.Kh)
        ]

        if self.mode_x == "concat":
            aug_x = torch.cat([x_t] + emas_x_new, dim=-1)
        else:
            aug_x = torch.cat([x_t] + [e - x_t for e in emas_x_new], dim=-1)

        if self.mode_h == "concat":
            aug_h = torch.cat([h_t] + emas_h_new, dim=-1)
        else:
            aug_h = torch.cat([h_t] + [e - h_t for e in emas_h_new], dim=-1)

        z = torch.cat([aug_x, aug_h], dim=-1)

        # Standard CfC (no LN on z).
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

        # NEW (round 183): LayerNorm on h_new (post-CfC).
        h_new = self.layer_norm(h_new)

        return h_new, emas_x_new, emas_h_new


class LearnedBetaPSLNOUTCfCStackedNetwork(nn.Module):
    """Stacked LearnedBetaPS+LNout-CfC with LayerNorm on h_new."""

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
                LearnedBetaPSLNOUTCfCCell(
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


def make_lbps_lno_h3_75(input_size, hidden_size, output_size, num_layers=3):
    return LearnedBetaPSLNOUTCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=3,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_lno_h2_75(input_size, hidden_size, output_size, num_layers=3):
    return LearnedBetaPSLNOUTCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=2,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_lno_h5_75(input_size, hidden_size, output_size, num_layers=3):
    return LearnedBetaPSLNOUTCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


__all__ = [
    "LearnedBetaPSLNOUTCfCCell",
    "LearnedBetaPSLNOUTCfCStackedNetwork",
    "make_lbps_lno_h3_75",
    "make_lbps_lno_h2_75",
    "make_lbps_lno_h5_75",
]
