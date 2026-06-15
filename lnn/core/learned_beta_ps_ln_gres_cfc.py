"""LearnedBetaPS+LN+GatedResidual-CfC (Per-Scale Learnable β + LayerNorm + Gated Residual) (PRD #10-147, Round 185, 2026-06-16).

Variant of round 179's LearnedBetaPS+LN-CfC with **gated
residual ON TOP of the CfC step** (not replacing it). This
preserves the closed-form CfC interpolation (which is
critical for stability, per round 184 finding) and adds a
learnable correction gated by a scalar α.

Hypothesis:
- H1 (positive): gated residual adds well-conditioned
  correction while preserving CfC stability
- H2 (negative): residual is redundant with CfC's tau_eff
- H3 (mixed): helps structured (preserves slow components)
  but hurts sin (over-emphasis on h_t history)

Audit context (91-184): 45 strictly positive + 18 target-dep +
45 negatives = 108 mechanism classes.

Mechanism::

    For each timestep t:
        # Per-scale EMAs (round 171):
        ema_x_k = β_x_k * ema_x_k + (1 - β_x_k) * x_t
        ema_h_k = β_h_k * ema_h_k + (1 - β_h_k) * h_{t-1}
        # Augmented input (concat or diff):
        z = cat([aug_x, aug_h])  # [B, (Kx+1)*D + (Kh+1)*H]
        # LN (round 179):
        z_norm = LayerNorm(z)
        # CfC closed-form (using z_norm):
        h_cfc = σ(-f·τ)·g + (1-σ(-f·τ))·h_branch
        # NEW (round 185): gated residual ON TOP of CfC
        alpha = sigmoid(alpha_raw)  # scalar per cell
        h_residual = residual_proj(z_norm)  # [B, H]
        h_new = h_cfc + alpha * h_residual
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.learned_beta_ps_ln_cfc import LearnedBetaPSLNCfCCell


def _logit(p):
    return float(torch.log(torch.tensor(p / (1.0 - p))).item())


class LearnedBetaPSLNGatedResidualCfCCell(LearnedBetaPSLNCfCCell):
    """Single CfC cell with LN + gated residual ON TOP of CfC step.

    Inherits LayerNorm-on-z + per-scale β from
    LearnedBetaPSLNCfCCell. Adds a learnable scalar α
    and a residual projection so the final output is
    ``h_cfc + alpha * Residual(LN(z))`` instead of pure CfC.
    """

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
        alpha_init=0.1,
        residual_init=0.1,
    ):
        super().__init__(
            input_size, hidden_size, Kx, Kh,
            mode_x=mode_x, mode_h=mode_h,
            beta_x_init=beta_x_init,
            beta_h_init=beta_h_init,
            ln_eps=ln_eps,
        )
        self.alpha_init = alpha_init
        aug_total = (Kx + 1) * input_size + (Kh + 1) * hidden_size
        # Scalar gate per cell, broadcast over batch and feature.
        self.alpha_raw = nn.Parameter(torch.tensor(_logit(alpha_init)))
        # Residual projection from z (already LN-normalized) → h.
        self.residual_proj = nn.Linear(aug_total, hidden_size)
        # Initialize residual to small magnitude.
        with torch.no_grad():
            self.residual_proj.weight.mul_(residual_init)
            self.residual_proj.weight.data -= self.residual_proj.weight.data
            self.residual_proj.bias.data.zero_()

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw)

    def forward(self, x_t, h_t, emas_x, emas_h, dt=1.0):
        x_t = torch.nan_to_num(x_t, nan=0.0)
        h_t_in = torch.nan_to_num(h_t, nan=0.0)
        emas_x = [torch.nan_to_num(e, nan=0.0) for e in emas_x]
        emas_h = [torch.nan_to_num(e, nan=0.0) for e in emas_h]

        beta_x = self.beta_x
        emas_x_new = [
            beta_x[k] * emas_x[k] + (1.0 - beta_x[k]) * x_t
            for k in range(self.Kx)
        ]
        beta_h = self.beta_h
        emas_h_new = [
            beta_h[k] * emas_h[k] + (1.0 - beta_h[k]) * h_t_in
            for k in range(self.Kh)
        ]

        if self.mode_x == "concat":
            aug_x = torch.cat([x_t] + emas_x_new, dim=-1)
        else:
            aug_x = torch.cat([x_t] + [e - x_t for e in emas_x_new], dim=-1)

        if self.mode_h == "concat":
            aug_h = torch.cat([h_t_in] + emas_h_new, dim=-1)
        else:
            aug_h = torch.cat([h_t_in] + [e - h_t_in for e in emas_h_new], dim=-1)

        z = torch.cat([aug_x, aug_h], dim=-1)
        z_norm = self.layer_norm(z)

        # CfC closed-form on LN-normalized z (UNCHANGED from round 179).
        f = self.f_gate(z_norm)
        g = self.g_branch(z_norm)
        h_branch = self.h_branch(z_norm)
        if isinstance(dt, torch.Tensor):
            dt_b = dt
            if dt_b.dim() < 2:
                dt_b = dt_b.unsqueeze(-1)
            tau_eff = torch.exp(-f * dt_b / torch.abs(self.time_scale))
        else:
            tau_eff = torch.exp(-f * float(dt) / torch.abs(self.time_scale))
        h_cfc = tau_eff * g + (1.0 - tau_eff) * h_branch

        # NEW (round 185): gated residual ON TOP of CfC.
        alpha = self.alpha
        h_residual = self.residual_proj(z_norm)
        h_new = h_cfc + alpha * h_residual

        return h_new, emas_x_new, emas_h_new


class LearnedBetaPSLNGatedResidualCfCStackedNetwork(nn.Module):
    """Stacked LearnedBetaPS+LN+GatedResidual-CfC with optional Kh ladder."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=3,
        Kh_ladder=None,
        Kx=5,
        mode_x="diff",
        mode_h="diff",
        beta_x_init=0.75,
        beta_h_init=0.75,
        return_sequences=True,
        alpha_init=0.1,
        residual_init=0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.Kx = Kx
        self.Kh_ladder = list(Kh_ladder) if Kh_ladder is not None else [3] * num_layers
        assert len(self.Kh_ladder) == num_layers
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else hidden_size
            self.cells.append(
                LearnedBetaPSLNGatedResidualCfCCell(
                    in_size, hidden_size, Kx, self.Kh_ladder[l],
                    mode_x=mode_x, mode_h=mode_h,
                    beta_x_init=beta_x_init,
                    beta_h_init=beta_h_init,
                    alpha_init=alpha_init,
                    residual_init=residual_init,
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
            [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.Kh_ladder[l])]
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


def make_lbps_lngr_h3_75(input_size, hidden_size, output_size, num_layers=3):
    """Kh=3 + LN + gated residual (control)."""
    return LearnedBetaPSLNGatedResidualCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[3] * num_layers, Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
        alpha_init=0.1, residual_init=0.1,
    )


def make_lbps_lngr_h2_75(input_size, hidden_size, output_size, num_layers=3):
    """Kh=2 + LN + gated residual."""
    return LearnedBetaPSLNGatedResidualCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[2] * num_layers, Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
        alpha_init=0.1, residual_init=0.1,
    )


def make_lbps_lngr_h5_75(input_size, hidden_size, output_size, num_layers=3):
    """Kh=5 + LN + gated residual."""
    return LearnedBetaPSLNGatedResidualCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[5] * num_layers, Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
        alpha_init=0.1, residual_init=0.1,
    )


__all__ = [
    "LearnedBetaPSLNGatedResidualCfCCell",
    "LearnedBetaPSLNGatedResidualCfCStackedNetwork",
    "make_lbps_lngr_h3_75",
    "make_lbps_lngr_h2_75",
    "make_lbps_lngr_h5_75",
]
