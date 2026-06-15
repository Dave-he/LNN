"""LearnedBetaPS+LN+Khl-CfC (Per-Scale Learnable β + LayerNorm + Kh Ladder) (PRD #10-142, Round 180, 2026-06-16).

Combines round 179 (LayerNorm SOTA) with round 173 (Kh ladder).

Hypothesis:
- H1 (positive): Kh ladder [2,3,5] + LN beats round 179 on structured
- H2 (negative): Kh=2 + LN is already optimal, ladder adds noise
- H3 (mixed): Kh ladder helps structured but regresses sin

Audit context (91-179): 44 strictly positive + 18 target-dep +
41 negatives = 103 mechanism classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.learned_beta_ps_ln_cfc import LearnedBetaPSLNCfCCell


class LearnedBetaPSLNKhlCfCStackedNetwork(nn.Module):
    """Stacked LearnedBetaPS+LN-CfC with per-layer Kh ladder."""

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
                LearnedBetaPSLNCfCCell(
                    in_size, hidden_size, Kx, self.Kh_ladder[l],
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


def make_lbps_ln_khl_2_2_2(input_size, hidden_size, output_size, num_layers=3):
    """Kh=[2,2,2] (control, round 179 winner)."""
    return LearnedBetaPSLNKhlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[2, 2, 2], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_ln_khl_2_3_5(input_size, hidden_size, output_size, num_layers=3):
    """Kh=[2,3,5] (round 173 winner + LN)."""
    return LearnedBetaPSLNKhlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[2, 3, 5], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_ln_khl_2_3_3(input_size, hidden_size, output_size, num_layers=3):
    """Kh=[2,3,3] (mild ladder)."""
    return LearnedBetaPSLNKhlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[2, 3, 3], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_ln_khl_3_3_3(input_size, hidden_size, output_size, num_layers=3):
    """Kh=[3,3,3] (Kh=3 + LN)."""
    return LearnedBetaPSLNKhlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[3, 3, 3], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_ln_khl_5_3_2(input_size, hidden_size, output_size, num_layers=3):
    """Kh=[5,3,2] (high-to-low)."""
    return LearnedBetaPSLNKhlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[5, 3, 2], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_ln_khl_2_5_2(input_size, hidden_size, output_size, num_layers=3):
    """Kh=[2,5,2] (low-high-low, structured-friendly)."""
    return LearnedBetaPSLNKhlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[2, 5, 2], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


__all__ = [
    "LearnedBetaPSLNKhlCfCStackedNetwork",
    "make_lbps_ln_khl_2_2_2",
    "make_lbps_ln_khl_2_3_5",
    "make_lbps_ln_khl_2_3_3",
    "make_lbps_ln_khl_3_3_3",
    "make_lbps_ln_khl_5_3_2",
    "make_lbps_ln_khl_2_5_2",
]
