"""LearnedBetaPS+LN+KxKh-CfC (Per-Scale Learnable β + LayerNorm + KxKh Combined Ladder) (PRD #10-143, Round 181, 2026-06-16).

Combines round 179 (LN), round 173 (Kh ladder), round 176 (Kx
ladder) — both Kh AND Kx vary per layer.

Hypothesis:
- H1 (positive): combined Kx×Kh ladder beats single-dim ladders
- H2 (negative): LN already captures scale info, ladder adds noise
- H3 (mixed): combined helps structured (multi-mode) but
  regresses sin (simpler data)

Audit context (91-180): 45 strictly positive + 18 target-dep +
41 negatives = 104 mechanism classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.learned_beta_ps_ln_cfc import LearnedBetaPSLNCfCCell


class LearnedBetaPSLNKxKhCfCStackedNetwork(nn.Module):
    """Stacked LearnedBetaPS+LN-CfC with per-layer Kx AND Kh ladders."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=3,
        Kx_ladder=None,
        Kh_ladder=None,
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
        self.Kx_ladder = list(Kx_ladder) if Kx_ladder is not None else [5] * num_layers
        self.Kh_ladder = list(Kh_ladder) if Kh_ladder is not None else [3] * num_layers
        assert len(self.Kx_ladder) == num_layers
        assert len(self.Kh_ladder) == num_layers
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else hidden_size
            self.cells.append(
                LearnedBetaPSLNCfCCell(
                    in_size, hidden_size, self.Kx_ladder[l], self.Kh_ladder[l],
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
            [torch.zeros(B, self.cells[l].input_size, device=device) for _ in range(self.Kx_ladder[l])]
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


def make_lbps_ln_kxkh_5_5_5_2_5_2(input_size, hidden_size, output_size, num_layers=3):
    """Kx=[5,5,5] Kh=[2,5,2] (round 180 sin winner)."""
    return LearnedBetaPSLNKxKhCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx_ladder=[5, 5, 5], Kh_ladder=[2, 5, 2],
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_ln_kxkh_5_5_5_5_3_2(input_size, hidden_size, output_size, num_layers=3):
    """Kx=[5,5,5] Kh=[5,3,2] (round 180 structured winner)."""
    return LearnedBetaPSLNKxKhCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx_ladder=[5, 5, 5], Kh_ladder=[5, 3, 2],
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_ln_kxkh_3_5_7_2_5_2(input_size, hidden_size, output_size, num_layers=3):
    """Kx=[3,5,7] Kh=[2,5,2] (Kx ladder + sin Kh)."""
    return LearnedBetaPSLNKxKhCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx_ladder=[3, 5, 7], Kh_ladder=[2, 5, 2],
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_ln_kxkh_7_5_3_5_3_2(input_size, hidden_size, output_size, num_layers=3):
    """Kx=[7,5,3] Kh=[5,3,2] (Kx reversed + structured Kh)."""
    return LearnedBetaPSLNKxKhCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx_ladder=[7, 5, 3], Kh_ladder=[5, 3, 2],
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_ln_kxkh_3_5_7_5_3_2(input_size, hidden_size, output_size, num_layers=3):
    """Kx=[3,5,7] Kh=[5,3,2] (both increasing)."""
    return LearnedBetaPSLNKxKhCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx_ladder=[3, 5, 7], Kh_ladder=[5, 3, 2],
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_ln_kxkh_7_5_3_2_5_2(input_size, hidden_size, output_size, num_layers=3):
    """Kx=[7,5,3] Kh=[2,5,2] (Kx reversed + sin Kh)."""
    return LearnedBetaPSLNKxKhCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx_ladder=[7, 5, 3], Kh_ladder=[2, 5, 2],
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


__all__ = [
    "LearnedBetaPSLNKxKhCfCStackedNetwork",
    "make_lbps_ln_kxkh_5_5_5_2_5_2",
    "make_lbps_ln_kxkh_5_5_5_5_3_2",
    "make_lbps_ln_kxkh_3_5_7_2_5_2",
    "make_lbps_ln_kxkh_7_5_3_5_3_2",
    "make_lbps_ln_kxkh_3_5_7_5_3_2",
    "make_lbps_ln_kxkh_7_5_3_2_5_2",
]
