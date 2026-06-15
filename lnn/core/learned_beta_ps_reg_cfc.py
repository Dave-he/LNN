"""LearnedBetaPS+Regularization-CfC (Per-Scale Learnable β + L2 Reg) (PRD #10-137, Round 175, 2026-06-16).

Variant of round 171's LearnedPerScaleBeta-CfC with **L2
regularization on β** — penalize deviation of β from a target
value (default 0.75).

Hypothesis:
- H1 (positive): regularization prevents extreme β (overfitting)
- H2 (negative): extreme β values are useful (round 171 found them)
- H3 (mixed): regularization helps structured (preserves mode
  boundaries), hurts sin (β needs to be free for uniform data)

Mechanism::

    For each layer:
        # Per-scale learned β (round 171):
        beta_h_k = sigmoid(beta_h_k_raw)  # shape [Kh]
        # L2 penalty on β deviation from target:
        reg_loss = λ * mean((beta_h - target) ** 2) +
                   λ * mean((beta_x - target) ** 2)
        # Add to task loss:
        total_loss = task_loss + reg_loss

Audit context (91-174): 43 strictly positive + 18 target-dep +
37 negatives = 98 mechanism classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.learned_beta_ps_cfc import LearnedBetaPSCfCCell


# ---------------------------------------------------------------------------
# Stacked network with β regularization
# ---------------------------------------------------------------------------


class LearnedBetaPSRegCfCStackedNetwork(nn.Module):
    """Stacked LearnedPerScaleBeta-CfC with L2 regularization on β."""

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
        beta_target=0.75,
        reg_lambda=0.01,
        return_sequences=True,
    ):
        """Initialize network.

        Args:
            input_size: number of input features.
            hidden_size: number of hidden units.
            output_size: number of output features.
            num_layers: number of layers.
            Kx: number of input-side EMA scales (shared).
            Kh: number of hidden-side EMA scales (shared).
            mode_x: 'diff' or 'concat' for x-side.
            mode_h: 'diff' or 'concat' for h-side.
            beta_x_init: initial scalar β value for x-side.
            beta_h_init: initial scalar β value for h-side.
            beta_target: target value for β (penalty center).
            reg_lambda: regularization strength (L2 penalty weight).
            return_sequences: if True, return all T outputs.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.Kx = Kx
        self.Kh = Kh
        self.beta_target = beta_target
        self.reg_lambda = reg_lambda
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
        """Forward a full sequence."""
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

    def reg_loss(self):
        """L2 penalty on deviation of β from target.

        Returns:
            scalar tensor with the regularization loss
        """
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for cell in self.cells:
            reg = reg + torch.mean((cell.beta_x - self.beta_target) ** 2)
            reg = reg + torch.mean((cell.beta_h - self.beta_target) ** 2)
        reg = reg / (2.0 * self.num_layers)  # average over layers and {x,h}
        return self.reg_lambda * reg


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def make_lbps_reg_l01(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kh=3, λ=0.01 (mild reg)."""
    return LearnedBetaPSRegCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=3,
        reg_lambda=0.01, return_sequences=True,
    )


def make_lbps_reg_l001(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kh=3, λ=0.001 (very mild reg)."""
    return LearnedBetaPSRegCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=3,
        reg_lambda=0.001, return_sequences=True,
    )


def make_lbps_reg_l1(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kh=3, λ=1.0 (strong reg)."""
    return LearnedBetaPSRegCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=3,
        reg_lambda=1.0, return_sequences=True,
    )


def make_lbps_reg_l10(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kh=3, λ=10.0 (very strong reg)."""
    return LearnedBetaPSRegCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=3,
        reg_lambda=10.0, return_sequences=True,
    )


def make_lbps_reg_kh2_l01(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kh=2, λ=0.01 (round 171 sin winner + mild reg)."""
    return LearnedBetaPSRegCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=2,
        reg_lambda=0.01, return_sequences=True,
    )


def make_lbps_reg_kh5_l01(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kh=5, λ=0.01 (round 171 structured winner + mild reg)."""
    return LearnedBetaPSRegCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=5,
        reg_lambda=0.01, return_sequences=True,
    )


__all__ = [
    "LearnedBetaPSRegCfCStackedNetwork",
    "make_lbps_reg_l01",
    "make_lbps_reg_l001",
    "make_lbps_reg_l1",
    "make_lbps_reg_l10",
    "make_lbps_reg_kh2_l01",
    "make_lbps_reg_kh5_l01",
]
