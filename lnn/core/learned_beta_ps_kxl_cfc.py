"""LearnedBetaPS+KxLadder-CfC (Per-Scale Learnable β + Kx Ladder) (PRD #10-138, Round 176, 2026-06-16).

Variant of round 171's LearnedPerScaleBeta-CfC with **per-layer
Kx ladder** — different Kx (input-side EMA scales) per layer.

Round 173 tested Kh ladder (h-side): Kh=[2,3,5] won structured
(-93% NEW BEST). This round tests Kx ladder (x-side).

Hypothesis:
- H1 (positive): Kx ladder helps (different input processing
  per layer)
- H2 (negative): Kx=5 constant is optimal
- H3 (mixed): Kx ladder helps structured (more input scales
  for multi-mode data)

Mechanism::

    For each layer l (Kx[l] varies):
        # Per-scale learned β (round 171):
        beta_x_k = sigmoid(beta_x_k_raw)  # shape [Kx[l]]
        # Per-sample x-EMAs:
        ema_x_k,t[b,d] = beta_x_k * ema_x_k,t-1[b,d] + (1 - beta_x_k) * x_t[b,d]
        z_t = cat(aug_x_t, aug_h_t)
        h_t = CfC(z_t)

Audit context (91-175): 43 strictly positive + 18 target-dep +
38 negatives = 99 mechanism classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.learned_beta_ps_cfc import LearnedBetaPSCfCCell


# ---------------------------------------------------------------------------
# Stacked network with per-layer Kx ladder
# ---------------------------------------------------------------------------


class LearnedBetaPSKxlCfCStackedNetwork(nn.Module):
    """Stacked LearnedPerScaleBeta-CfC with per-layer Kx ladder."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=3,
        Kx_ladder=None,  # list of Kx per layer
        Kh=3,
        mode_x="diff",
        mode_h="diff",
        beta_x_init=0.75,
        beta_h_init=0.75,
        return_sequences=True,
    ):
        """Initialize network.

        Args:
            input_size: number of input features.
            hidden_size: number of hidden units.
            output_size: number of output features.
            num_layers: number of layers.
            Kx_ladder: list of num_layers Kx values (one per layer).
                If None, defaults to [5, 5, 5].
            Kh: number of hidden-side EMA scales (shared).
            mode_x: 'diff' or 'concat' for x-side.
            mode_h: 'diff' or 'concat' for h-side.
            beta_x_init: initial scalar β value for x-side.
            beta_h_init: initial scalar β value for h-side.
            return_sequences: if True, return all T outputs.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.Kh = Kh
        self.Kx_ladder = list(Kx_ladder) if Kx_ladder is not None else [5] * num_layers
        assert len(self.Kx_ladder) == num_layers
        self.return_sequences = return_sequences

        # Build cells with per-layer Kx.
        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_size = input_size if layer_idx == 0 else hidden_size
            self.cells.append(
                LearnedBetaPSCfCCell(
                    in_size, hidden_size, self.Kx_ladder[layer_idx], Kh,
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
            [torch.zeros(B, self.cells[l].input_size, device=device) for _ in range(self.Kx_ladder[l])]
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


def make_lbps_kxl_5_5_5(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=[5,5,5] (round 171 control)."""
    return LearnedBetaPSKxlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx_ladder=[5, 5, 5], Kh=3,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_kxl_3_3_3(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=[3,3,3] (smaller Kx)."""
    return LearnedBetaPSKxlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx_ladder=[3, 3, 3], Kh=3,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_kxl_7_7_7(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=[7,7,7] (larger Kx)."""
    return LearnedBetaPSKxlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx_ladder=[7, 7, 7], Kh=3,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_kxl_3_5_7(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=[3,5,7] (low-to-high ladder)."""
    return LearnedBetaPSKxlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx_ladder=[3, 5, 7], Kh=3,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_kxl_7_5_3(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=[7,5,3] (high-to-low ladder)."""
    return LearnedBetaPSKxlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx_ladder=[7, 5, 3], Kh=3,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_kxl_3_5_5(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=[3,5,5] (sin-favoring)."""
    return LearnedBetaPSKxlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx_ladder=[3, 5, 5], Kh=3,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_kxl_7_5_5(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=[7,5,5] (structured-favoring)."""
    return LearnedBetaPSKxlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx_ladder=[7, 5, 5], Kh=3,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


__all__ = [
    "LearnedBetaPSKxlCfCStackedNetwork",
    "make_lbps_kxl_5_5_5",
    "make_lbps_kxl_3_3_3",
    "make_lbps_kxl_7_7_7",
    "make_lbps_kxl_3_5_7",
    "make_lbps_kxl_7_5_3",
    "make_lbps_kxl_3_5_5",
    "make_lbps_kxl_7_5_5",
]
