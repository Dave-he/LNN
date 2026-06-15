"""LearnedBetaPS+KhLadder-CfC (Per-Scale Learnable β + Kh Ladder) (PRD #10-135, Round 173, 2026-06-16).

Variant of round 171's LearnedPerScaleBeta-CfC with **different
Kh per layer** (Kh ladder). Tests if varying Kh across layers
beats constant Kh (round 171 winners).

Round 171 found:
- lb_ps_h2 (Kh=2 constant) wins sin -76%
- lb_ps_h5 (Kh=5 constant) wins structured -92%

This round tests if a Kh LADDER (different Kh per layer) can
beat constant Kh. E.g., layer 0 = Kh=5, layer 1 = Kh=3,
layer 2 = Kh=2.

Hypothesis:
- H1 (positive): Kh ladder combines strengths of different Kh
- H2 (negative): constant Kh is optimal
- H3 (mixed): Kh ladder wins structured (more diversity)

Mechanism::

    For each layer l (Kh[l] varies):
        # Per-scale learned β (round 171):
        beta_h_k = sigmoid(beta_h_k_raw)  # shape [Kh[l]]
        # Per-sample EMAs:
        ema_h_k,t[b,h] = beta_h_k * ema_h_k,t-1[b,h] + (1 - beta_h_k) * h_t[b,h]
        z_t = cat(aug_x_t, aug_h_t)
        h_t = CfC(z_t)

Audit context (91-172): 43 strictly positive + 17 target-dep +
36 negatives = 96 mechanism classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.learned_beta_ps_cfc import LearnedBetaPSCfCCell


# ---------------------------------------------------------------------------
# Stacked network with per-layer Kh ladder
# ---------------------------------------------------------------------------


class LearnedBetaPSKhlCfCStackedNetwork(nn.Module):
    """Stacked LearnedPerScaleBeta-CfC with different Kh per layer (Kh ladder)."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=3,
        Kx=5,
        Kh_ladder=None,  # list of Kh per layer
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
            Kx: number of input-side EMA scales (shared across layers).
            Kh_ladder: list of num_layers Kh values (one per layer).
                If None, defaults to [3, 3, 3].
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
        self.Kx = Kx
        self.Kh_ladder = list(Kh_ladder) if Kh_ladder is not None else [3] * num_layers
        assert len(self.Kh_ladder) == num_layers, (
            f"Kh_ladder length {len(self.Kh_ladder)} != num_layers {num_layers}"
        )
        self.return_sequences = return_sequences

        # Build cells with per-layer Kh.
        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_size = input_size if layer_idx == 0 else hidden_size
            self.cells.append(
                LearnedBetaPSCfCCell(
                    in_size, hidden_size, Kx, self.Kh_ladder[layer_idx],
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
        # Initialize hidden states and EMAs (different Kh per layer).
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


# ---------------------------------------------------------------------------
# Factory functions: Kh ladder presets
# ---------------------------------------------------------------------------


def make_lbps_khl_3_3_3(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kh=[3,3,3] (round 171 control)."""
    return LearnedBetaPSKhlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh_ladder=[3, 3, 3],
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_khl_2_2_2(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kh=[2,2,2] (round 171 sin winner)."""
    return LearnedBetaPSKhlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh_ladder=[2, 2, 2],
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_khl_5_5_5(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kh=[5,5,5] (round 171 structured winner)."""
    return LearnedBetaPSKhlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh_ladder=[5, 5, 5],
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_khl_5_3_2(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kh=[5,3,2] (high-to-low ladder)."""
    return LearnedBetaPSKhlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh_ladder=[5, 3, 2],
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_khl_2_3_5(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kh=[2,3,5] (low-to-high ladder)."""
    return LearnedBetaPSKhlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh_ladder=[2, 3, 5],
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_khl_3_2_2(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kh=[3,2,2] (high then low, sin-favoring)."""
    return LearnedBetaPSKhlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh_ladder=[3, 2, 2],
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_khl_5_5_2(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kh=[5,5,2] (high then low, structured-favoring)."""
    return LearnedBetaPSKhlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh_ladder=[5, 5, 2],
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


__all__ = [
    "LearnedBetaPSKhlCfCStackedNetwork",
    "make_lbps_khl_3_3_3",
    "make_lbps_khl_2_2_2",
    "make_lbps_khl_5_5_5",
    "make_lbps_khl_5_3_2",
    "make_lbps_khl_2_3_5",
    "make_lbps_khl_3_2_2",
    "make_lbps_khl_5_5_2",
]
