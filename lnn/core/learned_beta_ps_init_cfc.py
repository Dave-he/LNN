"""LearnedBetaPS+PerLayerInit-CfC (Per-Scale Learnable β + Per-Layer Init) (PRD #10-136, Round 174, 2026-06-16).

Variant of round 171's LearnedPerScaleBeta-CfC with **per-layer
β_init** — each layer gets a different initial β value, but β
remains fully learnable (no schedule constraint).

Round 172's per-layer SCHEDULE over-constrained the model
(constant = no schedule, REVERSE = constrains β by factor 0.5-1.0).
This round tests a lighter touch: per-layer INITIALIZATION only.

Round 173's Kh ladder [2,3,5] won structured (-93% NEW BEST) but
regressed sin. Per-layer init is an alternative: same Kh=2
const, different init per layer.

Hypothesis:
- H1 (positive): per-layer init gives each layer a different
  starting point, gradient finds different β per layer
- H2 (negative): per-layer init doesn't matter (gradient ignores)
- H3 (mixed): per-layer init helps structured (mimics Kh ladder)

Mechanism::

    For each layer l:
        # Per-scale learned β with PER-LAYER init (round 174):
        beta_x_k_raw (nn.Parameter, init from per-layer β_init_x[l])
        beta_h_k_raw (nn.Parameter, init from per-layer β_init_h[l])
        # No schedule constraint — β is fully free
        # Per-sample EMAs:
        ema_x_k,t[b,d] = sigmoid(beta_x_k_raw)[k] * ema_x_k,t-1[b,d] + (1 - sigmoid(...)) * x_t[b,d]
        z_t = cat(aug_x_t, aug_h_t)
        h_t = CfC(z_t)

Audit context (91-173): 43 strictly positive + 18 target-dep +
36 negatives = 97 mechanism classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.learned_beta_ps_cfc import LearnedBetaPSCfCCell


# ---------------------------------------------------------------------------
# Stacked network with per-layer β init
# ---------------------------------------------------------------------------


class LearnedBetaPSInitCfCStackedNetwork(nn.Module):
    """Stacked LearnedPerScaleBeta-CfC with per-layer β_init."""

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
        beta_x_inits=None,  # list of num_layers β_init for x-side
        beta_h_inits=None,  # list of num_layers β_init for h-side
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
            beta_x_inits: list of num_layers β_init for x-side (or None → 0.75).
            beta_h_inits: list of num_layers β_init for h-side (or None → 0.75).
            return_sequences: if True, return all T outputs.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.Kx = Kx
        self.Kh = Kh
        self.return_sequences = return_sequences

        # Default to constant 0.75 if not provided.
        if beta_x_inits is None:
            beta_x_inits = [0.75] * num_layers
        if beta_h_inits is None:
            beta_h_inits = [0.75] * num_layers
        assert len(beta_x_inits) == num_layers
        assert len(beta_h_inits) == num_layers
        self.beta_x_inits = list(beta_x_inits)
        self.beta_h_inits = list(beta_h_inits)

        # Build cells with per-layer β_init.
        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_size = input_size if layer_idx == 0 else hidden_size
            self.cells.append(
                LearnedBetaPSCfCCell(
                    in_size, hidden_size, Kx, Kh,
                    mode_x=mode_x, mode_h=mode_h,
                    beta_x_init=beta_x_inits[layer_idx],
                    beta_h_init=beta_h_inits[layer_idx],
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


# ---------------------------------------------------------------------------
# Factory functions: per-layer β init presets
# ---------------------------------------------------------------------------


def make_lbps_init_uniform(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, all β_init=0.75 (round 171 control)."""
    return LearnedBetaPSInitCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=3,
        beta_x_inits=[0.75] * num_layers,
        beta_h_inits=[0.75] * num_layers,
        return_sequences=True,
    )


def make_lbps_init_low_to_high(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, β_init ascending [0.5, 0.75, 0.95] (low-to-high)."""
    return LearnedBetaPSInitCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=3,
        beta_x_inits=[0.5, 0.75, 0.95],
        beta_h_inits=[0.5, 0.75, 0.95],
        return_sequences=True,
    )


def make_lbps_init_high_to_low(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, β_init descending [0.95, 0.75, 0.5] (high-to-low)."""
    return LearnedBetaPSInitCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=3,
        beta_x_inits=[0.95, 0.75, 0.5],
        beta_h_inits=[0.95, 0.75, 0.5],
        return_sequences=True,
    )


def make_lbps_init_wide(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, β_init wide spread [0.5, 0.85, 0.99]."""
    return LearnedBetaPSInitCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=3,
        beta_x_inits=[0.5, 0.85, 0.99],
        beta_h_inits=[0.5, 0.85, 0.99],
        return_sequences=True,
    )


def make_lbps_init_narrow(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, β_init narrow spread [0.7, 0.75, 0.8]."""
    return LearnedBetaPSInitCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=3,
        beta_x_inits=[0.7, 0.75, 0.8],
        beta_h_inits=[0.7, 0.75, 0.8],
        return_sequences=True,
    )


def make_lbps_init_kh2_low_to_high(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kh=2, β_init ascending [0.5, 0.75, 0.95] (round 171 sin winner)."""
    return LearnedBetaPSInitCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=2,
        beta_x_inits=[0.5, 0.75, 0.95],
        beta_h_inits=[0.5, 0.75, 0.95],
        return_sequences=True,
    )


def make_lbps_init_kh2_high_to_low(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kh=2, β_init descending [0.95, 0.75, 0.5]."""
    return LearnedBetaPSInitCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kx=5, Kh=2,
        beta_x_inits=[0.95, 0.75, 0.5],
        beta_h_inits=[0.95, 0.75, 0.5],
        return_sequences=True,
    )


__all__ = [
    "LearnedBetaPSInitCfCStackedNetwork",
    "make_lbps_init_uniform",
    "make_lbps_init_low_to_high",
    "make_lbps_init_high_to_low",
    "make_lbps_init_wide",
    "make_lbps_init_narrow",
    "make_lbps_init_kh2_low_to_high",
    "make_lbps_init_kh2_high_to_low",
]
