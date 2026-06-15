"""LearnedBetaPS+KxKhGrid-CfC (Per-Scale Learnable β + Kx×Kh Grid) (PRD #10-139, Round 177, 2026-06-16).

Variant of round 171's LearnedPerScaleBeta-CfC with **Kx×Kh
grid sweep** — explore all combinations of Kx (input-side EMA
scales) and Kh (hidden-side EMA scales).

Round 173 found Kh ladder [2,3,5] wins structured (-93% NEW
BEST). Round 176 found Kx=3 wins sin, Kx=7 wins structured.

This round: sweep the full Kx × Kh grid to find optimal combo.

Hypothesis:
- H1 (positive): Kx=3 + Kh=2 (small-small) wins sin
- H2 (positive): Kx=7 + Kh=5 (large-large) wins structured
- H3 (negative): grid combos don't beat round 171 SOTA

Audit context (91-176): 43 strictly positive + 18 target-dep +
39 negatives = 100 mechanism classes.
"""
from __future__ import annotations

from lnn.core.learned_beta_ps_cfc import LearnedBetaPSCfCStackedNetwork


# ---------------------------------------------------------------------------
# Factory functions: Kx × Kh grid (3 × 3 = 9 combos)
# ---------------------------------------------------------------------------


def _make(input_size, hidden_size, output_size, num_layers, Kx, Kh):
    return LearnedBetaPSCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=Kx,
        Kh=Kh,
        mode_x="diff",
        mode_h="diff",
        beta_x_init=0.75,
        beta_h_init=0.75,
        return_sequences=True,
    )


def make_lbps_grid_3_2(input_size, hidden_size, output_size, num_layers=3):
    """Kx=3, Kh=2 (small-small, sin-favoring)."""
    return _make(input_size, hidden_size, output_size, num_layers, Kx=3, Kh=2)


def make_lbps_grid_3_3(input_size, hidden_size, output_size, num_layers=3):
    """Kx=3, Kh=3 (small-Kh-3)."""
    return _make(input_size, hidden_size, output_size, num_layers, Kx=3, Kh=3)


def make_lbps_grid_3_5(input_size, hidden_size, output_size, num_layers=3):
    """Kx=3, Kh=5 (small-Kx, large-Kh)."""
    return _make(input_size, hidden_size, output_size, num_layers, Kx=3, Kh=5)


def make_lbps_grid_5_2(input_size, hidden_size, output_size, num_layers=3):
    """Kx=5, Kh=2 (large-Kx, small-Kh)."""
    return _make(input_size, hidden_size, output_size, num_layers, Kx=5, Kh=2)


def make_lbps_grid_5_3(input_size, hidden_size, output_size, num_layers=3):
    """Kx=5, Kh=3 (control, round 171 default)."""
    return _make(input_size, hidden_size, output_size, num_layers, Kx=5, Kh=3)


def make_lbps_grid_5_5(input_size, hidden_size, output_size, num_layers=3):
    """Kx=5, Kh=5 (both large)."""
    return _make(input_size, hidden_size, output_size, num_layers, Kx=5, Kh=5)


def make_lbps_grid_7_2(input_size, hidden_size, output_size, num_layers=3):
    """Kx=7, Kh=2 (large-Kx, small-Kh)."""
    return _make(input_size, hidden_size, output_size, num_layers, Kx=7, Kh=2)


def make_lbps_grid_7_3(input_size, hidden_size, output_size, num_layers=3):
    """Kx=7, Kh=3 (large-Kx, Kh=3)."""
    return _make(input_size, hidden_size, output_size, num_layers, Kx=7, Kh=3)


def make_lbps_grid_7_5(input_size, hidden_size, output_size, num_layers=3):
    """Kx=7, Kh=5 (large-large, structured-favoring)."""
    return _make(input_size, hidden_size, output_size, num_layers, Kx=7, Kh=5)


__all__ = [
    "make_lbps_grid_3_2",
    "make_lbps_grid_3_3",
    "make_lbps_grid_3_5",
    "make_lbps_grid_5_2",
    "make_lbps_grid_5_3",
    "make_lbps_grid_5_5",
    "make_lbps_grid_7_2",
    "make_lbps_grid_7_3",
    "make_lbps_grid_7_5",
]
