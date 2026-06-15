"""HybridBeta-XH-Deep-CfC (3-Layer Stacked) (PRD #10-126, Round 164, 2026-06-15).

Re-exports round 163's HybridBetaXHCfCStackedNetwork with
num_layers=3. Tests if DEEPER cells (3 layers vs 2) compound
the benefits of hybrid β (per-feature on x, scalar on h).

Mechanism::

    # Same as round 163, but with 3 stacked cells instead of 2.
    # For each of 3 layers:
    beta_x_k,d = sigmoid(beta_x_k_raw[d])  # shape [Kx, D]
    beta_h_k = fixed (e.g. 0.7, 0.95)
    ema_x_k,t[d] = beta_x_k,d * ema_x_k,t-1[d] + (1 - beta_x_k,d) * x_t[d]
    ema_h_k,t[d] = beta_h_k * ema_h_k,t-1[d] + (1 - beta_h_k) * h_t[d]
    z_t = cat(aug_x_t, aug_h_t)
    h_t = CfC(z_t)

This is a PARAMETER-ONLY test (num_layers=3 instead of 2) — no
new core code, just deeper stacking of round 163's winners.

Audit context (91-163): 31 strictly positive + 17 target-dep +
34 negatives = 82 mechanism classes.
"""
from __future__ import annotations

from lnn.core.hybrid_beta_xh_cfc import (
    HybridBetaXHCfCCell,
    HybridBetaXHCfCStackedNetwork,
)


# ---------------------------------------------------------------------------
# Factory functions for 3-layer variants
# ---------------------------------------------------------------------------


def make_hb_xh_deep_h1(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=1, Kh=1, scalar β=0.9 on h."""
    return HybridBetaXHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=1, Kh=1, betas_h=[0.9],
        return_sequences=True,
    )


def make_hb_xh_deep_h2(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=2, Kh=2, scalar β ∈ {0.7, 0.95} on h."""
    return HybridBetaXHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=2, Kh=2, betas_h=[0.7, 0.95],
        return_sequences=True,
    )


def make_hb_xh_deep_h2_3x(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=3, Kh=2, scalar β ∈ {0.7, 0.95} on h."""
    return HybridBetaXHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=3, Kh=2, betas_h=[0.7, 0.95],
        return_sequences=True,
    )


def make_hb_xh_deep_best(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=3, Kh=2, scalar β ∈ {0.7, 0.95} on h, both diff."""
    return HybridBetaXHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=3, Kh=2, betas_h=[0.7, 0.95],
        return_sequences=True,
    )


__all__ = [
    "HybridBetaXHCfCCell",
    "HybridBetaXHCfCStackedNetwork",
    "make_hb_xh_deep_h1",
    "make_hb_xh_deep_h2",
    "make_hb_xh_deep_h2_3x",
    "make_hb_xh_deep_best",
]
