"""HybridBeta-XH-Deep-4Layer-CfC (4-Layer Stacked) (PRD #10-128, Round 166, 2026-06-15).

Re-exports round 163's HybridBetaXHCfCStackedNetwork with
num_layers=4. Tests if 4-layer stacking compounds the benefits
of hybrid β (per-feature on x, scalar on h) beyond round 165's
3-layer winner.

Mechanism::

    # Same as round 165, but with 4 layers instead of 3.
    # For each of 4 layers:
    beta_x_k,d = sigmoid(beta_x_k_raw[d])  # shape [Kx, D]
    beta_h_k = fixed (e.g. 0.7, 0.95)
    ema_x_k,t[d] = beta_x_k,d * ema_x_k,t-1[d] + (1 - beta_x_k,d) * x_t[d]
    ema_h_k,t[d] = beta_h_k * ema_h_k,t-1[d] + (1 - beta_h_k) * h_t[d]
    z_t = cat(aug_x_t, aug_h_t)
    h_t = CfC(z_t)

This is a PARAMETER-ONLY test (num_layers=4 instead of 3) — no
new core code, just deeper stacking of round 165's winner.

Audit context (91-165): 39 strictly positive + 17 target-dep +
35 negatives = 91 mechanism classes.
"""
from __future__ import annotations

from lnn.core.hybrid_beta_xh_cfc import HybridBetaXHCfCStackedNetwork


# ---------------------------------------------------------------------------
# Factory functions for 4-layer variants
# ---------------------------------------------------------------------------


def make_hb_xh_4layer_h1(input_size, hidden_size, output_size, num_layers=4, return_sequences=True):
    """4-layer, Kx=1, Kh=1, scalar β=0.9 on h."""
    return HybridBetaXHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=1, Kh=1, betas_h=[0.9],
        return_sequences=return_sequences,
    )


def make_hb_xh_4layer_h2(input_size, hidden_size, output_size, num_layers=4, return_sequences=True):
    """4-layer, Kx=2, Kh=2, scalar β ∈ {0.7, 0.95} on h."""
    return HybridBetaXHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=2, Kh=2, betas_h=[0.7, 0.95],
        return_sequences=return_sequences,
    )


def make_hb_xh_4layer_h2_3x(input_size, hidden_size, output_size, num_layers=4, return_sequences=True):
    """4-layer, Kx=3, Kh=2, scalar β ∈ {0.7, 0.95} on h."""
    return HybridBetaXHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=3, Kh=2, betas_h=[0.7, 0.95],
        return_sequences=return_sequences,
    )


def make_hb_xh_4layer_h2_k5(input_size, hidden_size, output_size, num_layers=4, return_sequences=True):
    """4-layer, Kx=5, Kh=2, scalar β ∈ {0.7, 0.95} on h (round 165 best config)."""
    return HybridBetaXHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, Kh=2, betas_h=[0.7, 0.95],
        return_sequences=return_sequences,
    )


__all__ = [
    "HybridBetaXHCfCStackedNetwork",
    "make_hb_xh_4layer_h1",
    "make_hb_xh_4layer_h2",
    "make_hb_xh_4layer_h2_3x",
    "make_hb_xh_4layer_h2_k5",
]
