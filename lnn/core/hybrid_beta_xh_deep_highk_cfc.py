"""HybridBeta-XH-Deep-HighK-CfC (3-Layer + High K) (PRD #10-127, Round 165, 2026-06-15).

Re-exports round 163's HybridBetaXHCfCStackedNetwork with
num_layers=3 and HIGHER Kx (4 or 5). Tests if more time-scales
at 3 layers pushes structured even further than round 164.

Mechanism::

    # Same as round 164, but with Kx=4 or Kx=5.
    # For each of 3 layers:
    beta_x_k,d = sigmoid(beta_x_k_raw[d])  # shape [Kx, D], Kx=4 or 5
    beta_h_k = fixed (e.g. 0.7, 0.95)
    ema_x_k,t[d] = beta_x_k,d * ema_x_k,t-1[d] + (1 - beta_x_k,d) * x_t[d]
    ema_h_k,t[d] = beta_h_k * ema_h_k,t-1[d] + (1 - beta_h_k) * h_t[d]
    z_t = cat(aug_x_t, aug_h_t)
    h_t = CfC(z_t)

This is a PARAMETER-ONLY test (Kx=4 or Kx=5, num_layers=3) — no
new core code, just higher K of round 164's winner.

Audit context (91-164): 35 strictly positive + 17 target-dep +
35 negatives = 87 mechanism classes.
"""
from __future__ import annotations

from lnn.core.hybrid_beta_xh_cfc import HybridBetaXHCfCStackedNetwork


# ---------------------------------------------------------------------------
# Factory functions for 3-layer + high-K variants
# ---------------------------------------------------------------------------


def make_hb_xh_deep_h1_k4(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=4, Kh=1, scalar β=0.9 on h."""
    return HybridBetaXHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=4, Kh=1, betas_h=[0.9],
        return_sequences=True,
    )


def make_hb_xh_deep_h1_k5(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=1, scalar β=0.9 on h."""
    return HybridBetaXHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, Kh=1, betas_h=[0.9],
        return_sequences=True,
    )


def make_hb_xh_deep_h2_k4(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=4, Kh=2, scalar β ∈ {0.7, 0.95} on h."""
    return HybridBetaXHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=4, Kh=2, betas_h=[0.7, 0.95],
        return_sequences=True,
    )


def make_hb_xh_deep_h2_k5(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=2, scalar β ∈ {0.7, 0.95} on h."""
    return HybridBetaXHCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, Kh=2, betas_h=[0.7, 0.95],
        return_sequences=True,
    )


__all__ = [
    "HybridBetaXHCfCStackedNetwork",
    "make_hb_xh_deep_h1_k4",
    "make_hb_xh_deep_h1_k5",
    "make_hb_xh_deep_h2_k4",
    "make_hb_xh_deep_h2_k5",
]
