"""LayerDecay-H3-CfC (Kh=3 with REVERSE β) (PRD #10-130, Round 168, 2026-06-15).

Variant of round 167's LayerDecay-CfC with **Kh=3 or Kh=4** hidden-
side time-scales (instead of Kh=2). Combines REVERSE β schedule
(slow at low layers, fast at high layers) with more hidden-side
time-scales.

Mechanism::

    Same as round 167, but betas_h is a list of 3 or 4 values
    instead of 2.

Schedule modes:
- "constant": all layers use betas_h (control baseline)
- "reverse":  β_l_k = β_max_k - l * (β_max_k - β_min_k) / (L-1)

Audit context (91-167): 40 strictly positive + 17 target-dep +
35 negatives = 92 mechanism classes.
"""
from __future__ import annotations

from lnn.core.layer_decay_cfc import (
    LayerDecayCfCStackedNetwork,
    make_layer_beta_schedule,
)


# ---------------------------------------------------------------------------
# Factory functions for Kh=3 / Kh=4 variants
# ---------------------------------------------------------------------------


def make_ld_reverse_h3_k5(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=3, REVERSE β ∈ [0.99, 0.75, 0.5]."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.99, 0.75, 0.5], mode="reverse",
        return_sequences=True,
    )


def make_ld_reverse_h4_k5(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=4, REVERSE β ∈ [0.99, 0.83, 0.66, 0.5]."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.99, 0.83, 0.66, 0.5], mode="reverse",
        return_sequences=True,
    )


def make_ld_reverse_h3_wider(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=3, REVERSE β ∈ [0.999, 0.7, 0.3] (wider range)."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.999, 0.7, 0.3], mode="reverse",
        return_sequences=True,
    )


def make_ld_reverse_h3_k6(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=6, Kh=3, REVERSE β ∈ [0.99, 0.75, 0.5]."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=6, betas_h=[0.99, 0.75, 0.5], mode="reverse",
        return_sequences=True,
    )


def make_ld_reverse_h3_h2(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=3, REVERSE β ∈ [0.95, 0.85, 0.7] (narrow range)."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.95, 0.85, 0.7], mode="reverse",
        return_sequences=True,
    )


def make_ld_constant_h3(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=3, constant β ∈ {0.7, 0.85, 0.95} (control)."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.7, 0.85, 0.95], mode="constant",
        return_sequences=True,
    )


__all__ = [
    "LayerDecayCfCStackedNetwork",
    "make_layer_beta_schedule",
    "make_ld_reverse_h3_k5",
    "make_ld_reverse_h4_k5",
    "make_ld_reverse_h3_wider",
    "make_ld_reverse_h3_k6",
    "make_ld_reverse_h3_h2",
    "make_ld_constant_h3",
]
