"""LayerDecay-H4-CfC (Kh=4 with constant β) (PRD #10-131, Round 169, 2026-06-15).

Variant of round 168's LayerDecay-H3-CfC with **Kh=4 or Kh=5**
hidden-side time-scales. Tests if Kh=4 helps even more than Kh=3.

Mechanism::

    Same as round 168, but betas_h has 4 or 5 values instead of 3.
    All layers use the SAME betas_h list (constant schedule).

Audit context (91-168): 41 strictly positive + 17 target-dep +
35 negatives = 93 mechanism classes.
"""
from __future__ import annotations

from lnn.core.layer_decay_cfc import LayerDecayCfCStackedNetwork


# ---------------------------------------------------------------------------
# Factory functions for Kh=4 / Kh=5 variants
# ---------------------------------------------------------------------------


def make_ld_constant_h4_default(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=4, constant β ∈ {0.6, 0.75, 0.85, 0.95}."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.6, 0.75, 0.85, 0.95], mode="constant",
        return_sequences=True,
    )


def make_ld_constant_h4_wide(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=4, constant β ∈ {0.5, 0.7, 0.85, 0.99}."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.5, 0.7, 0.85, 0.99], mode="constant",
        return_sequences=True,
    )


def make_ld_constant_h4_narrow(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=4, constant β ∈ {0.8, 0.85, 0.9, 0.95}."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.8, 0.85, 0.9, 0.95], mode="constant",
        return_sequences=True,
    )


def make_ld_constant_h3_k6(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=6, Kh=3, constant β ∈ {0.7, 0.85, 0.95}."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=6, betas_h=[0.7, 0.85, 0.95], mode="constant",
        return_sequences=True,
    )


def make_ld_constant_h3_wider(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=3, constant β ∈ {0.6, 0.8, 0.99}."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.6, 0.8, 0.99], mode="constant",
        return_sequences=True,
    )


def make_ld_constant_h3_finer(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=3, constant β ∈ {0.75, 0.85, 0.95}."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.75, 0.85, 0.95], mode="constant",
        return_sequences=True,
    )


def make_ld_constant_h5(input_size, hidden_size, output_size, num_layers=3):
    """3-layer, Kx=5, Kh=5, constant β ∈ {0.5, 0.7, 0.85, 0.95, 0.99}."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.5, 0.7, 0.85, 0.95, 0.99], mode="constant",
        return_sequences=True,
    )


__all__ = [
    "LayerDecayCfCStackedNetwork",
    "make_ld_constant_h4_default",
    "make_ld_constant_h4_wide",
    "make_ld_constant_h4_narrow",
    "make_ld_constant_h3_k6",
    "make_ld_constant_h3_wider",
    "make_ld_constant_h3_finer",
    "make_ld_constant_h5",
]
