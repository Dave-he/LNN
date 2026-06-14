"""Round 99 — Segment Reliability Gate (PRD #10-61).

Implements the uncertainty-aware reliability gate from arXiv:2606.03631
(Xie et al., KDD 2026) — *AnchorMoE: Interpretable Time Series
Classification via Anchor-Routed MoE*.

The reliability gate is a **per-input** mechanism (different from the
per-expert gates in rounds 84-86, 89). It computes a reliability score
``r ∈ [0, 1]`` based on local input statistics, and uses it to dampen
the model's contribution on noisy / low-confidence inputs.

The intuition: an input segment with high local noise is less reliable
than a smooth segment, so its prediction should be weighted less in
the final output. This is the **input-side** analog of our existing
expert-side gates (EcologyGatedBalancer, CausalityGatedOrth).

Mechanism::

    r = 1 / (1 + sigma_local / sigma_min)
    y_gated = (1 - mix) * y_pred + mix * r * y_pred

where ``sigma_local = std(x_segment)`` and ``sigma_min`` is the threshold
above which an input is considered noisy.
"""
from __future__ import annotations

import torch


def segment_reliability(x: torch.Tensor, sigma_min: float = 0.01) -> torch.Tensor:
    """Per-input reliability score in [0, 1].

    Higher ``sigma_min`` → more permissive (more inputs are reliable).
    Lower ``sigma_min`` → more strict (only very smooth inputs are reliable).

    Args:
        x: Input tensor of any shape. The local standard deviation is
            computed over all elements.
        sigma_min: Threshold above which an input is considered noisy.
            Default 0.01 is calibrated for inputs normalized to [0, 1].

    Returns:
        Scalar tensor in [0, 1] — the reliability score.
    """
    if sigma_min <= 0:
        raise ValueError(f"sigma_min must be > 0, got {sigma_min}")
    sigma_local = x.std()
    r = 1.0 / (1.0 + sigma_local / sigma_min)
    # Clamp for numerical safety
    return torch.clamp(r, min=0.0, max=1.0)


def apply_reliability_gate(
    y_pred: torch.Tensor,
    x: torch.Tensor,
    sigma_min: float = 0.01,
    mix: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dampen ``y_pred`` by per-input reliability.

    Args:
        y_pred: Model output, any shape.
        x: Input used to compute the reliability. Same shape as the
            first arg of the model (typically (T,) or (B, T, 1)).
        sigma_min: Threshold for noise. Default 0.01.
        mix: Interpolation in [0, 1]. 0 = no gating, 1 = full gating.
            0.5 = blend half-gated and half-original.

    Returns:
        (y_gated, reliability) — y_gated is the gated output (same shape
        as y_pred), reliability is a scalar tensor in [0, 1].
    """
    if not (0.0 <= mix <= 1.0):
        raise ValueError(f"mix must be in [0, 1], got {mix}")
    r = segment_reliability(x, sigma_min=sigma_min)
    y_gated = (1.0 - mix) * y_pred + mix * r * y_pred
    return y_gated, r


__all__ = ["segment_reliability", "apply_reliability_gate"]
