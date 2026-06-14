"""Round 91 (PRD #10-53): temporal smoothness metrics.

Used to test the claim from arXiv:2606.07670 (Li, Pal, Tan, June 2026)
that CfC cells produce smoother outputs in t than equivalent MLPs
because of the closed-form time-constant inductive bias.

Three metrics, all based on finite differences of a 1D signal:
- :func:`total_variation` — mean |y[i+1] - y[i]|. Lower = smoother.
- :func:`l2_derivative` — RMS finite-diff derivative. Lower = smoother.
- :func:`max_gradient` — max |f'(t)|. Lower = smoother.

All metrics are scale-invariant to the absolute magnitude of y — they
only care about how rapidly y changes. For unnormalised data, divide
the y values by their range before computing.
"""
from __future__ import annotations

import torch


def total_variation(y: torch.Tensor) -> float:
    """Mean absolute first difference of a 1D signal.

    Args:
        y: 1D tensor (n,).

    Returns:
        TV = mean(|y[i+1] - y[i]|). Lower = smoother.
    """
    if y.numel() < 2:
        return 0.0
    return float((y[1:] - y[:-1]).abs().mean().item())


def l2_derivative(y: torch.Tensor, dt: float = 1.0) -> float:
    """RMS finite-difference derivative.

    Args:
        y: 1D tensor (n,).
        dt: spacing between consecutive samples. Default 1.0.

    Returns:
        sqrt(mean((y[i+1] - y[i])^2)) / dt. Lower = smoother.
    """
    if y.numel() < 2:
        return 0.0
    d = (y[1:] - y[:-1]) / dt
    return float((d ** 2).mean().sqrt().item())


def max_gradient(y: torch.Tensor, dt: float = 1.0) -> float:
    """Max absolute finite-difference derivative.

    Args:
        y: 1D tensor (n,).
        dt: spacing between consecutive samples. Default 1.0.

    Returns:
        max(|y[i+1] - y[i]|) / dt. Lower = smoother.
    """
    if y.numel() < 2:
        return 0.0
    d = (y[1:] - y[:-1]) / dt
    return float(d.abs().max().item())


def smoothness_summary(y: torch.Tensor, dt: float = 1.0) -> dict:
    """All three smoothness metrics in one call.

    Returns:
        dict with keys: tv, l2_deriv, max_grad, n.
    """
    return {
        "tv": total_variation(y),
        "l2_deriv": l2_derivative(y, dt=dt),
        "max_grad": max_gradient(y, dt=dt),
        "n": int(y.numel()),
    }
