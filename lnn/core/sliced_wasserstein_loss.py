"""Sliced Wasserstein Loss for Time Series Forecasting (PRD #10-152, Round 190, 2026-06-16).

Inspired by **DistDF** (Wang et al., ICLR 2026, arXiv:2510.24574)
which used **Bures-Wasserstein** between Gaussian fits. Round 189
showed BW regresses in 1D toy regime due to gradient dominance at
γ=0.5 and noisy cov estimation at B=32.

This module replaces BW with **Sliced Wasserstein Distance (SWD)**:
- Project target/prediction onto random 1D directions
- Compute 1D W2 = L2 between sorted values (closed form)
- Average over projections → unbiased estimator of full W2

Why SWD > BW:
- NO matrix sqrt (cheaper, more stable)
- Scales naturally to high dimensions
- 1D W2 = L2 of sorted values (closed-form, differentiable via sort)
- Random projections = Monte Carlo estimate of full W2
- Effective dims at projection: 1, not 32 → no gradient dominance

Reference: Rabin et al. (2012) "Optimal Transport with
Proximal Splitting"; GOTO-SWAP (Flatiron 2026).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _wasserstein1d_squared(x, y):
    """Squared 1D Wasserstein-2 distance between samples.

    W2²(x, y) = ∫ (F_x^{-1}(p) - F_y^{-1}(p))² dp
             = E[(sorted_x_i - sorted_y_i)²]  (empirical)

    Args:
        x: [N, D] samples (one distribution)
        y: [M, D] samples (other distribution)
    Returns:
        scalar W2² (per-feature average)
    """
    # Sort each column independently
    x_sorted, _ = torch.sort(x, dim=0)  # [N, D]
    y_sorted, _ = torch.sort(y, dim=0)  # [M, D]
    # Interpolate to common length
    N = min(x_sorted.shape[0], y_sorted.shape[0])
    # Truncate to common length
    x_trunc = x_sorted[:N]
    y_trunc = y_sorted[:N]
    # Per-feature squared diff, then mean
    return ((x_trunc - y_trunc) ** 2).mean()


def sliced_wasserstein2(target, prediction, n_projections=50, seed=0):
    """Sliced Wasserstein Distance via random projections.

    For each random 1D projection θ_i:
        proj_target_i = target @ θ_i
        proj_pred_i = prediction @ θ_i
        w2_i = W2²(proj_target_i, proj_pred_i)
    SWD = (1/K) · Σ_i w2_i

    Args:
        target: [B, T, D] true labels
        prediction: [B, T, D] model predictions
        n_projections: number of random 1D directions
        seed: random seed for reproducibility
    Returns:
        scalar SWD
    """
    B = target.shape[0]
    T = target.shape[1]
    D = target.shape[2]

    # Flatten to [B, T*D]
    target_flat = target.reshape(B, -1)
    pred_flat = prediction.reshape(B, -1)
    N = T * D  # feature dim

    # Generate random directions: [n_projections, N]
    gen = torch.Generator()
    gen.manual_seed(seed)
    # Random from N(0, 1), normalize to unit vectors
    theta = torch.randn(n_projections, N, generator=gen)
    theta = theta / (theta.norm(dim=1, keepdim=True) + 1e-8)

    # Project: [B, n_projections]
    proj_target = target_flat @ theta.T
    proj_pred = pred_flat @ theta.T

    # Compute 1D W2 per projection
    total = 0.0
    for i in range(n_projections):
        x = proj_target[:, i:i+1]  # [B, 1]
        y = proj_pred[:, i:i+1]    # [B, 1]
        total = total + _wasserstein1d_squared(x, y)
    return total / n_projections


def combined_swd_loss(target, prediction, gamma=0.1, n_projections=50, seed=0):
    """SWD-augmented MSE loss: γ · ℒ_SWD + (1-γ) · ℒ_MSE.

    Lower γ than BW (round 189) because SWD is already a lower-
    dimensional summary (avg of 1D projections vs full cov).

    Args:
        target: [B, T, D]
        prediction: [B, T, D]
        gamma: weight on SWD loss
        n_projections: number of random projections
        seed: random seed
    Returns:
        scalar combined loss
    """
    mse = F.mse_loss(prediction, target)
    swd = sliced_wasserstein2(target, prediction, n_projections=n_projections, seed=seed)
    return gamma * swd + (1.0 - gamma) * mse


__all__ = [
    "_wasserstein1d_squared",
    "sliced_wasserstein2",
    "combined_swd_loss",
]
