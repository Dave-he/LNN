"""Energy Distance Loss for Time Series Forecasting (PRD #10-153, Round 191, 2026-06-16).

Energy Distance (Székely & Rizzo 2004, 2013) is a metric on
probability distributions:
```
D²(F, G) = 2 E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]
```
where X, X' are iid from F and Y, Y' are iid from G.

Why Energy Distance (vs SWD/BW):
- NO sorting (unlike SWD 1D W2)
- NO random projections (unlike SWD)
- NO matrix sqrt (unlike BW)
- Just pairwise Euclidean distance expectations
- Differentiable via standard autograd
- O(B²) compute per pair (small B is fine)

Empirical estimator (over batch B):
```
D² = (2/B²) · Σ_{i,j} ||x_i - y_j||
     - (1/B²) · Σ_{i,j} ||x_i - x_j||
     - (1/B²) · Σ_{i,j} ||y_i - y_j||
```

D² = 0 iff F = G, and D² > 0 otherwise.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def energy_distance2(target, prediction):
    """Squared Energy Distance between target and prediction distributions.

    Args:
        target: [B, T, D] true labels
        prediction: [B, T, D] model predictions
    Returns:
        scalar D² (non-negative)
    """
    B = target.shape[0]
    # Flatten to [B, T*D]
    target_flat = target.reshape(B, -1)
    pred_flat = prediction.reshape(B, -1)

    # Pairwise distances
    # ||x - y||  : [B, B] cross distances
    cross_dist = torch.cdist(target_flat, pred_flat, p=2).mean()
    # ||x - x'||: [B, B] within target
    within_t = torch.cdist(target_flat, target_flat, p=2).mean()
    # ||y - y'||: [B, B] within prediction
    within_p = torch.cdist(pred_flat, pred_flat, p=2).mean()

    return 2.0 * cross_dist - within_t - within_p


def combined_energy_loss(target, prediction, gamma=0.1):
    """Energy-distance-augmented MSE: γ · ℒ_ED + (1-γ) · ℒ_MSE.

    Lower γ than SWD/BW because ED is unbiased and gradient-friendly.

    Args:
        target: [B, T, D]
        prediction: [B, T, D]
        gamma: weight on ED loss
    Returns:
        scalar combined loss
    """
    mse = F.mse_loss(prediction, target)
    ed = energy_distance2(target, prediction)
    return gamma * ed + (1.0 - gamma) * mse


__all__ = [
    "energy_distance2",
    "combined_energy_loss",
]
