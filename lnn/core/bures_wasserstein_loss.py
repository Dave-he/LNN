"""Bures-Wasserstein Loss for Time Series Forecasting (PRD #10-151, Round 189, 2026-06-16).

Inspired by **DistDF: Time-Series Forecasting Needs Joint-Distribution
Wasserstein Alignment** (Wang et al., ICLR 2026, arXiv:2510.24574).

Core idea: instead of pointwise MSE which assumes independence across
time, align the JOINT distribution of [X, Y] (history + labels) with
[X, Ŷ] (history + predictions) using Bures-Wasserstein between
Gaussian fits.

Mechanism::

    # Step 1: concatenate history and target along time axis
    Z  = [X, Y]     # [B, 2T, D_total]
    Ẑ  = [X, Ŷ]    # [B, 2T, D_total]

    # Step 2: estimate per-batch Gaussian
    μ_Z  = Z.mean(dim=0)         # [2T * D_total]
    Σ_Z  = cov(Z, dim=0)         # [2T * D_total, 2T * D_total]
    same for Ẑ

    # Step 3: Bures-Wasserstein closed form
    BW² = ||μ_Z - μ_Ẑ||²
        + Tr(Σ_Z + Σ_Ẑ - 2·√(Σ_Z^(1/2) · Σ_Ẑ · Σ_Z^(1/2)))

    # Step 4: combined loss
    ℒ = γ · ℒ_BW + (1 - γ) · ℒ_MSE

Why Bures-Wasserstein (not Sinkhorn):
- Closed-form (no iterations)
- Differentiable via matrix square root
- Captures both mean shift and covariance change
- Provably upper-bounds conditional distribution discrepancy
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _matrix_sqrt(A):
    """Differentiable symmetric positive semi-definite matrix square root.

    Uses eigendecomposition. Eigenvalues clamped to >= 0 for numerical
    stability (a small amount of regularization is added implicitly
    because the gradient flows through the clamped part).

    Args:
        A: [..., D, D] symmetric PSD matrix
    Returns:
        A^(1/2): [..., D, D]
    """
    # Eigh for symmetric matrices
    eigvals, eigvecs = torch.linalg.eigh(A)
    # Clamp to non-negative (PSD correction)
    eigvals = torch.clamp(eigvals, min=0.0)
    # Reconstruct sqrt
    sqrt_eigvals = torch.sqrt(eigvals + 1e-12)
    # A^(1/2) = V @ diag(sqrt(λ)) @ V.T
    return eigvecs @ torch.diag_embed(sqrt_eigvals) @ eigvecs.transpose(-1, -2)


def bures_wasserstein2(mu1, cov1, mu2, cov2):
    """Squared Bures-Wasserstein distance between two Gaussians.

    BW²(μ₁, Σ₁; μ₂, Σ₂) = ||μ₁-μ₂||² + Tr(Σ₁+Σ₂-2·√(Σ₁^(1/2)·Σ₂·Σ₁^(1/2)))

    Args:
        mu1: [D] mean of first Gaussian
        cov1: [D, D] covariance of first Gaussian
        mu2: [D] mean of second Gaussian
        cov2: [D, D] covariance of second Gaussian
    Returns:
        scalar (BW², non-negative)
    """
    # Mean term
    mean_diff = ((mu1 - mu2) ** 2).sum()
    # Covariance term
    sqrt_cov1 = _matrix_sqrt(cov1)  # [D, D]
    inner = sqrt_cov1 @ cov2 @ sqrt_cov1  # [D, D]
    sqrt_inner = _matrix_sqrt(inner)  # [D, D]
    # Add small eps for numerical stability of trace
    bures_trace = torch.trace(cov1) + torch.trace(cov2) - 2.0 * torch.trace(sqrt_inner)
    bures_trace = torch.clamp(bures_trace, min=0.0)
    return mean_diff + bures_trace


def joint_bures_wasserstein(target, prediction):
    """Compute Bures-Wasserstein between the joint distribution of
    target sequences and prediction sequences.

    The DistDF paper concatenates history and labels [X, Y] vs [X, Ŷ].
    For our setting (where history has different last-dim than target),
    we use the prediction sequence alone as the joint distribution over
    B of T-length sequences. This still captures:
    - Mean shift (per-feature, per-time)
    - Covariance structure across batch

    Args:
        target: [B, T, D] true labels
        prediction: [B, T, D] model predictions (must match target shape)
    Returns:
        scalar BW² loss
    """
    B = target.shape[0]
    Z_flat = target.reshape(B, -1)       # [B, T*D]
    Zh_flat = prediction.reshape(B, -1)  # [B, T*D]

    # Mean and covariance (over batch dim)
    mu_Z = Z_flat.mean(dim=0)        # [T*D]
    mu_Zh = Zh_flat.mean(dim=0)      # [T*D]
    # Centered
    Z_centered = Z_flat - mu_Z       # [B, T*D]
    Zh_centered = Zh_flat - mu_Zh    # [B, T*D]
    cov_Z = (Z_centered.T @ Z_centered) / (B - 1)      # [T*D, T*D]
    cov_Zh = (Zh_centered.T @ Zh_centered) / (B - 1)   # [T*D, T*D]

    return bures_wasserstein2(mu_Z, cov_Z, mu_Zh, cov_Zh)


def combined_distdf_loss(target, prediction, gamma=0.5):
    """DistDF combined loss: γ · ℒ_BW + (1-γ) · ℒ_MSE.

    Args:
        target: [B, T, D] true labels
        prediction: [B, T, D] model predictions
        gamma: weight on distributional loss (0 = pure MSE, 1 = pure BW)
    Returns:
        scalar combined loss
    """
    mse = F.mse_loss(prediction, target)
    bw = joint_bures_wasserstein(target, prediction)
    return gamma * bw + (1.0 - gamma) * mse


__all__ = [
    "_matrix_sqrt",
    "bures_wasserstein2",
    "joint_bures_wasserstein",
    "combined_distdf_loss",
]
