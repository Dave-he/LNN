"""State decorrelation loss — round 289 (basic) and round 290 (Barlow-Twins).

Paper grounding (/loop 2026-07-12, pivot from r284-r288 pulse line):
    arXiv:2607.01986 "Liquid Latent State Dynamics for Interpretable
    Turbofan Degradation Modeling" (Nie, Wang, Su, 2026-07) trains a
    liquid NN as latent dynamics for turbofan RUL on C-MAPSS and adds
    a **degradation/condition decorrelation loss** that beats GRU
    (RMSE 0.2266 vs 0.2438). The mechanism is to penalize cross-covariance
    between sub-processes of the latent state, forcing orthogonal
    representations.

    Round 289: implemented `state_decorrelation_loss(h, λ)` that
    penalizes `off_diag(C) / diag(C).mean()` on the hidden-state
    covariance. Result: TARGET-DEPENDENT-WITH-NUANCE (best structured
    Δ%=-32.1% in any r284-r289 variant). H3 (diag/off ratio ≥ 5) FAILED
    because the diag-normalization let the optimizer inflate the
    diagonal to make the ratio small without actually decorrelating.

    Round 290: reformulated as **Barlow-Twins-style** cross-correlation
    (Zbontar et al. 2021). Split h along feature dim into Z_A and Z_B,
    compute C = (Z_A - μ_A)(Z_B - μ_B)^T / (σ_A · σ_B · T), penalize
    off-diag(C) (decorrelate) and (diag(C) - 1) (invariance). The
    normalization is by per-feature std, so the optimizer cannot
    inflate the diagonal to escape.

API::

    from lnn.core.decorrelation_loss import (
        state_decorrelation_loss,           # r289 (basic, has H3 bug)
        barlow_twins_decorrelation_loss,    # r290 (Barlow-Twins style)
        state_covariance_diagnostics,
    )
"""

from __future__ import annotations

import torch


def state_decorrelation_loss(
    h: torch.Tensor,
    lambda_coeff: float = 0.01,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute the state decorrelation loss on a hidden-state tensor.

    Args:
        h: hidden states, shape (B, T, d_h).
        lambda_coeff: scalar multiplier on the loss. ``0`` disables.
        eps: numerical floor on the diagonal for normalization.

    Returns:
        Scalar tensor ``λ · off_diag(C) / diag(C).mean()`` — the
        normalized off-diagonal penalty on the hidden-state covariance.
    """
    if lambda_coeff == 0.0:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype)
    if h.dim() != 3:
        raise ValueError(
            f"h must be (B, T, d_h), got shape {tuple(h.shape)}")
    B, T, d_h = h.shape  # noqa: F841 (B kept for docstring clarity)
    if T < 2:
        # Single-timestep covariance is undefined; return zero loss.
        return torch.tensor(0.0, device=h.device, dtype=h.dtype)
    # Reshape to (B*d_h, T) so we estimate per-sample covariance along T.
    # Alternative: per-batch covariance (d_h, d_h) by pooling across B,T.
    # We use the per-batch-pooled covariance for stability.
    H = h.reshape(-1, d_h).T  # (d_h, B*T)
    H = H - H.mean(dim=1, keepdim=True)
    C = H @ H.T / max(H.shape[1] - 1, 1)  # (d_h, d_h)
    # Off-diagonal penalty (squared Frobenius of off-diag) divided by
    # diagonal mean for scale-invariance.
    diag = torch.diagonal(C)
    off_mask = 1.0 - torch.eye(d_h, device=C.device, dtype=C.dtype)
    off = (C * off_mask)
    off_sq_sum = (off * off).sum()
    diag_mean = diag.abs().mean() + eps
    decorr = off_sq_sum / (d_h * d_h * diag_mean * diag_mean)
    return lambda_coeff * decorr


def state_covariance_diagnostics(h: torch.Tensor) -> dict:
    """Return diagnostics on the hidden-state covariance structure.

    Useful for H3 (no-collapse / decorrelated axes): the ratio
    mean_diag / max_off_diag is large when axes are uncorrelated and
    small when they share variance.
    """
    if h.dim() != 3:
        raise ValueError(
            f"h must be (B, T, d_h), got shape {tuple(h.shape)}")
    B, T, d_h = h.shape  # noqa: F841
    if T < 2:
        return {"mean_diag": 0.0, "max_off_diag": 0.0, "ratio": 0.0}
    H = h.reshape(-1, d_h).T
    H = H - H.mean(dim=1, keepdim=True)
    C = H @ H.T / max(H.shape[1] - 1, 1)
    diag = torch.diagonal(C).abs()
    off_mask = 1.0 - torch.eye(d_h, device=C.device, dtype=C.dtype)
    off = (C.abs() * off_mask)
    return {
        "mean_diag": float(diag.mean().item()),
        "max_off_diag": float(off.max().item()),
        "ratio": float((diag.mean() / (off.max() + 1e-9)).item()),
    }


def barlow_twins_decorrelation_loss(
    h: torch.Tensor,
    lambda_off: float = 0.005,
    lambda_on: float = 0.005,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Barlow-Twins-style decorrelation loss on the hidden state.

    Splits the hidden state along the feature dimension into two
    halves Z_A and Z_B, computes the cross-correlation matrix, and
    penalizes:
      - off-diagonal elements (decorrelate features from each other)
      - deviation of diagonal from 1 (each feature should be invariant
        to the split)

    The normalization is by per-feature std, so the optimizer cannot
    inflate the diagonal to escape (which was the r289 H3 bug).

    Args:
        h: hidden states, shape (B, T, d_h). d_h must be even.
        lambda_off: weight on the off-diagonal penalty.
        lambda_on: weight on the diagonal-to-1 penalty.
        eps: numerical floor for std.

    Returns:
        Scalar tensor ``λ_off · Σoff² + λ_on · Σ(diag-1)²`` over the
        cross-correlation matrix.

    Reference: Zbontar et al. (2021), "Barlow Twins: Self-Supervised
    Learning via Redundancy Reduction".
    """
    if lambda_off == 0.0 and lambda_on == 0.0:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype)
    if h.dim() != 3:
        raise ValueError(
            f"h must be (B, T, d_h), got shape {tuple(h.shape)}")
    B, T, d_h = h.shape
    if d_h % 2 != 0:
        raise ValueError(
            f"d_h must be even for Barlow-Twins split, got {d_h}")
    if T < 2:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype)
    # Reshape to (B*T, d_h), split along feature dim.
    H = h.reshape(-1, d_h)
    Z_A = H[:, :d_h // 2]
    Z_B = H[:, d_h // 2:]
    # Per-feature normalization (zero mean, unit std).
    Z_A_n = (Z_A - Z_A.mean(dim=0, keepdim=True)) / (
        Z_A.std(dim=0, keepdim=True) + eps)
    Z_B_n = (Z_B - Z_B.mean(dim=0, keepdim=True)) / (
        Z_B.std(dim=0, keepdim=True) + eps)
    # Cross-correlation matrix C[i,j] = E[Z_A_n[:,i] · Z_B_n[:,j]].
    d = d_h // 2
    C = (Z_A_n.T @ Z_B_n) / max(Z_A_n.shape[0], 1)  # (d, d)
    eye = torch.eye(d, device=C.device, dtype=C.dtype)
    off = C * (1.0 - eye)
    on = C - eye
    loss = lambda_off * (off * off).sum() + lambda_on * (on * on).sum()
    return loss


def barlow_twins_covariance_diagnostics(h: torch.Tensor) -> dict:
    """Return diagnostics on the Barlow-Twins cross-correlation matrix."""
    if h.dim() != 3:
        raise ValueError(
            f"h must be (B, T, d_h), got shape {tuple(h.shape)}")
    B, T, d_h = h.shape  # noqa: F841
    if d_h % 2 != 0:
        return {"bt_diag": float("nan"), "bt_off": float("nan"),
                "bt_ratio": float("nan")}
    if T < 2:
        return {"bt_diag": 0.0, "bt_off": 0.0, "bt_ratio": 0.0}
    H = h.reshape(-1, d_h)
    Z_A = H[:, :d_h // 2]
    Z_B = H[:, d_h // 2:]
    Z_A_n = (Z_A - Z_A.mean(dim=0, keepdim=True)) / (
        Z_A.std(dim=0, keepdim=True) + 1e-6)
    Z_B_n = (Z_B - Z_B.mean(dim=0, keepdim=True)) / (
        Z_B.std(dim=0, keepdim=True) + 1e-6)
    d = d_h // 2
    C = (Z_A_n.T @ Z_B_n) / max(Z_A_n.shape[0], 1)
    eye = torch.eye(d, device=C.device, dtype=C.dtype)
    diag = torch.diagonal(C).abs()
    off = (C.abs() * (1.0 - eye))
    return {
        "bt_diag": float(diag.mean().item()),
        "bt_off": float(off.max().item()),
        "bt_ratio": float((diag.mean() / (off.max() + 1e-9)).item()),
    }


__all__ = [
    "state_decorrelation_loss",
    "barlow_twins_decorrelation_loss",
    "barlow_twins_covariance_diagnostics",
    "state_covariance_diagnostics",
]