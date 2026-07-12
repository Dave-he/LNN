"""State decorrelation loss — round 289.

Paper grounding (/loop 2026-07-12, pivot from r284-r288 pulse line):
    arXiv:2607.01986 "Liquid Latent State Dynamics for Interpretable
    Turbofan Degradation Modeling" (Nie, Wang, Su, 2026-07) trains a
    liquid NN as latent dynamics for turbofan RUL on C-MAPSS and adds
    a **degradation/condition decorrelation loss** that beats GRU
    (RMSE 0.2266 vs 0.2438). The mechanism is to penalize cross-covariance
    between sub-processes of the latent state, forcing orthogonal
    representations.

    This round implements a simplified, generic version: a **state
    decorrelation loss** that penalizes the off-diagonal of the hidden
    state's covariance matrix. The loss is plug-and-play for any cell
    that exposes a hidden-state tensor — usable with blend_gated, CfC,
    MoE cells, etc.

Mechanism::

    h ∈ R^{B × T × d_h}    # hidden states from a forward pass
    H = h.permute(0,2,1).reshape(B*d_h, T)    # or reshape(B, d_h, T)
    C = H @ H.T / T        # (B*d_h, B*d_h) covariance estimate
    L = mean(off_diag(C)) / mean(diag(C))    # normalized

    - ``lambda_coeff=0`` ⇒ loss is exactly 0 ⇒ no effect.
    - The loss is *unsupervised*: it depends only on the hidden state,
      not on labels, so it cannot directly conflict with task loss.

Hypotheses (PRD #10-130):

    H1 (target-independent): improves or maintains task loss on ALL 3
       datasets (toy_sin / structured / random) at λ ∈ {0.001, 0.01}.
    H2 (orthogonality): combines with blend gate without interference.
    H3 (no collapse): learned state covariance has
       mean_diag / max_off_diag ≥ 5 (decorrelated axes).
    H4 (strict-positive default): if H1 passes for any λ, the loss is
       +1 SP — first non-pulse SP in this 5-round pulse line.
    H5 (gradients flow): loss is differentiable end-to-end.

API::

    from lnn.core.decorrelation_loss import state_decorrelation_loss
    out, h = cell(x)                 # h: (B, T, d_h)
    loss_dec = state_decorrelation_loss(h, lambda_coeff=0.01)
    total_loss = task_loss + loss_dec
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


__all__ = ["state_decorrelation_loss", "state_covariance_diagnostics"]