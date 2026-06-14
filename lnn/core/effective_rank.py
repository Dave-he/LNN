"""Round 94 (PRD #10-56): effective rank helper.

Implements the effective rank metric from arXiv:2606.00243
(Williams/Payeur/Lajoie, ICML 2026), used to test the paper's
prediction that locality-constrained learning rules find
low-rank solutions.

The effective rank of a matrix W is::

    eff_rank(W) = (sum_i σ_i)^2 / (sum_i σ_i^2)

where σ_i are the singular values. eff_rank is a continuous,
differentiable proxy for algebraic rank: it equals 1 for rank-1
matrices and min(m, n) for full-rank matrices.

Round 94 uses this to test the hypothesis that CfC's smoothness
prior (round 91) is functionally a locality constraint, and so
its trained solutions should have lower effective rank than
MLP, LSTM, or GRU when trained on the same task.
"""
from __future__ import annotations

import torch


def effective_rank(W: torch.Tensor) -> float:
    """Compute the effective rank of a 2D matrix.

    Args:
        W: 2D tensor of shape (m, n).

    Returns:
        eff_rank(W) = (sum σ_i)^2 / (sum σ_i^2).
        Returns 0.0 if the matrix is effectively zero.
    """
    if W.dim() != 2:
        raise ValueError(f"expected 2D tensor, got {W.dim()}D")
    s = torch.linalg.svdvals(W.float())
    if s.numel() == 0:
        return 0.0
    s_sum = float(s.sum().item())
    s_sq_sum = float((s ** 2).sum().item())
    if s_sq_sum < 1e-12:
        return 0.0
    return (s_sum ** 2) / s_sq_sum


def mean_effective_rank(weights: list[torch.Tensor]) -> float:
    """Compute the mean effective rank over a list of 2D weight matrices.

    Args:
        weights: list of 2D tensors.

    Returns:
        mean of effective_rank(W) over the list. Returns 0.0 if empty.
    """
    if not weights:
        return 0.0
    return sum(effective_rank(W) for W in weights) / len(weights)


def effective_rank_trajectory(states: torch.Tensor) -> float:
    """Effective rank of a (T, d) hidden-state trajectory.

    Treats the time-major sequence as a (T, d) matrix and computes
    the effective rank. This measures the manifold dimension the
    network actually uses during inference.

    Args:
        states: 2D tensor of shape (T, d) where T is the number of
            time steps and d is the hidden dimension.

    Returns:
        eff_rank of the trajectory matrix.
    """
    if states.dim() == 1:
        states = states.unsqueeze(0)
    if states.dim() != 2:
        raise ValueError(f"expected 2D tensor, got {states.dim()}D")
    return effective_rank(states)


def rank_summary(
    weights: list[torch.Tensor],
    states: torch.Tensor | None = None,
) -> dict:
    """Summary of effective rank over weights and (optionally) a state trajectory.

    Args:
        weights: list of 2D weight tensors (mean + per-matrix values).
        states: optional (T, d) hidden-state trajectory.

    Returns:
        dict with keys:
          - 'mean_weight_eff_rank': mean across the weight matrices
          - 'per_weight_eff_rank': list of per-matrix eff_rank values
          - 'hidden_eff_rank': eff_rank of the states matrix (or None)
    """
    if not weights:
        raise ValueError("weights must be non-empty")
    per = [effective_rank(W) for W in weights]
    out: dict = {
        "mean_weight_eff_rank": sum(per) / len(per),
        "per_weight_eff_rank": per,
    }
    if states is not None:
        out["hidden_eff_rank"] = effective_rank_trajectory(states)
    else:
        out["hidden_eff_rank"] = None
    return out
