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
    # Avoid the unconditional `.float()` copy when W is already float32 —
    # saves one full-sized allocation per call on the hot diagnostic path.
    s = torch.linalg.svdvals(W) if W.dtype == torch.float32 else torch.linalg.svdvals(W.float())
    if s.numel() == 0:
        return 0.0
    # Single GPU sync via stacked reductions; `s.dot(s)` avoids the
    # intermediate `s ** 2` tensor that `(s ** 2).sum()` would allocate.
    s_sum, s_sq_sum = (s.sum(), s.dot(s))
    if s_sq_sum.item() < 1e-12:
        return 0.0
    return (s_sum.item() ** 2) / s_sq_sum.item()


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


def per_expert_effective_rank(cell) -> list[float]:
    """Compute the mean effective rank of each expert's 2D weight matrices.

    Iterates over ``cell.experts[i]`` (assumed to be an ``nn.ModuleList``)
    and, for each expert, collects every 2D weight parameter, calls
    ``effective_rank`` on it, and returns the **mean** across matrices
    for that expert.

    This is the round 95 (PRD #10-57) diagnostic for FAME/MR-MoE —
    direct measurement of whether the experts have **distinct** weight
    signatures after training (the "diverse experts" claim of the FAME
    paper, arXiv:2606.08896).

    Args:
        cell: An ``nn.Module`` with an ``experts: nn.ModuleList``
            attribute (FAMECfCCell, MRMoECfCCell, etc.).

    Returns:
        List of K floats, one per expert (K = len(cell.experts)).
        Each value is the mean eff_rank across that expert's 2D
        weight matrices.  Returns ``[]`` if the cell has no experts
        or no 2D weights.
    """
    if not hasattr(cell, "experts"):
        return []
    experts = cell.experts
    out: list[float] = []
    for expert in experts:
        per_weight: list[float] = []
        for _, p in expert.named_parameters():
            if p.dim() == 2:
                per_weight.append(effective_rank(p.detach()))
        if per_weight:
            out.append(sum(per_weight) / len(per_weight))
        else:
            out.append(0.0)
    return out


def expert_diversity_ratio(per_expert_ranks: list[float]) -> float:
    """Max/min ratio of per-expert effective ranks.

    A simple diversity measure: 1.0 means all experts have the
    same eff_rank (uniform / collapsed), > 1.5 means the experts
    are clearly distinct.

    Args:
        per_expert_ranks: list of K floats, one per expert.

    Returns:
        max(ranks) / min(ranks), or 0.0 if all ranks are zero.
    """
    if not per_expert_ranks:
        return 0.0
    mn = min(per_expert_ranks)
    mx = max(per_expert_ranks)
    if mn < 1e-12:
        # All-zero or near-zero → degenerate; return inf so callers
        # can flag it but never silently see 0/0 = nan.
        return float("inf") if mx > 1e-12 else 0.0
    return mx / mn


def expert_diversity_summary(cell) -> dict:
    """Combined per-expert effective rank diagnostic (round 95, PRD #10-57).

    Returns a dict with:
      - 'per_expert': list of K eff_rank values (one per expert)
      - 'mean': mean across experts
      - 'min': minimum across experts
      - 'max': maximum across experts
      - 'std': std across experts
      - 'diversity_ratio': max/min ratio
      - 'n_experts': K
      - 'n_dead': number of experts with eff_rank < 0.5 (collapsed)

    Args:
        cell: An ``nn.Module`` with ``experts: nn.ModuleList`` attribute.

    Returns:
        Dict as above. Empty cell → dict with zeros and n_experts=0.
    """
    per = per_expert_effective_rank(cell)
    if not per:
        return {
            "per_expert": [],
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "diversity_ratio": 0.0,
            "n_experts": 0,
            "n_dead": 0,
        }
    n_dead = sum(1 for r in per if r < 0.5)
    return {
        "per_expert": per,
        "mean": sum(per) / len(per),
        "min": min(per),
        "max": max(per),
        "std": _std(per),
        "diversity_ratio": expert_diversity_ratio(per),
        "n_experts": len(per),
        "n_dead": n_dead,
    }


def _std(xs: list[float]) -> float:
    """Population std (no Bessel correction) — tiny helper, avoids numpy import."""
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
