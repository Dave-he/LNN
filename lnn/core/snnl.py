"""Round 100 — Soft Nearest Neighbor Loss (PRD #10-62).

Implements SNNL from Frosst et al. 2019 (*Analyzing and Improving
Representations with the Soft Nearest Neighbor Loss*) and applies
it to MoE expert disentanglement (arXiv:2603.26734 Agarap & Azcarraga
March 2026 — *Mixture of Experts with Soft Nearest Neighbor Loss:
Resolving Expert Collapse via Representation Disentanglement*).

SNNL formula::

    L_SNNL = -1/B * Σ_i log( Σ_{j: y_i = y_j, j≠i} exp(-||f_i - f_j||²/T)
                            / Σ_{k≠i} exp(-||f_i - f_k||²/T) )

where:
- B is batch size
- T is temperature (lower = sharper, higher = softer)
- f_i = f(x_i) is the feature for example i
- The numerator is over same-class examples (excluding i)
- The denominator is over all examples (excluding i)

Intuitively: maximize the probability that same-class examples are
closer than different-class examples, in a soft k-NN sense.

For an MoE, "class" can be interpreted as the routing decision (which
expert handled the example). SNNL then encourages each expert to
handle a distinct cluster of inputs in feature space — promoting
**expert disentanglement** at the feature level (different from weight
orthogonality at the parameter level).
"""
from __future__ import annotations

import torch


def soft_nearest_neighbor_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Soft Nearest Neighbor Loss.

    Args:
        features: Tensor of shape (B, d) or (B,). Will be flattened to
            (B, d) if 1D.
        labels: Integer tensor of shape (B,) — the class/cluster of
            each example.
        temperature: Temperature parameter T. Lower T = sharper
            distribution, higher T = softer. Must be > 0.

    Returns:
        Scalar tensor — the SNNL. Returns 0 for degenerate cases
        (single-class batch, no positive pairs, B < 2).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if features.dim() == 1:
        features = features.unsqueeze(0)  # (1, B) → treated as 1 example
    B = features.shape[0]
    if B < 2:
        return torch.zeros((), device=features.device)
    # Pairwise squared distances: (B, B)
    diffs = features.unsqueeze(0) - features.unsqueeze(1)  # (B, B, d)
    sq_dist = (diffs ** 2).sum(dim=-1)  # (B, B)
    # Pairwise similarity (negative squared distance / T)
    log_prob = -sq_dist / temperature  # (B, B)
    # Mask out self
    mask = torch.eye(B, dtype=torch.bool, device=features.device)
    log_prob = log_prob.masked_fill(mask, float("-inf"))
    # Build positive-pair mask: same label, not self
    labels_row = labels.unsqueeze(0)  # (1, B)
    labels_col = labels.unsqueeze(1)  # (B, 1)
    pos_mask = (labels_row == labels_col) & ~mask  # (B, B)
    # If no positive pairs for any i, return 0
    if not pos_mask.any():
        return torch.zeros((), device=features.device)
    # SNNL: -1/B * Σ_i log( Σ_{j in pos_i} exp(log_prob_ij) / Σ_{k != i} exp(log_prob_ik) )
    # Use logsumexp for numerical stability
    # Numerator: logsumexp over positive pairs (for each i)
    log_prob_for_num = log_prob.masked_fill(~pos_mask, float("-inf"))
    log_num = torch.logsumexp(log_prob_for_num, dim=1)  # (B,)
    # Denominator: logsumexp over all non-self pairs
    log_den = torch.logsumexp(log_prob, dim=1)  # (B,)
    # Per-example loss
    per_example = log_num - log_den  # (B,)
    # Handle any -inf - -inf = nan by replacing with 0 (degenerate)
    per_example = torch.where(
        torch.isfinite(per_example), per_example, torch.zeros_like(per_example),
    )
    return -per_example.mean()


def expert_snnl_loss(
    expert_features: torch.Tensor,
    routing_decisions: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Apply SNNL to per-expert features, with routing decisions as labels.

    This is the MoE-specific wrapper. Expert features are collected into
    a (B, d) tensor where B = number of experts, d = feature dim, and
    the routing decision is the expert index for each feature.

    Args:
        expert_features: (K, d) tensor of expert features (K = n_experts).
        routing_decisions: (K,) integer tensor of routing decisions
            (typically 0..K-1, but any label works).
        temperature: SNNL temperature.

    Returns:
        Scalar tensor — the SNNL on expert features.
    """
    return soft_nearest_neighbor_loss(
        expert_features, routing_decisions, temperature=temperature,
    )


__all__ = ["soft_nearest_neighbor_loss", "expert_snnl_loss"]
