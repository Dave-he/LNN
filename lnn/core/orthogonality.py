"""Geometric orthogonality constraint for MoE expert representations (PRD #10-37, 2026-06-14).

Implements the minimum-viable instantiation of the
"geometric orthogonality constraint that penalizes representational
redundancy" idea from arXiv:2606.03631 (AnchorMoE, 2026-06-02):

    L_orth = λ * Σ_{i<j} cos_sim(h_i, h_j)^2

where ``h_i`` is the hidden-state output of expert ``i`` on the same
input batch.  The penalty is *zero* when all expert outputs are
mutually orthogonal in the cosine-similarity sense, and *positive*
when any two experts' representations align.

This is a **soft** regulariser.  Unlike φ-Balancing
(arXiv:2605.15403) which explicitly redistributes routing mass,
orthogonality only nudges the expert parameters so that their
*learned functions* end up diverse.  In combination with the
Causal Audit warning (arXiv:2606.10703 — observational metrics do
not necessarily reflect causal importance), the orthogonality
constraint is a *defensive* measure: even if the observational
routing distribution collapses, the underlying expert projections
remain at least decorrelated.

Usage::

    outs = [expert(x_t, h) for expert in experts]   # K × [B, H]
    aux  = orthogonality_loss(outs, lambda_coeff=0.01)
    loss = task_loss(y_pred, y) + aux
"""
from __future__ import annotations

import torch


def orthogonality_loss(
    expert_outputs: list[torch.Tensor],
    lambda_coeff: float = 0.01,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Geometric orthogonality penalty over K expert outputs.

    Args:
        expert_outputs: list of K tensors, each of shape ``[B, H]``
            (or any shape — the last dim is treated as the
            "representation" axis and the leading dims are flattened).
        lambda_coeff: scaling factor λ applied to the penalty.  A
            value of 0.0 disables the penalty entirely (returns 0
            without reading the inputs).
        eps: numerical stability constant for cosine-similarity
            normalisation (avoid division by zero when an expert
            output is exactly all-zero).

    Returns:
        aux_loss: scalar tensor ``λ * Σ_{i<j} cos_sim(h_i, h_j)^2``.
        Returned as a 0-d tensor so it composes with the main task
        loss via simple addition.
    """
    if lambda_coeff == 0.0:
        # Back-compat / fast path.
        return torch.zeros((), device=expert_outputs[0].device if expert_outputs else "cpu")
    if len(expert_outputs) < 2:
        # One or zero experts → no pairwise penalty possible.
        return torch.zeros((), device=expert_outputs[0].device if expert_outputs else "cpu")

    # Stack and flatten all leading dims so the last dim is the
    # representation axis.  We then L2-normalise along that axis
    # and compute the Gram matrix of cosine similarities.
    vecs = torch.stack([v.reshape(-1, v.shape[-1]) for v in expert_outputs], dim=0)  # [K, B*H, H]
    norms = vecs.norm(dim=-1, keepdim=True).clamp_min(eps)  # [K, B*H, 1]
    normed = vecs / norms  # [K, B*H, H]
    # Mean representation per expert (averaged over the batch):
    #   [K, H]
    mean_vec = normed.mean(dim=1)
    mean_norm = mean_vec.norm(dim=-1, keepdim=True).clamp_min(eps)  # [K, 1]
    mean_unit = mean_vec / mean_norm
    # Cosine-similarity Gram matrix: [K, K]
    gram = mean_unit @ mean_unit.t()
    # Take upper triangle (i<j), square, sum.
    K = gram.shape[0]
    iu = torch.triu_indices(K, K, offset=1, device=gram.device)  # [2, K*(K-1)/2]
    pairs = gram[iu[0], iu[1]]  # [K*(K-1)/2]
    penalty = (pairs ** 2).sum()
    return lambda_coeff * penalty
