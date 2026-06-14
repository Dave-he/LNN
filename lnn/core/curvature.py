"""Round 101 — Ollivier-Ricci Curvature routing signal (PRD #10-63).

Implements the Ollivier-Ricci Curvature (ORC) of a k-NN graph built from
a set of feature points, as a routing regularizer for MoE expert
disentanglement. Response to arXiv:2603.22317 (Cao et al., March 2026)
— *Geometric Mixture-of-Experts with Curvature-Guided Adaptive Routing*
(GeoMoE).

ORC formula for an edge (i, j) in a k-NN graph::

    ORC(i, j) = 1 - W_1(mu_i, mu_j) / d(x_i, x_j)

where:
- mu_i = uniform distribution over {i} ∪ N_k(i) (k-nearest-neighbors of i)
- mu_j = uniform distribution over {j} ∪ N_k(j)
- W_1 = Wasserstein-1 distance (earth mover's distance)
- d(x_i, x_j) = Euclidean distance between points i and j

Interpretation:
- ORC ≈ 1  : neighborhoods are far apart relative to edge length
  (local tree-like structure → experts in different regimes)
- ORC ≈ 0  : neighborhoods overlap proportionally to edge length
  (local "flat" structure → experts in similar regions)
- ORC < 0  : neighborhoods overlap MORE than edge length
  (local clustered structure → experts redundant)

For MoE routing, **high mean ORC** between experts means the expert
manifold is "tree-like" (good for diversity), while **low/negative
mean ORC** means experts overlap in feature space (bad for diversity).

The Wasserstein-1 distance is computed via the Sinkhorn-Knopp
entropic-regularized approximation, which is differentiable and
standard for small-scale optimal transport.

Functions:
- ``ollivier_ricci_curvature(points, k=2, sinkhorn_iters=10)``
  → (N, N) symmetric matrix of ORC values, zero on diagonal.
- ``mean_ollivier_ricci(points, k=2, sinkhorn_iters=10)``
  → scalar mean ORC over all edges.
- ``curvature_routing_loss(expert_features, k=2, lambda_coeff=0.001)``
  → ``λ * mean(1 - ORC)`` penalty that encourages tree-like manifold.
"""
from __future__ import annotations

import torch


def _pairwise_distances(points: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances.

    Args:
        points: Tensor of shape (N, d).

    Returns:
        Tensor of shape (N, N) where entry (i, j) is ||points[i] - points[j]||_2.
    """
    diffs = points.unsqueeze(0) - points.unsqueeze(1)  # (N, N, d)
    return torch.sqrt((diffs ** 2).sum(dim=-1) + 1e-12)  # (N, N)


def _knn_indices(distances: torch.Tensor, k: int) -> torch.Tensor:
    """Find k-nearest neighbors for each point (excluding self).

    Args:
        distances: (N, N) pairwise distance matrix (diagonal is 0).
        k: number of neighbors.

    Returns:
        Long tensor of shape (N, k) with indices of k-nearest neighbors
        for each point.
    """
    N = distances.shape[0]
    k = min(k, N - 1)
    # Set diagonal to +inf so we don't pick self
    inf_diag = distances + torch.eye(N, device=distances.device) * 1e10
    # argpartition would be faster, but for small N (K experts) full sort is fine
    _, nn_idx = inf_diag.topk(k, dim=1, largest=False)  # (N, k)
    return nn_idx


def _sinkhorn_transport(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    reg: float = 0.1,
    n_iters: int = 10,
) -> torch.Tensor:
    """Sinkhorn-Knopp algorithm for entropic-regularized optimal transport.

    Computes the entropic-regularized transport plan T* solving::

        min_T <C, T> + reg * H(T)
        s.t. T 1 = a, T^T 1 = b, T >= 0

    Args:
        cost: (m, n) cost matrix C.
        a: (m,) source marginal (sums to 1).
        b: (n,) target marginal (sums to 1).
        reg: entropic regularization (smaller = closer to exact OT).
        n_iters: number of Sinkhorn iterations.

    Returns:
        (m, n) transport plan T*. W_1 = <C, T*>.
    """
    K = torch.exp(-cost / reg)  # (m, n)
    u = torch.ones_like(a) / a.shape[0]  # (m,)
    v = torch.ones_like(b) / b.shape[0]  # (n,)
    for _ in range(n_iters):
        v = b / (K.t() @ u + 1e-12)
        u = a / (K @ v + 1e-12)
    T = u.unsqueeze(1) * K * v.unsqueeze(0)  # (m, n)
    return T


def ollivier_ricci_curvature(
    points: torch.Tensor,
    k: int = 2,
    sinkhorn_iters: int = 10,
    sinkhorn_reg: float = 0.1,
) -> torch.Tensor:
    """Compute Ollivier-Ricci Curvature of the k-NN graph of points.

    For each edge (i, j) in the k-NN graph::

        ORC(i, j) = 1 - W_1(mu_i, mu_j) / d(x_i, x_j)

    where mu_i = uniform over {i} ∪ N_k(i), mu_j = uniform over
    {j} ∪ N_k(j), and W_1 is the Wasserstein-1 distance.

    Args:
        points: Tensor of shape (N, d). N = number of points (e.g. experts).
        k: number of nearest neighbors (excluding self). Must be in [1, N-1].
        sinkhorn_iters: number of Sinkhorn iterations for W_1.
        sinkhorn_reg: entropic regularization for Sinkhorn (smaller = exact).

    Returns:
        Symmetric (N, N) tensor of ORC values, with zeros on the diagonal.
        Note: only the upper-triangle is computed; lower-triangle is mirrored.

    Raises:
        ValueError: if N < 2 or k < 1.
    """
    N = points.shape[0]
    if N < 2:
        raise ValueError(f"Need at least 2 points, got {N}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > N - 1:
        # Clamp k to N-1 (can't have more neighbors than other points)
        k = N - 1
    distances = _pairwise_distances(points)  # (N, N)
    nn_idx = _knn_indices(distances, k)  # (N, k) — neighbors of each point

    orc = torch.zeros((N, N), device=points.device, dtype=points.dtype)
    for i in range(N):
        for j in range(i + 1, N):
            d_ij = distances[i, j].clamp(min=1e-12)
            # Build neighborhoods: N_i = {i} ∪ N_k(i), N_j = {j} ∪ N_k(j)
            nbr_i = torch.cat([
                torch.tensor([i], device=points.device),
                nn_idx[i],
            ])
            nbr_j = torch.cat([
                torch.tensor([j], device=points.device),
                nn_idx[j],
            ])
            mu_i_pts = points[nbr_i]  # (k+1, d)
            mu_j_pts = points[nbr_j]  # (k+1, d)
            # Uniform distributions
            m = mu_i_pts.shape[0]
            n = mu_j_pts.shape[0]
            a = torch.ones(m, device=points.device) / m
            b = torch.ones(n, device=points.device) / n
            # Cost matrix
            C = torch.cdist(mu_i_pts.unsqueeze(0), mu_j_pts.unsqueeze(0)).squeeze(0)
            C = C + 1e-12
            # W_1 via Sinkhorn
            T = _sinkhorn_transport(C, a, b, reg=sinkhorn_reg, n_iters=sinkhorn_iters)
            w1 = (C * T).sum()
            orc_ij = 1.0 - w1 / d_ij
            orc[i, j] = orc_ij
            orc[j, i] = orc_ij
    return orc


def mean_ollivier_ricci(
    points: torch.Tensor,
    k: int = 2,
    sinkhorn_iters: int = 10,
    sinkhorn_reg: float = 0.1,
) -> torch.Tensor:
    """Mean ORC over all edges of the k-NN graph.

    Args:
        points: (N, d) tensor of points.
        k: number of nearest neighbors.
        sinkhorn_iters: Sinkhorn iterations.
        sinkhorn_reg: Sinkhorn regularization.

    Returns:
        Scalar tensor — the mean ORC over all N*(N-1)/2 edges.
    """
    orc = ollivier_ricci_curvature(
        points, k=k, sinkhorn_iters=sinkhorn_iters, sinkhorn_reg=sinkhorn_reg,
    )
    N = orc.shape[0]
    # Mean over upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones((N, N), device=orc.device, dtype=torch.bool), diagonal=1)
    return orc[mask].mean()


def curvature_routing_loss(
    expert_features: torch.Tensor,
    k: int = 2,
    lambda_coeff: float = 0.001,
    sinkhorn_iters: int = 10,
    sinkhorn_reg: float = 0.1,
) -> torch.Tensor:
    """Curvature-routing loss for MoE expert disentanglement.

    Penalizes low mean ORC, encouraging a tree-like (diverse) expert manifold::

        L = λ * mean(1 - ORC)

    Interpretation:
    - High ORC → experts in different regions → low loss
    - Low/negative ORC → experts clustered → high loss

    Args:
        expert_features: (K, d) tensor of per-expert features.
        k: number of nearest neighbors for the k-NN graph.
        lambda_coeff: penalty weight (small value, e.g. 0.001).
        sinkhorn_iters: Sinkhorn iterations for W_1.
        sinkhorn_reg: Sinkhorn regularization.

    Returns:
        Scalar tensor — the curvature-routing loss.

    Raises:
        ValueError: if K < 2 or lambda_coeff < 0.
    """
    K = expert_features.shape[0]
    if K < 2:
        raise ValueError(f"Need at least 2 experts, got {K}")
    if lambda_coeff < 0:
        raise ValueError(f"lambda_coeff must be >= 0, got {lambda_coeff}")
    mean_orc = mean_ollivier_ricci(
        expert_features,
        k=k,
        sinkhorn_iters=sinkhorn_iters,
        sinkhorn_reg=sinkhorn_reg,
    )
    return lambda_coeff * (1.0 - mean_orc)


__all__ = [
    "ollivier_ricci_curvature",
    "mean_ollivier_ricci",
    "curvature_routing_loss",
]
