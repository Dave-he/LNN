"""InterBasinDistanceCfCCell (round 257).

Pivots the 11-round arc (r246-256, all about aux gating) to a new
**basin geometry diversification** axis.

  * r246-r252: basin geometry as side-effect of aux supervision
  * r253-r256: gating axis (per-branch / per-step / 2D / time)
  * r257 (this): EXPLICIT inter-basin repulsion — push basin centers
    away from each other so they occupy distinct regions of state space

Existing `per_branch_separation_loss` (r248) uses a small `pd_eps=1e-2`
margin with a hinge loss; this rarely activates because basin
initialization already separates them. Round 257 adds a STRONGER
**d_min-driven repulsion** that:
  * Uses configurable `d_min` (default 1.0 — much larger than pd_eps)
  * Computes pairwise distances across ALL basin centers in each branch
  * Applies quadratic penalty `sum((d_min - dist(c_i, c_j))^2)+` for
    pairs that are too close
  * Optionally applies cross-branch repulsion (push branch-k basin i away
    from branch-j basin i — same-index basins across branches)

Hypothesis (PRD #10-94, round 257):
  H1: increased basin diversity (higher H_per_branch) than r248.
  H2: improved task loss on structured (more diverse geometry helps
      structured data, like r249).
  H3: composition with r252/r256 aux — combines geometric diversification
      with aux supervision.

API::

    InterBasinDistanceCfCCell(input_size, hidden_size, n_branches=4,
                              n_basin=3, d_min=1.0,
                              cross_branch_lambda=0.0)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.per_branch_multibasin_lyapunov_cfc import (
    PerBranchMultiBasinLyapunovCfCCell,
)


def inter_basin_repulsion_loss(
    basin_centers_k: torch.Tensor,
    d_min: float = 1.0,
) -> torch.Tensor:
    """Quadratic repulsion loss pushing basin centers apart.

    Args:
        basin_centers_k: (n_basin, hidden_size) basin centers for one branch.
        d_min: Minimum acceptable distance between any two centers.

    Returns:
        Scalar loss: sum over pairs of max(0, d_min - dist)^2.
    """
    K = basin_centers_k.shape[0]
    if K < 2:
        return torch.tensor(0.0, device=basin_centers_k.device)
    diff = basin_centers_k.unsqueeze(0) - basin_centers_k.unsqueeze(1)
    dist_sq = (diff * diff).sum(dim=-1)  # (K, K)
    # Use max(.,eps) inside sqrt to keep gradient finite when dist=0.
    dist = torch.sqrt(torch.clamp(dist_sq, min=1e-12))
    # Take upper-triangular (i < j) to avoid double-counting.
    iu, ju = torch.triu_indices(K, K, offset=1, device=dist.device)
    pair_dist = dist[iu, ju]
    penalty = torch.clamp(float(d_min) - pair_dist, min=0.0) ** 2
    return penalty.sum()


def cross_branch_repulsion_loss(
    basin_centers: torch.Tensor,
    d_min: float = 1.0,
) -> torch.Tensor:
    """Push same-index basin centers across different branches apart.

    Args:
        basin_centers: (n_branches, n_basin, hidden_size).
        d_min: Minimum distance between (branch_a, basin_i) and
               (branch_b, basin_i) for a != b.

    Returns:
        Scalar loss: sum over (a, b, i) with a != b of max(0, d_min - dist)^2.
    """
    n_branches, n_basin, _ = basin_centers.shape
    if n_branches < 2 or n_basin < 1:
        return torch.tensor(0.0, device=basin_centers.device)
    total = torch.tensor(0.0, device=basin_centers.device)
    for a in range(n_branches):
        for b in range(a + 1, n_branches):
            diff = basin_centers[a] - basin_centers[b]  # (n_basin, hidden)
            dist_sq = (diff * diff).sum(dim=-1)  # (n_basin,)
            dist = torch.sqrt(torch.clamp(dist_sq, min=1e-12))
            penalty = torch.clamp(float(d_min) - dist, min=0.0) ** 2
            total = total + penalty.sum()
    return total


class InterBasinDistanceCfCCell(PerBranchMultiBasinLyapunovCfCCell):
    """PerBranch + inter-basin repulsion (round 257).

    Args:
        d_min: Minimum acceptable pairwise distance within each branch.
        cross_branch_lambda: Weight for cross-branch repulsion (0 = off).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_branches: int = 4,
        n_basin: int = 3,
        tau_min: float = 0.05,
        tau_max: float = 20.0,
        seed: int = 42,
        alpha: float = 0.05,
        beta_v: float = 2.0,
        pd_eps: float = 1e-2,
        d_min: float = 1.0,
        cross_branch_lambda: float = 0.0,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            n_branches=n_branches,
            n_basin=n_basin,
            tau_min=tau_min,
            tau_max=tau_max,
            seed=seed,
            alpha=alpha,
            beta_v=beta_v,
            pd_eps=pd_eps,
            learn_mix=True,
        )
        self.d_min = float(d_min)
        self.cross_branch_lambda = float(cross_branch_lambda)

    def inter_basin_loss(self) -> torch.Tensor:
        """Sum of within-branch repulsion losses across all branches."""
        total = torch.tensor(0.0, device=self.basin_centers.device)
        for k in range(self.n_branches):
            total = total + inter_basin_repulsion_loss(
                self.basin_centers[k], d_min=self.d_min,
            )
        return total

    def cross_branch_loss(self) -> torch.Tensor:
        return cross_branch_repulsion_loss(
            self.basin_centers, d_min=self.d_min,
        )

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h_list: list[torch.Tensor],
        lyap_lambda: float = 0.0,
        sep_lambda: float = 0.0,
        dist_lambda: float = 1.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        h_next, outs = self.forward(x_t, h_list)

        V_per_branch = []
        H_per_branch = []
        lyap_per_branch = []
        for k, (h_prev_k, h_next_k) in enumerate(zip(h_list, outs)):
            V_per_branch.append(self.per_branch_lyapunov_value(h_next_k, k).mean())
            H_per_branch.append(self.per_branch_basin_entropy(h_next_k, k).mean())
            lyap_per_branch.append(self.per_branch_lyap_decay(
                h_prev_k, h_next_k, k,
            ))
        V_per_branch_t = torch.stack(V_per_branch)
        H_per_branch_t = torch.stack(H_per_branch)
        lyap_per_branch_t = torch.stack(lyap_per_branch)

        sep = self.per_branch_separation_loss()
        ibl = self.inter_basin_loss()
        cbl = self.cross_branch_loss()

        lyap_const = lyap_lambda * lyap_per_branch_t.sum()

        aux: dict[str, torch.Tensor] = {
            "h_next": h_next,
            "alpha_mix": self.alpha_mix.detach(),
            "per_branch_V_next": V_per_branch_t,
            "per_branch_basin_H": H_per_branch_t,
            "mean_basin_H": H_per_branch_t.mean(),
            "lyap_loss": lyap_const,
            "lyap_per_branch": lyap_per_branch_t,
            "sep_loss": sep,
            "inter_basin_loss": ibl,
            "cross_branch_loss": cbl,
        }
        if lyap_lambda > 0:
            aux["lyap_loss_total"] = lyap_const
        if sep_lambda > 0:
            aux["sep_loss_total"] = sep_lambda * sep
        if dist_lambda > 0:
            aux["inter_basin_loss_total"] = dist_lambda * ibl
        if self.cross_branch_lambda > 0:
            aux["cross_branch_loss_total"] = (
                self.cross_branch_lambda * cbl
            )
        return h_next, outs, aux


__all__ = [
    "InterBasinDistanceCfCCell",
    "inter_basin_repulsion_loss",
    "cross_branch_repulsion_loss",
]
