"""InterBasinGraphCfCCell (round 258).

The structural gap identified in
``docs/research/2026-06-25_round257_bridge_to_neuronwise_research.md``
after r257 (inter-basin repulsion):

  "after r257, the basin centers are forced to be geometrically separated,
   but they still act independently through the softmax. The 2026 frontier
   (TND, MA-GLTC) shows the next step is to add an explicit interaction
   operator between the per-basin units, not just a geometric separation."

Round 258 introduces a **learned sparse basin adjacency matrix** A ∈ ℝ^{K×K}
that mediates inter-basin message passing within each branch:

  1. Basin assignment probability (existing, from r248):
       p_i = softmax(-β_v * ||h_k - c_k_i||²)
  2. Graph-mixed assignment (NEW, r258):
       q_i = (A_k @ p)[i]         # linear mixing
       q = q / q.sum()             # renormalize
  3. Aux path uses q (not p) for entropy / lyapunov values.

The adjacency A_k is row-stochastic via softmax over rows in every forward
pass (so it remains a proper probability distribution over source basins).

Auxiliary regularizers (train-time, opt-in):

  * **symmetry break** = ||A - A^T||_F²       — encourages directed graph
  * **sparsity**       = ||A||_1              — encourages sparse, interpretable A
  * **stochastic**     = handled via softmax  — invariant on construction

The forward pass itself is unchanged from r257; only the **aux** path is
modified to use the graph-mixed assignment. This keeps the cell
non-invasive (no risk of breaking r257's strict-win behavior) while
adding the structural coupling that the bridge document calls for.

Hypotheses (PRD #10-95):

  H1: graph mix INCREASES basin selectivity (lower H_per_branch final vs r257)
      because the directed graph biases routing toward a subset of basins.
  H2: r258 (with auxiliary regularizers) matches or beats r257_d2 on
      toy_sin and random while preserving structured gains.
  H3: the learned adjacency matrix A becomes SPARSE (avg |A_off_diag| < 0.1)
      and ASYMMETRIC (||A - A^T||_F > 0.1) after training — evidence the
      graph learns structure rather than collapsing to identity.

API::

    InterBasinGraphCfCCell(input_size, hidden_size, n_branches=4,
                            n_basin=3, d_min=1.0,
                            sym_lambda=0.0, sparse_lambda=0.0)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.inter_basin_distance_cfc import (
    InterBasinDistanceCfCCell,
)


def basin_assignment_prob(
    h: torch.Tensor,
    basin_centers: torch.Tensor,
    beta_v: float = 2.0,
) -> torch.Tensor:
    """Soft basin assignment probabilities (raw form).

    Args:
        h: (batch, hidden_size) hidden state.
        basin_centers: (n_basin, hidden_size).
        beta_v: Soft-min temperature.

    Returns:
        (batch, n_basin) softmax(-beta_v * ||h - c_i||^2).
    """
    # Squared distances: (B, K).
    diff = h.unsqueeze(1) - basin_centers.unsqueeze(0)
    dist_sq = (diff * diff).sum(dim=-1)
    return torch.softmax(-float(beta_v) * dist_sq, dim=-1)


def inter_basin_graph_mix(p: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Apply row-stochastic graph mix to assignment probabilities.

    Args:
        p: (batch, n_basin) — soft assignment probabilities (sum to 1 per row).
        A: (n_basin, n_basin) — adjacency matrix (already row-stochastic).

    Returns:
        (batch, n_basin) — graph-mixed assignment, renormalized to sum 1.
    """
    q = p @ A.t()  # (B, K) — graph propagation
    q_sum = q.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return q / q_sum


def inter_basin_graph_regularizer(A: torch.Tensor) -> dict[str, torch.Tensor]:
    """Auxiliary regularizers for the learned adjacency matrix.

    Args:
        A: (n_basin, n_basin) OR (n_branches, n_basin, n_basin) — row-stochastic.

    Returns:
        Dict with keys:
          * ``symmetry_break`` — ||A - A^T||_F^2 summed across branches (large
            = directed graph, small = symmetric)
          * ``sparsity``       — ||A||_1 summed across branches (encourages
            sparse A; small = sparse, large = dense)
          * ``total``          — combined scalar for convenience.
    """
    if A.dim() == 3:
        # (n_branches, n_basin, n_basin) → per-branch symmetric + sparsity.
        sym = (A - A.transpose(-1, -2)).pow(2).sum()
        sp = A.abs().sum()
    else:
        sym = (A - A.t()).pow(2).sum()
        sp = A.abs().sum()
    return {
        "symmetry_break": sym,
        "sparsity": sp,
        "total": sym + sp,
    }


class InterBasinGraphCfCCell(InterBasinDistanceCfCCell):
    """r257 + learned sparse basin adjacency (round 258).

    Adds:
      * ``self.adjacency_k`` — (n_branches, n_basin, n_basin) learnable
        matrix; row-stochastic via softmax over rows in every forward pass.
      * Graph-mixed basin assignments in the aux path.
      * Optional auxiliary regularizers for symmetry + sparsity.

    Args:
        sym_lambda: Weight for symmetry-break loss (0 = off).
        sparse_lambda: Weight for sparsity loss (0 = off).
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
        sym_lambda: float = 0.0,
        sparse_lambda: float = 0.0,
        adj_init_scale: float = 0.1,
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
            d_min=d_min,
            cross_branch_lambda=cross_branch_lambda,
        )
        # Adjacency per branch: (n_branches, n_basin, n_basin).
        # Initialized small so q ≈ p at start (graph-mix ≈ identity at init).
        adj = torch.randn(self.n_branches, self.n_basin, self.n_basin)
        adj = adj * float(adj_init_scale)
        # Pre-fill diagonal with small positive so initial softmax is roughly
        # identity-like (each basin mostly routes to itself).
        for k in range(self.n_branches):
            for i in range(self.n_basin):
                adj[k, i, i] += 1.0
        self.adjacency = nn.Parameter(adj)

        self.sym_lambda = float(sym_lambda)
        self.sparse_lambda = float(sparse_lambda)

    def adjacency_stochastic(self) -> torch.Tensor:
        """Row-stochastic projection: softmax over rows."""
        return torch.softmax(self.adjacency, dim=-1)

    def graph_mixed_assignment(
        self, h_k: torch.Tensor, k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (p_raw, q_graph) for branch k.

        p_raw: (B, K) raw softmax(-beta_v * dist_sq).
        q_graph: (B, K) graph-mixed and renormalized.
        """
        p = basin_assignment_prob(
            h_k, self.basin_centers[k], beta_v=self.beta_v,
        )
        A = self.adjacency_stochastic()[k]
        q = inter_basin_graph_mix(p, A)
        return p, q

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h_list: list[torch.Tensor],
        lyap_lambda: float = 0.0,
        sep_lambda: float = 0.0,
        dist_lambda: float = 1.0,
        graph_lambda: float = 1.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        h_next, outs = self.forward(x_t, h_list)

        V_per_branch = []
        H_per_branch_raw = []
        H_per_branch_graph = []
        lyap_per_branch = []
        p_per_branch = []
        q_per_branch = []

        A = self.adjacency_stochastic()

        for k, (h_prev_k, h_next_k) in enumerate(zip(h_list, outs)):
            # Use the GRAPH-MIXED assignment for entropy reporting.
            p, q = self.graph_mixed_assignment(h_next_k, k)
            H_per_branch_graph.append(
                -(q * (q.clamp(min=1e-12)).log()).sum(dim=-1).mean(),
            )
            H_per_branch_raw.append(
                -(p * (p.clamp(min=1e-12)).log()).sum(dim=-1).mean(),
            )
            # V_per_branch uses the existing multi-basin lyapunov value
            # (geometric, not affected by graph mix — graph mix is
            # about assignment, not geometry).
            V_per_branch.append(
                self.per_branch_lyapunov_value(h_next_k, k).mean(),
            )
            lyap_per_branch.append(
                self.per_branch_lyap_decay(h_prev_k, h_next_k, k),
            )
            p_per_branch.append(p.mean(dim=0))
            q_per_branch.append(q.mean(dim=0))

        V_per_branch_t = torch.stack(V_per_branch)
        H_per_branch_graph_t = torch.stack(H_per_branch_graph)
        H_per_branch_raw_t = torch.stack(H_per_branch_raw)
        lyap_per_branch_t = torch.stack(lyap_per_branch)
        p_per_branch_t = torch.stack(p_per_branch)  # (n_branches, n_basin)
        q_per_branch_t = torch.stack(q_per_branch)

        sep = self.per_branch_separation_loss()
        ibl = self.inter_basin_loss()
        cbl = self.cross_branch_loss()
        graph_reg = inter_basin_graph_regularizer(A)

        lyap_const = lyap_lambda * lyap_per_branch_t.sum()
        graph_const = (self.sym_lambda * graph_reg["symmetry_break"]
                       + self.sparse_lambda * graph_reg["sparsity"])

        aux: dict[str, torch.Tensor] = {
            "h_next": h_next,
            "alpha_mix": self.alpha_mix.detach(),
            "per_branch_V_next": V_per_branch_t,
            "per_branch_basin_H": H_per_branch_graph_t,  # GRAPH-MIXED
            "per_branch_basin_H_raw": H_per_branch_raw_t,
            "mean_basin_H": H_per_branch_graph_t.mean(),
            "mean_basin_H_raw": H_per_branch_raw_t.mean(),
            "lyap_loss": lyap_const,
            "lyap_per_branch": lyap_per_branch_t,
            "sep_loss": sep,
            "inter_basin_loss": ibl,
            "cross_branch_loss": cbl,
            "adjacency_stochastic": A,
            "graph_symmetry": graph_reg["symmetry_break"],
            "graph_sparsity": graph_reg["sparsity"],
            "graph_loss_total": graph_const,
            "p_per_branch": p_per_branch_t,
            "q_per_branch": q_per_branch_t,
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
        if graph_lambda > 0:
            aux["graph_loss_applied"] = graph_const
        return h_next, outs, aux


__all__ = [
    "InterBasinGraphCfCCell",
    "basin_assignment_prob",
    "inter_basin_graph_mix",
    "inter_basin_graph_regularizer",
]