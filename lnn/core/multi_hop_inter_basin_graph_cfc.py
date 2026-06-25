"""MultiHopInterBasinGraphCfCCell (round 259).

Tests whether **graph depth** (number of message-passing hops) matters
for the inter-basin graph mix introduced in r258.

Inherits ``InterBasinGraphCfCCell`` (round 258, single-hop graph mix
A ∈ R^{K×K} per branch) and iterates the graph propagation n_hops times::

    q_0 = p                      # raw basin assignment
    for t in range(n_hops):
        q_{t+1} = (q_t @ A_k^T) / q_t.sum(-1)    # row-stochastic mix

For ``n_hops=1``, r259 reduces to r258 exactly. Higher ``n_hops``
allows the assignment probability to propagate information across
basin "neighbors" — a soft analog of K-hop message passing in GNNs.

Hypotheses (PRD #10-96):

  H1: K=2 (2-hop) marginally beats K=1 (r258) on structured (more
      propagation helps when basins encode multi-step patterns).
  H2: K=3+ OVER-SMOOTHS — q converges to uniform (H → log K), losing
      the basin-center selectivity that drove r258's gain.
  H3: r259_hop2 IS THE NEW BEST on structured AND preserves r258's
      random -50% gain.

The 2026 analog: MA-GLTC (arXiv:2606.15807) uses graph-coupled
recurrent conductance — multiple timesteps of neighbor information
mixing. r259 tests whether K-step message passing in the BASIN
graph (not state graph) helps.

API::

    MultiHopInterBasinGraphCfCCell(input_size, hidden_size, n_branches=4,
                                    n_basin=3, n_hops=2, d_min=1.0,
                                    sym_lambda=0.0, sparse_lambda=0.0)
"""

from __future__ import annotations

import torch

from lnn.core.inter_basin_graph_cfc import (
    InterBasinGraphCfCCell,
    inter_basin_graph_mix,
)


class MultiHopInterBasinGraphCfCCell(InterBasinGraphCfCCell):
    """r258 with n_hops iterations of inter-basin message passing.

    Args:
        n_hops: Number of message-passing iterations. n_hops=1 == r258.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_branches: int = 4,
        n_basin: int = 3,
        n_hops: int = 2,
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
            sym_lambda=sym_lambda,
            sparse_lambda=sparse_lambda,
            adj_init_scale=adj_init_scale,
        )
        self.n_hops = max(1, int(n_hops))

    def multi_hop_mix(
        self, p: torch.Tensor, A: torch.Tensor,
    ) -> torch.Tensor:
        """Apply A repeatedly n_hops times, starting from p.

        Each step uses row-stochastic renormalization so q always sums to 1.
        """
        q = p
        for _ in range(self.n_hops):
            q = inter_basin_graph_mix(q, A)
        return q

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
        q_per_hop_per_branch: list[list[torch.Tensor]] = []

        A = self.adjacency_stochastic()

        for k, (h_prev_k, h_next_k) in enumerate(zip(h_list, outs)):
            p = self.basin_assignment_prob_single(h_next_k, k)
            q = self.multi_hop_mix(p, A[k])
            # H_graph at FINAL hop (most informative).
            H_per_branch_graph.append(
                -(q * (q.clamp(min=1e-12)).log()).sum(dim=-1).mean(),
            )
            H_per_branch_raw.append(
                -(p * (p.clamp(min=1e-12)).log()).sum(dim=-1).mean(),
            )
            V_per_branch.append(
                self.per_branch_lyapunov_value(h_next_k, k).mean(),
            )
            lyap_per_branch.append(
                self.per_branch_lyap_decay(h_prev_k, h_next_k, k),
            )
            p_per_branch.append(p.mean(dim=0))
            q_per_branch.append(q.mean(dim=0))
            # Track q at each hop (mean over batch) for trajectory analysis.
            q_traj = [p.mean(dim=0)]
            q_cur = p
            for _ in range(self.n_hops):
                q_cur = inter_basin_graph_mix(q_cur, A[k])
                q_traj.append(q_cur.mean(dim=0))
            q_per_hop_per_branch.append(q_traj)

        V_per_branch_t = torch.stack(V_per_branch)
        H_per_branch_graph_t = torch.stack(H_per_branch_graph)
        H_per_branch_raw_t = torch.stack(H_per_branch_raw)
        lyap_per_branch_t = torch.stack(lyap_per_branch)
        p_per_branch_t = torch.stack(p_per_branch)
        q_per_branch_t = torch.stack(q_per_branch)

        sep = self.per_branch_separation_loss()
        ibl = self.inter_basin_loss()
        cbl = self.cross_branch_loss()
        graph_reg = self.inter_basin_graph_regularizer_fn(A)

        lyap_const = lyap_lambda * lyap_per_branch_t.sum()
        graph_const = (self.sym_lambda * graph_reg["symmetry_break"]
                       + self.sparse_lambda * graph_reg["sparsity"])

        aux: dict[str, torch.Tensor] = {
            "h_next": h_next,
            "alpha_mix": self.alpha_mix.detach(),
            "per_branch_V_next": V_per_branch_t,
            "per_branch_basin_H": H_per_branch_graph_t,
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
            "n_hops": self.n_hops,
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

    # --- Helpers (factor out to avoid duplicating parent's logic) ---

    def basin_assignment_prob_single(self, h_k: torch.Tensor, k: int
                                      ) -> torch.Tensor:
        from lnn.core.inter_basin_graph_cfc import basin_assignment_prob
        return basin_assignment_prob(
            h_k, self.basin_centers[k], beta_v=self.beta_v,
        )

    def inter_basin_graph_regularizer_fn(self, A: torch.Tensor
                                          ) -> dict[str, torch.Tensor]:
        from lnn.core.inter_basin_graph_cfc import inter_basin_graph_regularizer
        return inter_basin_graph_regularizer(A)


__all__ = ["MultiHopInterBasinGraphCfCCell"]