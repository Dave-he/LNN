"""MixStaticInterBasinGraphCfCCell (round 261).

Fuses the static learned adjacency (r258) with the input-dependent
adjacency (r260) using a **learnable per-branch mixing coefficient**:

    A_t_k = sigmoid(α_k) * A_input_t + (1 - sigmoid(α_k)) * A_static_k

When α_k → 1: trust input (r260 behavior).
When α_k → 0: trust static (r258 behavior).

This addresses r260's regression on random data — there the static A
provides a useful prior that pure input-dependent A lacks. The
learnable α lets the model decide per-branch whether the input
carries useful basin-routing information.

Hypotheses (PRD #10-98):

  H1: r261 beats both r258 and r260 on at least one dataset (best of
      both worlds when data is mixed).
  H2: learned α differs across branches (some branches specialize on
      input, others on static prior).
  H3: r261 never regresses vs r258 (the static prior provides a
      safety floor).

API::

    MixStaticInterBasinGraphCfCCell(input_size, hidden_size, n_branches=4,
                                     n_basin=3, d_min=2.0,
                                     init_alpha=0.5, sym_lambda=0.0,
                                     sparse_lambda=0.0)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.per_step_inter_basin_graph_cfc import (
    PerStepInterBasinGraphCfCCell,
    basin_assignment_prob,
    batched_graph_mix,
    input_dependent_adjacency,
)
from lnn.core.inter_basin_graph_cfc import inter_basin_graph_regularizer


def mix_static_and_input(
    A_static: torch.Tensor,
    A_input: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Mix static and input-dependent adjacency.

    Args:
        A_static: (B, K, K) row-stochastic per batch (broadcast from K×K).
        A_input: (B, K, K) row-stochastic per batch.
        alpha: scalar in (0, 1) (sigmoid of learned logit).

    Returns:
        (B, K, K) row-stochastic mixed adjacency.
    """
    mixed = alpha * A_input + (1.0 - alpha) * A_static
    # Re-normalize rows (alpha and 1-alpha are scalars; the sum
    # alpha + (1-alpha) = 1, so this is already row-stochastic when
    # A_static and A_input are. But adding two stochastic matrices
    # with weights summing to 1 yields a stochastic matrix.)
    return mixed


class MixStaticInterBasinGraphCfCCell(PerStepInterBasinGraphCfCCell):
    """r260 + learnable static/input mix coefficient per branch.

    Inherits everything from r260 (input-dependent A_t) plus the
    static A from r258. The forward computes a convex combination:

        A_t_k = sigmoid(α_k) * A_input_t + (1 - sigmoid(α_k)) * A_static_k

    `alpha_logit_k` is a learnable scalar per branch; sigmoid(α_k)
    starts at `init_alpha` (default 0.5).
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
        mlp_hidden: int = 0,
        init_alpha: float = 0.5,
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
            mlp_hidden=mlp_hidden,
        )
        self.init_alpha = float(init_alpha)
        # Per-branch alpha logit, initialized so sigmoid(alpha_logit)
        # = init_alpha. Inverse: logit(p) = log(p/(1-p)).
        import math
        init_logit = math.log(init_alpha / (1.0 - init_alpha))
        self.alpha_logit = nn.Parameter(
            torch.full((n_branches,), init_logit),
        )

    def per_branch_alpha(self) -> torch.Tensor:
        """Sigmoid-activated per-branch alpha in (0, 1)."""
        return torch.sigmoid(self.alpha_logit)

    def per_step_adjacency(self, x_t: torch.Tensor, k: int) -> torch.Tensor:
        """Compute mixed adjacency for branch k.

        A_t_k = sigmoid(α_k) * A_input_t + (1 - sigmoid(α_k)) * A_static_k

        Args:
            x_t: (B, d_in) input.
            k: Branch index.

        Returns:
            (B, n_basin, n_basin) row-stochastic adjacency.
        """
        A_input = input_dependent_adjacency(
            x_t, self.a_mlp, self.n_basin,
        )  # (B, K, K)
        A_static_k = self.adjacency_stochastic()[k]  # (K, K)
        # Broadcast to (B, K, K).
        A_static_b = A_static_k.unsqueeze(0).expand_as(A_input)
        alpha_k = torch.sigmoid(self.alpha_logit[k])
        mixed = mix_static_and_input(A_static_b, A_input, alpha_k)
        return mixed

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
        A_per_branch = []
        A_input_per_branch = []

        for k, (h_prev_k, h_next_k) in enumerate(zip(h_list, outs)):
            p = basin_assignment_prob(
                h_next_k, self.basin_centers[k], beta_v=self.beta_v,
            )
            A_t = self.per_step_adjacency(x_t, k)  # (B, K, K)
            A_input = input_dependent_adjacency(
                x_t, self.a_mlp, self.n_basin,
            )  # (B, K, K)
            q = batched_graph_mix(p, A_t)  # (B, K)
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
            A_per_branch.append(A_t)
            A_input_per_branch.append(A_input)

        V_per_branch_t = torch.stack(V_per_branch)
        H_per_branch_graph_t = torch.stack(H_per_branch_graph)
        H_per_branch_raw_t = torch.stack(H_per_branch_raw)
        lyap_per_branch_t = torch.stack(lyap_per_branch)
        p_per_branch_t = torch.stack(p_per_branch)
        q_per_branch_t = torch.stack(q_per_branch)
        A_t_t = torch.stack(A_per_branch, dim=0)  # (n_branches, B, K, K)
        A_input_t = torch.stack(A_input_per_branch, dim=0)

        sep = self.per_branch_separation_loss()
        ibl = self.inter_basin_loss()
        cbl = self.cross_branch_loss()
        A_static = self.adjacency_stochastic()
        graph_reg_static = inter_basin_graph_regularizer(A_static)

        lyap_const = lyap_lambda * lyap_per_branch_t.sum()
        graph_const = (self.sym_lambda * graph_reg_static["symmetry_break"]
                       + self.sparse_lambda * graph_reg_static["sparsity"])

        alpha_per_branch = self.per_branch_alpha()  # (n_branches,)

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
            "adjacency_stochastic": A_static,
            "adjacency_per_step": A_t_t,  # MIXED
            "adjacency_input_only": A_input_t,  # input-only
            "graph_symmetry": graph_reg_static["symmetry_break"],
            "graph_sparsity": graph_reg_static["sparsity"],
            "graph_loss_total": graph_const,
            "p_per_branch": p_per_branch_t,
            "q_per_branch": q_per_branch_t,
            "alpha_per_branch": alpha_per_branch,  # NEW
            "alpha_mean": alpha_per_branch.mean(),  # NEW
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
    "MixStaticInterBasinGraphCfCCell",
    "mix_static_and_input",
]