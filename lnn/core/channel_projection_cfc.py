"""ChannelProjectionCfCCell (round 262).

A new mechanism for **multi-channel inputs** (d_in > 1). The cell
learns a projection from the raw multi-channel input to a 1D
"routing context" feature, which is then used for basin graph
routing. This addresses the toy-regime bottleneck identified in
r260/r261 — with d_in=1, the input MLP has no real signal to
extract; with d_in>1, a learned projection can capture useful
routing features.

The mechanism is two-stage:

  1. **Channel projection**: c_t = LayerNorm(W_c @ x_t)   (B, d_ctx)
  2. **Routing**: A_t = softmax(MLP(c_t))                (B, K, K)
  3. **Forward pass**: unchanged (still uses raw x_t for the
     CfC dynamics, like r260).

The projection is what enables the cell to USE the multi-channel
input. Without it (d_in=1), the projection is identity.

Hypotheses (PRD #10-99):

  H1: with d_in=4 input, r262 beats r260 on at least one dataset
      (the projection extracts useful features that MLP alone misses).
  H2: c_t has more variance than x_t (projection amplifies signal).
  H3: r262 is a strict superset of r260 (the projection can learn
      to be identity when not needed).

API::

    ChannelProjectionCfCCell(input_size, hidden_size, n_branches=4,
                             n_basin=3, d_ctx=8, mlp_hidden=0)
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


class ChannelProjectionCfCCell(PerStepInterBasinGraphCfCCell):
    """r260 + learnable channel projection before routing.

    Adds:
      * `self.channel_proj` — Linear(d_in, d_ctx) layer that
        projects multi-channel input to a routing context.
      * Routing uses the projected context, not the raw input.
      * The forward pass (CfC dynamics) is unchanged.

    Args:
        d_ctx: Width of the projected context (default 8).
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
        d_ctx: int = 8,
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
        self.d_ctx = int(d_ctx)
        # Channel projection: x_t (B, d_in) → c_t (B, d_ctx).
        self.channel_proj = nn.Linear(input_size, d_ctx)
        # MLP from c_t to A_t logits.
        if mlp_hidden > 0:
            self.a_mlp = nn.Sequential(
                nn.Linear(d_ctx, mlp_hidden),
                nn.Tanh(),
                nn.Linear(mlp_hidden, n_basin * n_basin),
            )
        else:
            self.a_mlp = nn.Linear(d_ctx, n_basin * n_basin)
        # Initialize small so A_t ≈ uniform at start.
        for m in self.a_mlp.modules() if isinstance(self.a_mlp, nn.Sequential) else [self.a_mlp]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None and m.out_features == n_basin * n_basin:
                    bias = torch.zeros(n_basin * n_basin)
                    for i in range(n_basin):
                        bias[i * n_basin + i] = 1.0
                    with torch.no_grad():
                        m.bias.copy_(bias)
                elif m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Init channel_proj small so projection ≈ 0 at start.
        nn.init.normal_(self.channel_proj.weight, std=0.1)
        nn.init.zeros_(self.channel_proj.bias)

    def project_input(self, x_t: torch.Tensor) -> torch.Tensor:
        """Project raw input to routing context.

        Args:
            x_t: (B, d_in) raw input.

        Returns:
            (B, d_ctx) routing context.
        """
        return self.channel_proj(x_t)

    def per_step_adjacency(self, x_t: torch.Tensor, k: int) -> torch.Tensor:
        """Compute per-step adjacency for branch k using projected context.

        Args:
            x_t: (B, d_in) raw input.
            k: Branch index (unused).

        Returns:
            (B, n_basin, n_basin) row-stochastic adjacency.
        """
        del k
        c_t = self.project_input(x_t)
        return input_dependent_adjacency(
            c_t, self.a_mlp, self.n_basin,
        )

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

        c_t = self.project_input(x_t)  # (B, d_ctx)

        for k, (h_prev_k, h_next_k) in enumerate(zip(h_list, outs)):
            p = basin_assignment_prob(
                h_next_k, self.basin_centers[k], beta_v=self.beta_v,
            )
            A_t = self.per_step_adjacency(x_t, k)  # (B, K, K)
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

        V_per_branch_t = torch.stack(V_per_branch)
        H_per_branch_graph_t = torch.stack(H_per_branch_graph)
        H_per_branch_raw_t = torch.stack(H_per_branch_raw)
        lyap_per_branch_t = torch.stack(lyap_per_branch)
        p_per_branch_t = torch.stack(p_per_branch)
        q_per_branch_t = torch.stack(q_per_branch)
        A_t_t = torch.stack(A_per_branch, dim=0)

        sep = self.per_branch_separation_loss()
        ibl = self.inter_basin_loss()
        cbl = self.cross_branch_loss()
        A_static = self.adjacency_stochastic()
        graph_reg_static = inter_basin_graph_regularizer(A_static)

        lyap_const = lyap_lambda * lyap_per_branch_t.sum()
        graph_const = (self.sym_lambda * graph_reg_static["symmetry_break"]
                       + self.sparse_lambda * graph_reg_static["sparsity"])

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
            "adjacency_per_step": A_t_t,
            "graph_symmetry": graph_reg_static["symmetry_break"],
            "graph_sparsity": graph_reg_static["sparsity"],
            "graph_loss_total": graph_const,
            "p_per_branch": p_per_branch_t,
            "q_per_branch": q_per_branch_t,
            "routing_context": c_t,  # NEW: (B, d_ctx) projected
            "routing_context_var": c_t.var(dim=0, unbiased=False).mean(),  # NEW
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


__all__ = ["ChannelProjectionCfCCell"]