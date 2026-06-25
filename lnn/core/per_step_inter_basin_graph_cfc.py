"""PerStepInterBasinGraphCfCCell (round 260).

Tests whether **input-dependent adjacency** improves over r258's
static learned adjacency. Inherits ``InterBasinGraphCfCCell`` (round
258) and replaces the static A ∈ R^{K×K} per branch with a
**per-timestep** A_t = softmax(MLP(x_t)) — a small MLP that maps the
input at time t to a row-stochastic K×K adjacency.

This is the basin-level analog of TND's per-neuron dynamics
(arXiv:2606.21295 Cai & Zhao 2026): r258's adjacency is fixed
per branch (shared operator), while r260's adjacency is conditioned
on the current input (per-neuron-like operator).

Hypotheses (PRD #10-97):

  H1: per-step A_t improves over static A on structured (input-
      dependent routing helps when the input carries information
      about which basin to activate).
  H2: H_per_branch becomes MORE VARIABLE across timesteps (the
      input signal reaches the basin graph).
  H3: per-step A protects against r258's static-A overfitting on
      random (different A per timestep = soft denoiser per step).

API::

    PerStepInterBasinGraphCfCCell(input_size, hidden_size, n_branches=4,
                                   n_basin=3, n_hops=1, d_min=1.0,
                                   mlp_hidden=0, sym_lambda=0.0,
                                   sparse_lambda=0.0)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.inter_basin_graph_cfc import (
    InterBasinGraphCfCCell,
    basin_assignment_prob,
    inter_basin_graph_regularizer,
)


def input_dependent_adjacency(
    x_t: torch.Tensor,
    mlp: nn.Module,
    n_basin: int,
) -> torch.Tensor:
    """Compute per-batch row-stochastic adjacency from input.

    Args:
        x_t: (B, d_in) input at time t.
        mlp: nn.Module mapping (B, d_in) → (B, n_basin*n_basin) logits.
        n_basin: Number of basins (K).

    Returns:
        (B, n_basin, n_basin) row-stochastic adjacency.
    """
    B = x_t.shape[0]
    logits = mlp(x_t)  # (B, K*K)
    A = logits.view(B, n_basin, n_basin)
    return torch.softmax(A, dim=-1)


def batched_graph_mix(p: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Apply row-stochastic graph mix per-batch element.

    Args:
        p: (B, K) assignment probabilities.
        A: (B, K, K) row-stochastic adjacency per batch.

    Returns:
        (B, K) graph-mixed and renormalized.
    """
    # q_b[i] = sum_j p_b[j] * A_b[j, i]   (bmm over batch).
    q = torch.bmm(p.unsqueeze(1), A).squeeze(1)  # (B, K)
    q_sum = q.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return q / q_sum


class PerStepInterBasinGraphCfCCell(InterBasinGraphCfCCell):
    """r258 with input-dependent adjacency A_t = softmax(MLP(x_t)).

    The static A is REPLACED by a per-step A_t. The static A's
    parameters remain in the model (for symmetry-break /
    sparsity regularizers to act on) but are not used in the
    forward pass — they serve as a learned bias term.

    Args:
        mlp_hidden: Hidden width of the per-step MLP (0 = no hidden).
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
        self.mlp_hidden = int(mlp_hidden)
        # Per-step MLP: x_t → K*K logits (row softmax applied after).
        if mlp_hidden > 0:
            self.a_mlp = nn.Sequential(
                nn.Linear(input_size, mlp_hidden),
                nn.Tanh(),
                nn.Linear(mlp_hidden, n_basin * n_basin),
            )
        else:
            self.a_mlp = nn.Linear(input_size, n_basin * n_basin)
        # Initialize MLP small so A_t ≈ uniform at start (r258-like init).
        for m in self.a_mlp.modules() if isinstance(self.a_mlp, nn.Sequential) else [self.a_mlp]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    # Bias toward identity-like initial adjacency:
                    # set bias such that softmax(bias) ≈ I.
                    if m.out_features == n_basin * n_basin:
                        bias = torch.zeros(n_basin * n_basin)
                        for i in range(n_basin):
                            bias[i * n_basin + i] = 1.0
                        with torch.no_grad():
                            m.bias.copy_(bias)
                    else:
                        nn.init.zeros_(m.bias)

    def per_step_adjacency(self, x_t: torch.Tensor, k: int) -> torch.Tensor:
        """Compute per-step adjacency for branch k.

        Args:
            x_t: (B, d_in) input.
            k: Branch index (unused but kept for API symmetry).

        Returns:
            (B, n_basin, n_basin) row-stochastic adjacency.
        """
        del k  # branch index unused — same MLP shared across branches
        return input_dependent_adjacency(
            x_t, self.a_mlp, self.n_basin,
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
        A_per_branch = []  # (B, K, K) per step, per branch

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
        # Stack A_per_branch over branches: (n_branches, B, K, K).
        A_t_t = torch.stack(A_per_branch, dim=0)

        sep = self.per_branch_separation_loss()
        ibl = self.inter_basin_loss()
        cbl = self.cross_branch_loss()
        # Regularizers apply to the STATIC A (kept as a learned bias).
        A_static = self.adjacency_stochastic()
        graph_reg_static = inter_basin_graph_regularizer(A_static)
        # Plus a per-step A diversity regularizer: mean ||A_t - A_static||_F
        # (encourages per-step A to differ from static baseline).
        if A_t_t.dim() == 4:
            diversity = (A_t_t - A_static.unsqueeze(1)).pow(2).mean()
        else:
            diversity = torch.tensor(0.0)

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
            "adjacency_per_step": A_t_t,  # (n_branches, B, K, K)
            "graph_symmetry": graph_reg_static["symmetry_break"],
            "graph_sparsity": graph_reg_static["sparsity"],
            "graph_loss_total": graph_const,
            "p_per_branch": p_per_branch_t,
            "q_per_branch": q_per_branch_t,
            "A_diversity": diversity,
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
    "PerStepInterBasinGraphCfCCell",
    "input_dependent_adjacency",
    "batched_graph_mix",
]