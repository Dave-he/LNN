"""Round 120 — DAG-MoE (Structural Aggregation) for CfC.

Response to arXiv:2606.01062 (Feng et al., ICML 2026):
"DAG-MoE: From Simple Mixture to Structural Aggregation in
 Mixture-of-Experts".

DAG-MoE replaces the standard MoE's permutation-invariant weighted
summation of expert outputs with a directed acyclic graph (DAG)
over the selected K experts.  Each expert occupies a distinct
structural role, and a lightweight DAG learning module refines
the aggregation over L iterations with learned edge gates.

Forward pass (L iterations of DAG refinement, K experts):
  x_i^0 = g_{k[i]}(x) * E_{k[i]}(x) + (1/K) * x
  x_{i,down}^l = W_down^l * LayerNorm(x_i^{l-1})
  e_{(i,j)}^l = sigmoid(W_edge * Concat(x_{i,down}^l, x_{j,down}^l))
  x_i^l = W_up^l * Sum_j ( e_{(i,j)}^l * W_node * Concat(x_{i,down}^l, x_{j,down}^l) )
          + x_i^{l-1}
  y = Sum_i x_i^L  (sum depth-L node representations)

Notes
-----
* All sub-MLP experts (CfC cells), consistent with the 91-119 audit
  pattern (8 winners all use sub-MLP experts).
* The router is unchanged (top-K softmax on combined [x_t, h]).
* Up-projection is zero-initialized for early-training stability.
* The DAG is fully connected at every depth — every node attends
  to every previous-depth node via the edge gate.  This is
  structurally similar to a single-layer graph attention block
  (GAT) but applied within one MoE layer's expert pool.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class DAGEdgeGate(nn.Module):
    """Learnable edge gate between two DAG nodes.

    For each pair (i, j), computes e_{(i,j)} = sigmoid(W_edge *
    Concat(x_i_down, x_j_down)).  Edge gate is a scalar per (i, j)
    pair, modulating W_node's contribution to node i from node j.

    The "down" projection is shared across all pairs at the same
    depth (so we don't re-project the same node K times).
    """

    def __init__(self, hidden_size: int, down_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.down_dim = down_dim
        # Down-projection per depth (shared across all pairs)
        self.W_down = nn.Linear(hidden_size, down_dim, bias=True)
        # Edge gate: concat(x_i, x_j) -> scalar in [0, 1]
        self.W_edge = nn.Linear(2 * down_dim, 1, bias=True)
        # Node projection: concat(x_i, x_j) -> hidden_size
        self.W_node = nn.Linear(2 * down_dim, hidden_size, bias=True)
        # Up-projection: hidden_size -> hidden_size (zero-initialized)
        self.W_up = nn.Linear(hidden_size, hidden_size, bias=True)
        with torch.no_grad():
            self.W_up.weight.zero_()
            if self.W_up.bias is not None:
                self.W_up.bias.zero_()

    def forward(
        self,
        node_outs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one DAG refinement iteration.

        Parameters
        ----------
        node_outs : [B, K, hidden_size]  current node representations

        Returns
        -------
        new_outs : [B, K, hidden_size]  refined node representations
        """
        B, K, H = node_outs.shape
        # Down-project all nodes: [B, K, down_dim]
        normed = F.layer_norm(node_outs, (H,))
        x_down = self.W_down(normed)  # [B, K, D]
        # Pairwise concat: [B, K, K, 2*D] (all i, all j)
        x_i = x_down.unsqueeze(2).expand(B, K, K, self.down_dim)
        x_j = x_down.unsqueeze(1).expand(B, K, K, self.down_dim)
        pair = torch.cat([x_i, x_j], dim=-1)  # [B, K, K, 2D]
        # Edge gate per (i, j): [B, K, K]
        e = torch.sigmoid(self.W_edge(pair).squeeze(-1))
        # Node projection per (i, j): [B, K, K, H]
        node_proj = self.W_node(pair)
        # Modulate by edge gate
        gated = e.unsqueeze(-1) * node_proj  # [B, K, K, H]
        # Sum over j (predecessors)
        agg = gated.sum(dim=2)  # [B, K, H]
        # Up-project with residual
        new_outs = self.W_up(agg) + node_outs  # [B, K, H]
        return new_outs


class DAGAggregation(nn.Module):
    """L iterations of DAG refinement over K nodes.

    Each iteration applies an independent DAGEdgeGate with its own
    parameters, so the network can learn different edge patterns at
    different depths.
    """

    def __init__(self, hidden_size: int, n_nodes: int, n_iterations: int = 2, down_dim: int = 8):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_iterations = n_iterations
        self.layers = nn.ModuleList(
            [DAGEdgeGate(hidden_size, down_dim) for _ in range(n_iterations)]
        )

    def forward(self, node_outs: torch.Tensor) -> torch.Tensor:
        """Apply L iterations of DAG refinement.

        Parameters
        ----------
        node_outs : [B, K, hidden_size]

        Returns
        -------
        final_outs : [B, K, hidden_size]  depth-L node representations
        """
        h = node_outs
        for layer in self.layers:
            h = layer(h)
        return h


# ---------------------------------------------------------------------------
# Cell + Network
# ---------------------------------------------------------------------------


class DAGMoECfCCell(nn.Module):
    """DAG-MoE CfC cell.

    Structure:
      base_cfc : shared base CfC (CfCCell)
      experts  : K sub-CfC experts (CfCCell, with output_size=hidden_size)
      router   : K-dim softmax over [x_t, h] (top-K sparse)
      dag      : DAGAggregation(L iterations)

    Forward:
      h_base = base_cfc(x_t, h, dt)
      scores = router([x_t, h])           # [B, K]
      g, top_idx = top_k_sparse(scores, top_k)  # [B, K], [B, K]
      # For each selected expert compute its delta contribution
      h_experts = sum_i g_i * E_i([x_t, h])    # [B, K, H] but [B, H] after sum
      # Init node_outs: g_i * E_i(x) + (1/K) * h_base
      node_outs = stack(g_i * E_i(x) for i in top_idx) + (1/K) * h_base.unsqueeze(1)
      # DAG refine
      refined = dag(node_outs)              # [B, K, H]
      h_new = h_base + sum_i refined[:, i, :]  # [B, H]

    Notes:
      - Top-K must be K=K (all experts in pool) since DAG is over them.
        We can use top_k = K (full pool) for the simplest variant, or
        a smaller K for sparsity.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 3,
        top_k: int = 3,
        n_dag_iterations: int = 2,
        dag_down_dim: int = 8,
        use_residual: bool = True,
    ):
        super().__init__()
        if top_k > n_experts:
            raise ValueError(f"top_k={top_k} cannot exceed n_experts={n_experts}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = n_experts
        self.top_k = top_k
        self.use_residual = use_residual

        from lnn.core.cfc import CfCCell

        # Shared base CfC
        self.base_cfc = CfCCell(input_size, hidden_size)
        # K sub-MLP experts (CfC cells, output_size=hidden_size)
        self.experts = nn.ModuleList(
            [CfCCell(input_size, hidden_size) for _ in range(n_experts)]
        )
        # Router: K-dim softmax over [x_t, h]
        self.router = nn.Linear(input_size + hidden_size, n_experts, bias=True)
        # DAG aggregation over selected experts
        self.dag = DAGAggregation(
            hidden_size=hidden_size,
            n_nodes=top_k,
            n_iterations=n_dag_iterations,
            down_dim=dag_down_dim,
        )

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """One step forward.

        Parameters
        ----------
        x_t : [B, I]  input at this step
        h : [B, H]  hidden state
        dt : float  time delta (for CfC)

        Returns
        -------
        h_new : [B, H]  new hidden state
        """
        B = x_t.size(0)
        # Base CfC
        h_base = self.base_cfc(x_t, h, dt=dt)  # [B, H]
        # Router
        combined = torch.cat([x_t, h], dim=-1)  # [B, I+H]
        scores = self.router(combined)  # [B, K]
        # Sparse top-K routing
        top_k = self.top_k
        top_scores, top_idx = scores.topk(top_k, dim=-1)  # [B, k], [B, k]
        g = F.softmax(top_scores, dim=-1)  # [B, k]
        # Compute selected expert outputs
        # node_outs[b, k, :] = g[b, k] * E_{top_idx[b, k]}(x_t[b])
        # For efficiency, we batch all selected experts
        # but the indices differ per row, so we use a gather pattern.
        # First compute all K expert outputs: [B, K, H]
        all_expert_outs = torch.stack(
            [expert(x_t, h, dt=dt) for expert in self.experts],
            dim=1,
        )  # [B, K, H]
        # Gather selected: index by top_idx
        # top_idx: [B, k] -> expand to [B, k, H]
        gather_idx = top_idx.unsqueeze(-1).expand(B, top_k, self.hidden_size)
        selected_expert_outs = all_expert_outs.gather(1, gather_idx)  # [B, k, H]
        # Weight by routing gates
        weighted = g.unsqueeze(-1) * selected_expert_outs  # [B, k, H]
        # Init node_outs with base
        node_outs = weighted + (1.0 / top_k) * h_base.unsqueeze(1)  # [B, k, H]
        # DAG refine
        refined = self.dag(node_outs)  # [B, k, H]
        # Sum over K nodes
        h_lora = refined.sum(dim=1)  # [B, H]
        # New hidden = base + delta
        if self.use_residual:
            h_new = h_base + h_lora
        else:
            h_new = h_lora
        return h_new

    def forward_with_aux(self, x_t: torch.Tensor, h: torch.Tensor, dt: float = 1.0):
        """Forward that also returns the per-expert outputs (for diagnostics)."""
        B = x_t.size(0)
        h_base = self.base_cfc(x_t, h, dt=dt)
        combined = torch.cat([x_t, h], dim=-1)
        scores = self.router(combined)
        top_k = self.top_k
        top_scores, top_idx = scores.topk(top_k, dim=-1)
        g = F.softmax(top_scores, dim=-1)
        all_expert_outs = torch.stack(
            [expert(x_t, h, dt=dt) for expert in self.experts],
            dim=1,
        )
        gather_idx = top_idx.unsqueeze(-1).expand(B, top_k, self.hidden_size)
        selected_expert_outs = all_expert_outs.gather(1, gather_idx)
        weighted = g.unsqueeze(-1) * selected_expert_outs
        node_outs = weighted + (1.0 / top_k) * h_base.unsqueeze(1)
        refined = self.dag(node_outs)
        h_lora = refined.sum(dim=1)
        h_new = h_base + h_lora if self.use_residual else h_lora
        return h_new, {
            "router_probs_full": F.softmax(scores, dim=-1),
            "router_top_scores": top_scores,
            "router_top_idx": top_idx,
            "router_g": g,
            "all_expert_outs": all_expert_outs,
            "selected_expert_outs": selected_expert_outs,
            "node_outs": node_outs,
            "refined_node_outs": refined,
            "h_lora": h_lora,
            "h_base": h_base,
        }


class DAGMoECfCNetwork(nn.Module):
    """Stacked DAG-MoE CfC network.

    Each layer is a DAGMoECfCCell.  The network handles NaN inputs
    (replacing them with 0) and returns either per-step or last-step
    output.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        return_sequences: bool = True,
        n_experts: int = 3,
        top_k: int = 3,
        n_dag_iterations: int = 2,
        dag_down_dim: int = 8,
        use_residual: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in = input_size if layer_idx == 0 else hidden_size
            self.cells.append(
                DAGMoECfCCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    n_experts=n_experts,
                    top_k=top_k,
                    n_dag_iterations=n_dag_iterations,
                    dag_down_dim=dag_down_dim,
                    use_residual=use_residual,
                )
            )
        # Output head
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Forward over a sequence.

        Parameters
        ----------
        x : [B, T, I]  input sequence (may contain NaN)
        dt : float  time delta

        Returns
        -------
        out : [B, T, O] if return_sequences else [B, O]
        """
        x = torch.nan_to_num(x, nan=0.0)
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            layer_in = x_t
            for cell in self.cells:
                h = cell(layer_in, h, dt=dt)
                layer_in = h
            out_t = self.head(h)
            outputs.append(out_t)
        outputs = torch.stack(outputs, dim=1)  # [B, T, O]
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def dag_moe_utilization(cell: DAGMoECfCCell) -> dict:
    """Diagnostic for a DAG-MoE cell's expert utilization.

    Returns
    -------
    dict with keys:
        n_experts
        top_k
        n_dag_iterations
        n_params
        n_dag_params
        n_expert_params
    """
    n_total = sum(p.numel() for p in cell.parameters())
    n_dag = sum(p.numel() for p in cell.dag.parameters())
    n_experts = sum(p.numel() for p in cell.experts.parameters())
    n_base = sum(p.numel() for p in cell.base_cfc.parameters())
    n_router = sum(p.numel() for p in cell.router.parameters())
    return {
        "n_experts": cell.n_experts,
        "top_k": cell.top_k,
        "n_dag_iterations": cell.dag.n_iterations,
        "n_params": n_total,
        "n_dag_params": n_dag,
        "n_expert_params": n_experts,
        "n_base_params": n_base,
        "n_router_params": n_router,
    }
