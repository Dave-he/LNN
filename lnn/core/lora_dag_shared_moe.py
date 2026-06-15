"""Round 124 — LoRA-DAG-Shared-MoE (TRIPLE hybrid) for CfC.

Combines three orthogonal winners from the 91-123 audit:
- **LoRA-MoRE (round 118)**: low-rank expert deltas (rank r) on a shared
  base CfC, B-init-zero warm start.
- **DAG-MoE (round 120)**: directed acyclic graph aggregation over K
  selected experts with L iterations of learned edge gates.
- **DeepSeek shared (round 113)**: a single always-on shared expert that
  contributes additively to the routed path (DeepSeekMoE arXiv:2401.06066).

Round 122 (ProbLoRA = routing × expert) failed because those dimensions
are coupled in 1D. Round 123 (LoRA × DAG = expert × aggregation)
succeeded because they're orthogonal — 1 NEW BEST on structured_irr.
This round (LoRA × DAG × Shared) tests whether the orthogonal
mechanism stack can grow to 3 dimensions.

Forward pass (L iterations of DAG, K routed + 1 shared expert):
  h_base = base_cfc(x_t, h, dt)              # shared base
  combined = [x_t; h]                        # [B, I+H]
  h_shared = (alpha/r) * B_shared(combined @ A_shared)   # always-on
  all_routed = stack([(alpha/r) * B_i(combined @ A_i) for i in K])
  g, top_idx = router(x_t, h)                # top-K sparse
  selected = gather(all_routed, top_idx)     # [B, k, H]
  top_g = gather(g, top_idx)                 # [B, k]
  node_outs = top_g * selected + (1/k) * h_base
  refined = dag(node_outs)                   # L iterations
  h_routed = sum_i refined[:, i, :]           # [B, H]
  h_new = h_base + h_shared + h_routed

Notes
-----
* The shared LoRA adapter uses B-init-zero like the routed experts, so at
  init the model is identical to the base CfC + DAG (warm start).
* The shared contribution is added after the routed DAG (DeepSeek pattern).
* All three mechanisms are orthogonal:
  - expert family: low-rank LoRA (vs sub-CfC)
  - aggregation: DAG (vs weighted sum)
  - shared pathway: always-on additive (vs only-routed)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell
from lnn.core.dag_moe import DAGEdgeGate
from lnn.core.lora_moe import LoRAExpert
from lnn.core.sequence_utils import select_step_delta


class LoRADAGSharedAggregation(nn.Module):
    """L iterations of DAG refinement over K LoRA-delta nodes (round 124)."""

    def __init__(self, hidden_size: int, n_nodes: int, n_iterations: int = 2, down_dim: int = 8):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_iterations = n_iterations
        self.layers = nn.ModuleList(
            [DAGEdgeGate(hidden_size, down_dim) for _ in range(n_iterations)]
        )

    def forward(self, node_outs: torch.Tensor) -> torch.Tensor:
        h = node_outs
        for layer in self.layers:
            h = layer(h)
        return h


class LoRADAGSharedMoECfCCell(nn.Module):
    """LoRA-DAG-Shared-MoE CfC cell: shared + routed LoRA with DAG aggregation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 3,           # K routed experts
        top_k: int = 3,               # top-K sparse (default: dense)
        rank: int = 4,
        alpha: float = 1.0,
        n_dag_iterations: int = 2,
        dag_down_dim: int = 8,
        router_type: str = "learned",
        router_hidden: int = 0,
        n_tau_base: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        lora_dropout: float = 0.0,
        small_init: bool = True,
        use_residual: bool = True,
        use_shared: bool = True,
        n_shared: int = 1,           # K_s shared experts (DeepSeek pattern)
    ):
        super().__init__()
        if top_k > n_experts:
            raise ValueError(f"top_k={top_k} cannot exceed n_experts={n_experts}")
        assert rank >= 1, f"rank must be >= 1, got {rank}"
        assert router_type in ("learned", "sigmoid"), (
            f"router_type must be 'learned' or 'sigmoid', got {router_type!r}"
        )
        if router_type == "learned":
            assert top_k >= 1, "learned router requires top_k >= 1"
        if use_shared:
            assert n_shared >= 1, f"n_shared must be >= 1 when use_shared=True, got {n_shared}"
        else:
            n_shared = 0
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.router_type = str(router_type)
        self.use_residual = bool(use_residual)
        self.use_shared = bool(use_shared)
        self.n_shared = int(n_shared)
        self.adapter_dim = self.input_size + self.hidden_size

        # Shared base CfC
        self.base_cfc = CfCCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            n_tau=n_tau_base,
            tau_scales=tau_scales,
        )

        # Always-on shared LoRA experts (DeepSeek pattern, K_s = n_shared)
        if use_shared:
            self.shared_experts = nn.ModuleList(
                [
                    LoRAExpert(
                        in_features=self.adapter_dim,
                        out_features=self.hidden_size,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=lora_dropout,
                        small_init=small_init,
                    )
                    for _ in range(self.n_shared)
                ]
            )
        else:
            self.shared_experts = nn.ModuleList()

        # K routed LoRA experts (B-init-zero warm start)
        self.experts = nn.ModuleList(
            [
                LoRAExpert(
                    in_features=self.adapter_dim,
                    out_features=self.hidden_size,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=lora_dropout,
                    small_init=small_init,
                )
                for _ in range(self.n_experts)
            ]
        )

        # Router
        if router_type == "learned":
            from lnn.core.forecastability_router import ForecastabilityRouter
            self.router = ForecastabilityRouter(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                n_experts=self.n_experts,
                top_k=self.top_k,
                router_hidden=router_hidden,
            )
        else:  # "sigmoid" — supports dense (top_k=0)
            from lnn.core.sigmoid_moe import SigmoidRouter
            self.router = SigmoidRouter(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                n_experts=self.n_experts,
                top_k=self.top_k,
                use_bias=True,
                router_hidden=router_hidden,
                small_init=True,
            )

        # DAG aggregation
        self.dag = LoRADAGSharedAggregation(
            hidden_size=self.hidden_size,
            n_nodes=self.top_k,
            n_iterations=n_dag_iterations,
            down_dim=dag_down_dim,
        )

        # Side-channels
        self.last_g = None
        self.last_top_idx = None
        self.last_expert_util = None
        self.last_shared_delta = None

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        h_new, _ = self.forward_with_aux(x_t, h, dt=dt)
        return h_new

    def forward_with_aux(self, x_t: torch.Tensor, h: torch.Tensor, dt: float | torch.Tensor = 1.0):
        B = x_t.size(0)
        h_base = self.base_cfc(x_t, h, dt=dt)  # [B, H]

        combined = torch.cat([x_t, h], dim=-1)  # [B, I+H]

        # 1) Shared pathway: always-on LoRA (K_s experts, mean-aggregated)
        if len(self.shared_experts) > 0:
            shared_outs = [expert(combined) for expert in self.shared_experts]
            h_shared = torch.stack(shared_outs, dim=1).mean(dim=1)  # [B, H]
        else:
            h_shared = torch.zeros(B, self.hidden_size, device=x_t.device, dtype=x_t.dtype)

        # 2) Routed LoRA deltas
        all_deltas = torch.stack(
            [expert(combined) for expert in self.experts],
            dim=1,
        )  # [B, K, H]

        # 3) Router mixing
        if self.router_type == "learned":
            g = self.router(x_t, h)  # [B, n_experts] sparse
            top_idx = self.router.last_top_idx  # [B, top_k]
            top_k = self.top_k
            g_idx = top_idx.unsqueeze(-1).expand(B, top_k, self.n_experts)
            g_full = g.unsqueeze(1).expand(B, self.n_experts, self.n_experts)
            top_g = g_full.gather(1, g_idx)[:, :, 0]  # [B, top_k]
        else:
            g = self.router(x_t, h)  # [B, K]
            if self.top_k > 0:
                top_scores, top_idx = g.topk(self.top_k, dim=-1)  # [B, k]
                top_g = top_scores
            else:
                top_idx = torch.arange(self.n_experts, device=g.device).unsqueeze(0).expand(B, -1)
                top_g = g
            top_k = self.top_k if self.top_k > 0 else self.n_experts

        # 4) Gather selected deltas + DAG refine
        gather_idx = top_idx.unsqueeze(-1).expand(B, top_k, self.hidden_size)
        selected = all_deltas.gather(1, gather_idx)  # [B, k, H]
        weighted = top_g.unsqueeze(-1) * selected  # [B, k, H]
        node_outs = weighted + (1.0 / top_k) * h_base.unsqueeze(1)  # [B, k, H]
        refined = self.dag(node_outs)  # [B, k, H]
        h_routed = refined.sum(dim=1)  # [B, H]

        # 5) Additive combination: base + shared + routed (DeepSeek pattern)
        h_new = h_base + h_shared + h_routed if self.use_residual else h_shared + h_routed

        # Side-channels
        self.last_g = top_g.detach()
        self.last_top_idx = top_idx.detach() if top_idx is not None else None
        if top_g.dim() == 2 and top_g.size(0) == B:
            self.last_expert_util = top_g.mean(dim=0).detach()
        self.last_shared_delta = h_shared.detach() if len(self.shared_experts) > 0 else None

        return h_new, {
            "all_deltas": all_deltas,
            "selected_deltas": selected,
            "g": top_g,
            "top_idx": top_idx,
            "weighted": weighted,
            "node_outs": node_outs,
            "refined": refined,
            "h_routed": h_routed,
            "h_shared": h_shared,
            "h_base": h_base,
        }


class LoRADAGSharedMoECfCNetwork(nn.Module):
    """Stacked LoRA-DAG-Shared-MoE CfC network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        return_sequences: bool = True,
        n_experts: int = 3,
        top_k: int = 3,
        rank: int = 4,
        alpha: float = 1.0,
        n_dag_iterations: int = 2,
        dag_down_dim: int = 8,
        router_type: str = "learned",
        router_hidden: int = 0,
        n_tau_base: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        lora_dropout: float = 0.0,
        small_init: bool = True,
        use_residual: bool = True,
        use_shared: bool = True,
        n_shared: int = 1,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.return_sequences = bool(return_sequences)
        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in = input_size if layer_idx == 0 else hidden_size
            self.cells.append(
                LoRADAGSharedMoECfCCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    n_experts=n_experts,
                    top_k=top_k,
                    rank=rank,
                    alpha=alpha,
                    n_dag_iterations=n_dag_iterations,
                    dag_down_dim=dag_down_dim,
                    router_type=router_type,
                    router_hidden=router_hidden,
                    n_tau_base=n_tau_base,
                    tau_scales=tau_scales,
                    lora_dropout=lora_dropout,
                    small_init=small_init,
                    use_residual=use_residual,
                    use_shared=use_shared,
                    n_shared=n_shared,
                )
            )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0)
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        layer_input = x
        for cell in self.cells:
            outputs = []
            h_i = h
            for t in range(T):
                dt_t = select_step_delta(dt, t, B, T, x.device, x.dtype)
                x_t = layer_input[:, t, :]
                h_i = cell(x_t, h_i, dt=dt_t)
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = h_i
        out = self.head(layer_input)
        if self.return_sequences:
            return out
        return out[:, -1, :]


def lora_dag_shared_moe_utilization(cell: LoRADAGSharedMoECfCCell) -> dict:
    """Diagnostic for LoRA-DAG-Shared-MoE cell's expert utilization."""
    n_total = sum(p.numel() for p in cell.parameters())
    n_dag = sum(p.numel() for p in cell.dag.parameters())
    n_lora_routed = sum(p.numel() for e in cell.experts for p in e.parameters())
    n_lora_shared = sum(
        p.numel() for e in cell.shared_experts for p in e.parameters()
    )
    n_base = sum(p.numel() for p in cell.base_cfc.parameters())
    n_router = sum(p.numel() for p in cell.router.parameters())
    out = {
        "n_experts": cell.n_experts,
        "top_k": cell.top_k,
        "rank": cell.rank,
        "alpha": cell.alpha,
        "scaling": cell.alpha / cell.rank,
        "n_dag_iterations": cell.dag.n_iterations,
        "use_shared": cell.use_shared,
        "n_shared": cell.n_shared,
        "n_params": n_total,
        "n_dag_params": n_dag,
        "n_lora_routed_params": n_lora_routed,
        "n_lora_shared_params": n_lora_shared,
        "n_base_params": n_base,
        "n_router_params": n_router,
        "routing_entropy": 0.0,
        "expert_util": [],
    }
    if cell.last_expert_util is not None:
        util = cell.last_expert_util.detach()
        p = util / (util.sum() + 1e-12)
        entropy = -(p * (p + 1e-12).log()).sum().item()
        out["routing_entropy"] = float(entropy)
        out["expert_util"] = util.tolist()
    return out
