"""LoRA Mixture-of-Experts MoE for CfC (PRD #10-80, round 118, 2026-06-15).

Response to arXiv:2505.22694 — *MoRE: A Mixture of Low-Rank Experts for
Adaptive Multi-Task Learning* (Zhang et al., ACL 2025 Findings, May 2025),
and inspired by the broader LoRA literature (arXiv:2106.09685).

Key idea (from MoRE/LoRA):
- A single **base** network is shared across all K experts.
- Each expert is a **rank-r low-rank adapter** applied additively to the
  base output: ``ΔW_k = (alpha/r) · B_k · A_k`` with ``A_k ∈ R^{d×r}``,
  ``B_k ∈ R^{r×d}``.
- The router (a sparse top-K) picks which K' of the K adapters to apply.
- Parameter cost of the adapter is ``O(K · r · (I+H))`` vs ``O(K · d^2)``
  for a full dense expert — a real saving when ``r << d``.

In this CfC setting:
- The base is a single ``CfCCell(input_size, hidden_size)``.
- Each adapter maps ``[x_t; h] → R^hidden_size`` via two low-rank Linear
  layers (no activation between them, matching LoRA's standard
  configuration).
- Initialise ``B = 0`` so the model starts identical to the base CfC
  (the canonical LoRA warm-start).
- Standard LoRA scaling: ``alpha / r``.

Why this fits the 91-117 audit pattern:
- Structural: changes the **expert family** (dense CfC expert → low-rank
  LoRA adapter), not the router. The router is the FAME top-K.
- Data-structure-independent: rank is a free integer hyperparameter.
- Preserves recurrent state mixing: ``h_new = base(x, h) + Σ g_i · LoRA_i(x, h)``
  has the same form as FAME/ReMoE/sigmoid. The base CfCCell.forward is
  untouched.
- Fills the "low-rank expert" gap in the audit.

Forward pass::

    h_base = base_cfc(x_t, h)              # [B, H]
    combined = [x_t; h]                    # [B, I+H]
    Δ_i = (alpha/r) · (combined @ A_i) @ B_i  # [B, H], K such deltas
    g = router(x_t, h)                     # [B, K] top-K
    h_new = h_base + Σ_i g_i · Δ_i         # [B, H]
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.forecastability_router import ForecastabilityRouter
from lnn.core.sequence_utils import select_step_delta, select_step_mask


class LoRAExpert(nn.Module):
    """Single low-rank adapter: ``Δ = (alpha/r) · (x @ A^T) @ B^T``.

    Args:
        in_features: Input dimension (typically ``input_size + hidden_size``).
        out_features: Output dimension (typically ``hidden_size``).
        rank: LoRA rank r (1, 2, 4, ...).  Must be >= 1.
        alpha: LoRA scaling alpha.  Effective scale = alpha / rank.
        dropout: Optional dropout on the intermediate hidden (B @ A applied
            to x).  Default 0.0.
        small_init: If True, init A with kaiming_uniform and B with zeros
            (canonical LoRA warm-start).  Default True.

    Forward shape:
        x: [B, in_features] → Δ: [B, out_features]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        small_init: bool = True,
    ):
        super().__init__()
        assert rank >= 1, f"rank must be >= 1, got {rank}"
        assert alpha > 0, f"alpha must be > 0, got {alpha}"
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = float(dropout)
        self.small_init = bool(small_init)

        # LoRA matrices (A is in_features × rank, B is rank × out_features).
        # Note: nn.Parameter, not Linear, to keep the math obvious and
        # avoid an extra matmul call against an internal identity.
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.rank))
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.out_features))
        if self.dropout > 0.0:
            self._drop = nn.Dropout(p=self.dropout)
        else:
            self._drop = None

        self._init_params()

    def _init_params(self):
        if self.small_init:
            # kaiming_uniform for A, zeros for B (canonical LoRA).
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Δ = (alpha / r) · (x @ A) @ B   →  [B, in] → [B, r] → [B, out]
        # We store A as (in, r) and B as (r, out); matmul directly.
        z = x @ self.lora_A                       # [B, rank]
        if self._drop is not None:
            z = self._drop(z)
        delta = z @ self.lora_B                   # [B, out]
        return delta * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"alpha={self.alpha}, scaling={self.scaling:.4f}"
        )


class LoRACfCCell(nn.Module):
    """MoRE-style cell: shared base CfC + K low-rank LoRA adapters.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension (H).
        n_experts: Number of LoRA experts K.
        top_k: Number of experts activated per step (0 = dense).
        rank: LoRA rank r, applied to every adapter.
        alpha: LoRA scaling alpha (effective scale = alpha / rank).
        router_hidden: Router MLP width (0 = linear).
        n_tau_base: ``n_tau`` for the shared base CfC cell.  Default 1.
        tau_scales: Per-branch initial τ for the base CfC.
        router_type: ``"learned"`` (default) uses
            ``ForecastabilityRouter``; ``"sigmoid"`` reuses the round 116
            ``SigmoidRouter`` (so we can ablate router x expert family in
            isolation).  ``"cosine"`` is also available for completeness.
        lora_dropout: Optional dropout inside each LoRA adapter.  Default 0.
        small_init: If True, init B=0 (warm-start).  Default True.

    Notes:
        - The hidden state h_t is preserved across steps (no modification).
        - The combination is ``h_new = base(x_t, h_t) + Σ g_i · LoRA_i([x_t, h_t])``,
          additive over the base.  Equivalent in form to FAME (which is
          instead ``Σ g_i · expert_i(...)``); we replace the dense expert
          with a rank-r delta.
        - When the router is in dense mode (top_k=0), all K adapters
          contribute; when sparse, only the top-K.

    Compatibility:
        - ``forward(x_t, h, dt)`` matches the FAME / sigmoid / gumbel cell
          interface.
        - ``forward_with_aux(x_t, h, dt)`` returns ``(h_new, deltas)`` where
          ``deltas`` is a list of K [B, H] tensors — useful for downstream
          diagnostics (e.g. weight-orth loss, expert diversity).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 3,
        top_k: int = 2,
        rank: int = 4,
        alpha: float = 1.0,
        router_hidden: int = 0,
        n_tau_base: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        router_type: str = "learned",
        lora_dropout: float = 0.0,
        small_init: bool = True,
    ):
        super().__init__()
        assert n_experts >= 1, f"n_experts must be >= 1, got {n_experts}"
        assert rank >= 1, f"rank must be >= 1, got {rank}"
        assert top_k >= 0, f"top_k must be >= 0 (0 = dense), got {top_k}"
        if top_k > 0:
            assert top_k <= n_experts, (
                f"top_k must be <= n_experts, got {top_k} vs {n_experts}"
            )
        assert router_type in ("learned", "sigmoid", "cosine"), (
            f"router_type must be 'learned'/'sigmoid'/'cosine', got {router_type!r}"
        )
        if router_type in ("learned", "cosine"):
            assert top_k >= 1, (
                f"router_type={router_type!r} requires top_k >= 1 (no dense mode), "
                f"got top_k={top_k}.  Use router_type='sigmoid' for dense."
            )
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.router_hidden = int(router_hidden)
        self.n_tau_base = int(n_tau_base)
        self.tau_scales = tuple(tau_scales)
        self.router_type = str(router_type)
        self.small_init = bool(small_init)
        self.adapter_dim = self.input_size + self.hidden_size

        # Shared base CfC cell (the "frozen" part in the LoRA analogy;
        # here it is trainable, but the experts are deltas over it).
        self.base_cfc = CfCCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            n_tau=self.n_tau_base,
            tau_scales=self.tau_scales,
        )

        # K LoRA adapters.  Each maps [B, I+H] → [B, H].
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

        # Router (FAME learned softmax by default; sigmoid/cosine ablations
        # are exposed for the bench).
        if router_type == "learned":
            self.router = ForecastabilityRouter(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                n_experts=self.n_experts,
                top_k=self.top_k,
                router_hidden=self.router_hidden,
            )
        elif router_type == "sigmoid":
            from lnn.core.sigmoid_moe import SigmoidRouter
            self.router = SigmoidRouter(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                n_experts=self.n_experts,
                top_k=self.top_k,
                use_bias=True,
                router_hidden=self.router_hidden,
                small_init=True,
            )
        else:  # "cosine"
            from lnn.core.cosine_router import CosineRouter
            self.router = CosineRouter(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                n_experts=self.n_experts,
                top_k=self.top_k,
            )

        # Diagnostic stash (consumed by lora_moe_utilization).
        self.last_expert_util: torch.Tensor | None = None
        self.last_g: torch.Tensor | None = None
        self.last_top_idx: torch.Tensor | None = None

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One step of LoRA-MoE.

        Args:
            x_t: [B, input_size] input at this step.
            h:   [B, hidden_size] previous hidden state.
            dt:  scalar or [B] per-sample time delta.

        Returns:
            h_new: [B, hidden_size] base + Σ g_i · LoRA_i([x_t, h]).
        """
        h_new, _ = self.forward_with_aux(x_t, h, dt=dt)
        return h_new

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Like ``forward`` but also returns the K LoRA deltas (each [B, H]).

        Useful for the round-90 / round-97 orthogonality losses and the
        round-95 / round-99 per-expert diversity diagnostics.
        """
        B = x_t.size(0)
        # 1) Base CfC output (the "frozen" part of LoRA).
        h_base = self.base_cfc(x_t, h, dt=dt)  # [B, H]

        # 2) K LoRA deltas over the concatenated input.
        combined = torch.cat([x_t, h], dim=-1)  # [B, I+H]
        deltas = [expert(combined) for expert in self.experts]  # K × [B, H]
        stacked = torch.stack(deltas, dim=1)  # [B, K, H]

        # 3) Router mixing.
        g = self.router(x_t, h)  # [B, K] (sparse or dense)
        h_new = h_base + (g.unsqueeze(-1) * stacked).sum(dim=1)  # [B, H]

        # 4) Side-channels.
        self.last_g = g.detach()
        # Pull last_top_idx from router (ForecastabilityRouter/Sigmoid
        # both expose it; cosine does too).  Fallback to argmax if not.
        if hasattr(self.router, "last_top_idx") and self.router.last_top_idx is not None:
            self.last_top_idx = self.router.last_top_idx.detach()
        if g.dim() == 2 and g.size(0) == B:
            self.last_expert_util = g.mean(dim=0).detach()  # [K]

        return h_new, deltas


class LoRACfCNetwork(nn.Module):
    """Stacked LoRA-MoE CfC network.

    Mirrors the ``CfCNetwork`` / ``FAMECfCNetwork`` API (return_sequences,
    mask, dt) but uses ``LoRACfCCell`` for every layer.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension.
        output_size: Output dimension.
        num_layers: Number of stacked LoRA-MoE cells.
        return_sequences: If True, return the full sequence; else last step.
        n_experts: Number of LoRA experts K per layer.
        top_k: Number of experts activated per step (0 = dense).
        rank: LoRA rank r.
        alpha: LoRA scaling alpha (effective = alpha / rank).
        router_hidden: Router MLP width (0 = linear).
        n_tau_base: ``n_tau`` for the base CfC.
        tau_scales: Per-branch initial τ.
        router_type: ``"learned"`` / ``"sigmoid"`` / ``"cosine"``.
        lora_dropout: Optional LoRA dropout.
        small_init: If True, init B=0.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
        n_experts: int = 3,
        top_k: int = 2,
        rank: int = 4,
        alpha: float = 1.0,
        router_hidden: int = 0,
        n_tau_base: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        router_type: str = "learned",
        lora_dropout: float = 0.0,
        small_init: bool = True,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.return_sequences = bool(return_sequences)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.router_hidden = int(router_hidden)
        self.n_tau_base = int(n_tau_base)
        self.tau_scales = tuple(tau_scales)
        self.router_type = str(router_type)
        self.small_init = bool(small_init)

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(
                LoRACfCCell(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    n_experts=self.n_experts,
                    top_k=self.top_k,
                    rank=self.rank,
                    alpha=self.alpha,
                    router_hidden=self.router_hidden,
                    n_tau_base=self.n_tau_base,
                    tau_scales=self.tau_scales,
                    router_type=self.router_type,
                    lora_dropout=lora_dropout,
                    small_init=small_init,
                )
            )
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process a batch of sequences.

        Args:
            x: [B, T, F] input sequence.
            h0: Optional [num_layers, B, H] initial hidden state.
            dt: Same per-step time-delta shapes as ``CfCNetwork``.
            mask: Same mask shapes as ``CfCNetwork``.

        Returns:
            [B, T, output_size] (return_sequences=True) or [B, output_size].
        """
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype,
            )

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            for t in range(seq_len):
                dt_t = select_step_delta(dt, t, batch_size, seq_len, x.device, x.dtype)
                input_mask, update_mask = select_step_mask(
                    mask, t, batch_size, seq_len, self.input_size, x.device, x.dtype,
                )
                x_t = torch.nan_to_num(layer_input[:, t, :])
                if i == 0 and input_mask is not None:
                    x_t = x_t * input_mask
                h_candidate = cell(x_t, h_i, dt=dt_t)
                h_i = h_candidate if update_mask is None else update_mask * h_candidate + (1.0 - update_mask) * h_i
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat(
                [h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)],
                dim=0,
            )

        if self.return_sequences:
            return self.output_proj(layer_input)
        return self.output_proj(layer_input[:, -1, :])

    def forward_with_aux(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[list[list[torch.Tensor]]]]:
        """Like ``forward`` but returns per-step, per-layer, per-expert LoRA deltas.

        Returns:
            (y_pred, expert_deltas):
                y_pred: same shape as ``forward`` would return.
                expert_deltas: nested list ``[num_layers][T][K]`` of
                    ``[B, hidden_size]`` tensors (the LoRA deltas; useful
                    for round-90/95/97 diagnostics).
        """
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype,
            )

        h = h0
        layer_input = x
        expert_deltas: list[list[list[torch.Tensor]]] = [[] for _ in range(self.num_layers)]
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            for t in range(seq_len):
                dt_t = select_step_delta(dt, t, batch_size, seq_len, x.device, x.dtype)
                input_mask, update_mask = select_step_mask(
                    mask, t, batch_size, seq_len, self.input_size, x.device, x.dtype,
                )
                x_t = torch.nan_to_num(layer_input[:, t, :])
                if i == 0 and input_mask is not None:
                    x_t = x_t * input_mask
                h_candidate, deltas_t = cell.forward_with_aux(x_t, h_i, dt=dt_t)
                expert_deltas[i].append(deltas_t)
                h_i = h_candidate if update_mask is None else update_mask * h_candidate + (1.0 - update_mask) * h_i
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat(
                [h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)],
                dim=0,
            )

        if self.return_sequences:
            y_pred = self.output_proj(layer_input)
        else:
            y_pred = self.output_proj(layer_input[:, -1, :])
        return y_pred, expert_deltas


def lora_moe_utilization(cell: LoRACfCCell) -> dict:
    """Diagnostic for LoRA MoE cell's expert utilization.

    Returns:
        Dict with:
            - "expert_util": [K] tensor of mean per-expert gate.
            - "expert_count": [K] rough count (util * B for batch B).
            - "routing_entropy": scalar — entropy of expert_util (in nats).
            - "sparsity_mode": "dense" or f"top_{cell.top_k}".
            - "rank": int — LoRA rank.
            - "alpha": float — LoRA alpha.
            - "scaling": float — effective LoRA scaling (alpha / rank).
            - "n_lora_params": int — total LoRA parameter count.
            - "router_type": str.
    """
    if cell.last_expert_util is None:
        return {
            "expert_util": [],
            "expert_count": [],
            "routing_entropy": 0.0,
            "sparsity_mode": "dense" if cell.top_k == 0 else f"top_{cell.top_k}",
            "rank": cell.rank,
            "alpha": cell.alpha,
            "scaling": cell.alpha / cell.rank,
            "n_lora_params": sum(
                p.numel() for p in cell.parameters() if p.dim() == 2
                and (p.shape[0] == cell.adapter_dim or p.shape[1] == cell.adapter_dim or p.shape[0] == cell.rank)
            ),
            "router_type": cell.router_type,
        }

    util = cell.last_expert_util.detach()
    # Routing entropy (in nats).
    p = util / (util.sum() + 1e-12)
    entropy = -(p * (p + 1e-12).log()).sum().item()
    n_lora = sum(
        e.lora_A.numel() + e.lora_B.numel() for e in cell.experts
    )
    return {
        "expert_util": util,
        "expert_count": (util * util.size(0)).tolist(),
        "routing_entropy": float(entropy),
        "sparsity_mode": "dense" if cell.top_k == 0 else f"top_{cell.top_k}",
        "rank": cell.rank,
        "alpha": cell.alpha,
        "scaling": cell.alpha / cell.rank,
        "n_lora_params": n_lora,
        "router_type": cell.router_type,
    }
