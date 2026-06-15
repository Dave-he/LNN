"""Sigmoid Routing MoE for CfC (PRD #10-78, round 116, 2026-06-15).

Response to Qwen2-MoE (arXiv:2407.10671, June 2024) and related works that
use ``sigmoid`` instead of ``softmax`` for routing. The key insight:
softmax normalizes scores (so they sum to 1, creating a "competition"
between experts), while sigmoid lets each expert fire independently in
[0, 1]. This is the 4th major router family in the 91-115 audit (after
softmax, ReLU, cosine) and the 1st router without normalization.

Why this fits the 91-115 audit pattern:
- Structural: changes the routing topology (softmax → sigmoid), but
  the experts/CfCCell structure is unchanged.
- Data-structure-independent: no data-dependent bias, no regime detection.
- Preserves recurrent state mixing: ``h_new = sum_i g_i * expert_i(x_t, h_t)``
  has the same form as FAME/ReMoE. The inner CfCCell.forward is untouched.
- Fills a real gap: rounds 78/103 (softmax), 114 (ReLU), 82 (cosine) all use
  a normalized or zero-suppressed gate. Sigmoid is the first purely
  independent scoring gate.

Three properties of sigmoid routing:
1. **No normalization** — each expert gets an independent score in [0, 1].
   Multiple experts can fire simultaneously with no "softmax budget" competition.
2. **Naturally sparse via small init** — early in training, W ~ 0 so g ~ 0.5,
   but as W magnitudes diverge, only some experts fire strongly.
3. **Per-expert bias optional** — Qwen2-MoE uses bias similar to DeepSeek-V3
   AuxLF. We include an optional bias (initialized to 0) for future use.

Forward pass::

    g = sigmoid(W x + b)        # [B, K] each entry in [0, 1], NOT normalized
    g_sparse = topk(g, top_k)   # [B, K] with K' nonzeros (sparse mode)
    h_new = sum_i g_i * expert_i(x_t, h_t)  # [B, hidden]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.sequence_utils import select_step_delta, select_step_mask


class SigmoidRouter(nn.Module):
    """Sigmoid-based router: g = sigmoid(W x) ∈ [0, 1]^K (no normalization).

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension (H).
        n_experts: Number of experts K.
        top_k: Number of experts activated per step (0 = dense, all K).
        use_bias: If True, add a learnable per-expert bias (initialized to 0).
            This is the Qwen2-MoE variant.
        router_hidden: Optional 2-layer router MLP width (``0`` = linear).
        small_init: If True, init W ~ N(0, 0.01) to avoid early saturation.
            Default True (Qwen2-MoE recipe).

    Forward:
        x_t: [B, input_size] input.
        h:   [B, hidden_size] previous hidden.
        Returns: g: [B, n_experts] sigmoid scores (in [0, 1], not normalized).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int,
        top_k: int = 0,
        use_bias: bool = True,
        router_hidden: int = 0,
        small_init: bool = True,
    ):
        super().__init__()
        assert n_experts >= 1, f"n_experts must be >= 1, got {n_experts}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.use_bias = bool(use_bias)
        self.router_hidden = int(router_hidden)
        self.small_init = bool(small_init)

        if router_hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(input_size + hidden_size, router_hidden),
                nn.Tanh(),
                nn.Linear(router_hidden, n_experts),
            )
        else:
            self.net = nn.Linear(input_size + hidden_size, n_experts)

        if self.small_init:
            # Qwen2-MoE recipe: small init avoids early sigmoid saturation
            last_layer = self.net[-1] if router_hidden > 0 else self.net
            nn.init.normal_(last_layer.weight, mean=0.0, std=0.01)
            if last_layer.bias is not None:
                nn.init.zeros_(last_layer.bias)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(n_experts))
        else:
            self.register_parameter("bias", None)

        # Side-channel: last routing scores and top-K indices
        self.last_g: torch.Tensor | None = None
        self.last_top_idx: torch.Tensor | None = None
        self.last_logits: torch.Tensor | None = None  # raw pre-sigmoid (round 88 compat)

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sigmoid routing scores per timestep.

        Args:
            x_t: [B, input_size] input.
            h:   [B, hidden_size] previous hidden.

        Returns:
            g: [B, n_experts] sigmoid scores (sparse if top_k > 0).
        """
        combined = torch.cat([x_t, h], dim=-1)
        logits = self.net(combined)
        if self.bias is not None:
            logits = logits + self.bias
        self.last_logits = logits

        g = torch.sigmoid(logits)
        self.last_g = g.detach()

        if self.top_k > 0 and self.top_k < self.n_experts:
            # Sparse mode: zero out everything except top-K
            top_vals, top_idx = g.topk(self.top_k, dim=-1)
            mask = torch.full_like(g, 0.0)
            mask.scatter_(-1, top_idx, 1.0)
            g = g * mask
            self.last_top_idx = top_idx.detach()
        else:
            # Dense mode: all K experts fire
            # Bump top_idx to be a placeholder for compat with diagnostics
            self.last_top_idx = torch.arange(
                self.n_experts, device=g.device,
            ).expand(g.size(0), -1).contiguous()

        return g


class SigmoidMoECfCCell(nn.Module):
    """Sigmoid-Routed MoE cell: K experts, dense or sparse top-K.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension (H).
        n_experts: Number of experts K.
        top_k: Number of experts activated per step (0 = dense).
        n_tau_per_expert: Per-expert ``n_tau``.
        tau_scales: Per-branch τ init, forwarded to every expert.
        use_router_bias: If True, router has a learnable per-expert bias.
        router_hidden: Router MLP width (``0`` = linear).
        small_init: If True, init router W ~ N(0, 0.01).

    Notes:
        - The hidden state h_t is preserved across steps (no modification).
        - The combination is ``h_new = sum_i g_i * expert_i(x_t, h_t)``,
          the same form as FAME/ReMoE.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 3,
        top_k: int = 0,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        use_router_bias: bool = True,
        router_hidden: int = 0,
        small_init: bool = True,
    ):
        super().__init__()
        assert n_experts >= 1
        assert top_k >= 0, f"top_k must be >= 0 (0 = dense), got {top_k}"
        if top_k > 0:
            assert top_k <= n_experts, f"top_k must be <= n_experts, got {top_k}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.use_router_bias = bool(use_router_bias)
        self.small_init = bool(small_init)

        # K experts, each a CfC cell.
        self.experts = nn.ModuleList(
            [
                CfCCell(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    n_tau=n_tau_per_expert,
                    tau_scales=tau_scales,
                )
                for _ in range(self.n_experts)
            ]
        )
        self.router = SigmoidRouter(
            input_size=input_size,
            hidden_size=hidden_size,
            n_experts=self.n_experts,
            top_k=self.top_k,
            use_bias=self.use_router_bias,
            router_hidden=router_hidden,
            small_init=small_init,
        )

        # Diagnostic stash.
        self.last_expert_util: torch.Tensor | None = None

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One step of sigmoid-routed MoE.

        Args:
            x_t: [B, input_size] input at this step.
            h:   [B, hidden_size] previous hidden state.
            dt:  scalar or [B] per-sample time delta.

        Returns:
            h_new: [B, hidden_size] mixed expert output.
        """
        B = x_t.size(0)

        # 1) Compute sigmoid routing scores.
        g = self.router(x_t, h)  # [B, K]

        # 2) Run all K experts.
        outs = [expert(x_t, h, dt=dt) for expert in self.experts]  # K × [B, H]
        stacked = torch.stack(outs, dim=1)  # [B, K, H]

        # 3) Weighted combination (same form as FAME/ReMoE).
        h_new = (g.unsqueeze(-1) * stacked).sum(dim=1)  # [B, H]

        # 4) Track expert utilization.
        if g.dim() == 2 and g.size(0) == B:
            self.last_expert_util = g.mean(dim=0).detach()  # [K] mean per-expert gate

        return h_new

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Same as forward but also returns per-expert outputs (for diagnostics)."""
        g = self.router(x_t, h)
        outs = [expert(x_t, h, dt=dt) for expert in self.experts]
        stacked = torch.stack(outs, dim=1)
        h_new = (g.unsqueeze(-1) * stacked).sum(dim=1)
        if g.dim() == 2 and g.size(0) == x_t.size(0):
            self.last_expert_util = g.mean(dim=0).detach()
        return h_new, outs


class SigmoidMoECfCNetwork(nn.Module):
    """Stacked Sigmoid-Routed MoE CfC network.

    Mirrors the ``CfCNetwork`` / ``FAMECfCNetwork`` API (return_sequences,
    mask, dt) but uses ``SigmoidMoECfCCell`` for every layer.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension.
        output_size: Output dimension.
        num_layers: Number of stacked sigmoid-routed MoE cells.
        return_sequences: If True, return the full sequence; else last step.
        n_experts: Number of experts K per layer.
        top_k: Number of experts activated per step (0 = dense).
        n_tau_per_expert: Per-expert ``n_tau``.
        tau_scales: Per-branch τ init, forwarded to every expert.
        use_router_bias: Forward to every layer's router.
        router_hidden: Router MLP width.
        small_init: Forward to every layer's router.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
        n_experts: int = 3,
        top_k: int = 0,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        use_router_bias: bool = True,
        router_hidden: int = 0,
        small_init: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.use_router_bias = bool(use_router_bias)
        self.router_hidden = int(router_hidden)
        self.small_init = bool(small_init)

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(
                SigmoidMoECfCCell(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    n_experts=self.n_experts,
                    top_k=self.top_k,
                    n_tau_per_expert=self.n_tau_per_expert,
                    tau_scales=self.tau_scales,
                    use_router_bias=self.use_router_bias,
                    router_hidden=self.router_hidden,
                    small_init=self.small_init,
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


def sigmoid_moe_utilization(cell: SigmoidMoECfCCell) -> dict:
    """Diagnostic for Sigmoid MoE cell's expert utilization.

    Returns:
        Dict with:
            - "expert_util": [K] tensor of mean sigmoid gate per expert.
            - "expert_count": [K] rough count (util * B for batch B).
            - "routing_entropy": scalar — entropy of expert_util (in nats).
            - "sparsity_mode": "dense" or f"top_{cell.top_k}".
    """
    if cell.last_expert_util is None:
        return {
            "expert_util": torch.zeros(cell.n_experts),
            "expert_count": torch.zeros(cell.n_experts),
            "routing_entropy": torch.tensor(0.0),
            "sparsity_mode": f"top_{cell.top_k}" if cell.top_k > 0 else "dense",
        }
    util = cell.last_expert_util.cpu()
    eps = 1e-8
    # Normalize to a probability distribution for entropy.
    util_p = util / (util.sum() + eps)
    entropy = -(util_p * torch.log(util_p + eps)).sum().cpu()
    return {
        "expert_util": util,
        "expert_count": (util * 100).long(),  # rough count
        "routing_entropy": entropy,
        "sparsity_mode": f"top_{cell.top_k}" if cell.top_k > 0 else "dense",
    }
