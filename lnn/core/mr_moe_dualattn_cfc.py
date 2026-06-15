"""MR-MoE w/ Dual Attention: Multi-Rate + feature-level + temporal attention
for CfC (PRD #10-92, 2026-06-15).

Implements the multi-rate + dual-attention pattern from
arXiv:2606.12240 (Zong, Boker, Eldardiry, NeurIPS 2026 submission).

This is an **extension** of the round 77 ``MRMoECfCCell`` (see
``mr_moe_cfc.py``) which only had K experts + softmax router.  The
new axis here is **dual attention**:

- **Feature-level attention** (paper §3.1): per-step input gate
  ``α_t ∈ [0,1]^D`` from a small MLP over ``[x_t; h_prev]``. Applied
  as ``x_t' = α_t ⊙ x_t``. Suppresses noisy input variables.
- **Temporal attention** (paper §3.2): softmax over a small window of
  past hidden states (default window=4) to focus on the most
  informative historical state.

Key design choices vs round 77 MRMoECfCCell:
- K=3 CfC experts, each with a **distinct τ_init** (0.1, 1.0, 10.0)
  — fast / medium / slow.  The original MRMoECfCCell shared τ across
  experts.
- Dual attention is enabled by default; can be turned off via
  ``use_dual_attention=False`` (ablation knob).
- Reuses ``ForecastabilityRouter`` from round 78 (top-K sparse softmax).
- Reuses ``CfCCell(n_tau=1, tau_scales=...)`` from round 76.

Why this should work in our 1D setting:
- Multi-rate experts give the cell an explicit inductive bias for
  **regime-switch data** (structured_irr) — one expert locks onto
  slow drift, another on the fast transient.
- Feature-level attention acts as a **per-step input denoiser** that
  does NOT touch the recurrent W_h·h force (preserves the
  nonlinearity property shared by all 12 STRICTLY POSITIVE winners
  in the 91-129 audit).
- Temporal attention is essentially a **learned skip connection** that
  helps on long-context regime switches.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.forecastability_router import ForecastabilityRouter
from lnn.core.sequence_utils import select_step_delta, select_step_mask


class MRMoEDualAttnCfCCell(nn.Module):
    """MR-MoE + Dual Attention CfC cell (PRD #10-92).

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        n_experts: Number of experts (K). Default 3 (fast/medium/slow).
        top_k: Number of experts activated per step.
        tau_inits: Per-expert initial time constant, length K.
            Default ``(0.1, 1.0, 10.0)``.
        temporal_window: Number of past hidden states for temporal
            attention.  Default 4.
        router_hidden: Width of the optional 2-layer router MLP.
        feat_attn_hidden: Width of the feature-level attention MLP.
            ``0`` = single linear projection.
        use_dual_attention: If True, apply both feature and temporal
            attention.  If False, plain multi-rate routing only.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 3,
        top_k: int = 2,
        tau_inits: tuple = (0.1, 1.0, 10.0),
        temporal_window: int = 4,
        router_hidden: int = 0,
        feat_attn_hidden: int = 0,
        use_dual_attention: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = n_experts
        self.top_k = top_k
        # Pad / truncate tau_inits.
        sc = list(tau_inits)[:n_experts]
        if not sc:
            sc = [1.0] * n_experts
        while len(sc) < n_experts:
            sc.append(sc[-1] * 10.0)
        self.tau_inits = tuple(sc)
        self.temporal_window = temporal_window
        self.use_dual_attention = use_dual_attention

        # K experts, each with its own τ_init.
        self.experts = nn.ModuleList(
            [
                CfCCell(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    n_tau=1,
                    tau_scales=(float(self.tau_inits[i]),),
                )
                for i in range(self.n_experts)
            ]
        )
        # Initialize each expert's time_scale to the configured τ.
        with torch.no_grad():
            for i, expert in enumerate(self.experts):
                expert.time_scale.fill_(float(self.tau_inits[i]))

        # Router: FAME top-K softmax over [x_t; h_prev].
        self.router = ForecastabilityRouter(
            input_size=input_size,
            hidden_size=hidden_size,
            n_experts=self.n_experts,
            top_k=self.top_k,
            router_hidden=router_hidden,
        )

        # Feature-level attention: per-step input gate.
        if feat_attn_hidden > 0:
            self.feat_attn = nn.Sequential(
                nn.Linear(input_size + hidden_size, feat_attn_hidden),
                nn.Tanh(),
                nn.Linear(feat_attn_hidden, input_size),
            )
        else:
            self.feat_attn = nn.Linear(input_size + hidden_size, input_size)

        # Temporal attention: learned query/key projections.
        self.temporal_query = nn.Linear(hidden_size, hidden_size)
        self.temporal_key = nn.Linear(hidden_size, hidden_size)
        # Project temporal context (hidden_size) to input_size so it
        # can be used as a bias on the (gated) input.
        self.temporal_context_proj = nn.Linear(hidden_size, input_size)

        # Internal state for the temporal window.
        self._temporal_window: list[torch.Tensor] = []
        # Caches for diagnostics.
        self.last_g: torch.Tensor | None = None
        self.last_router_logits: torch.Tensor | None = None
        self.last_feat_alpha: torch.Tensor | None = None
        self.last_temporal_attn: torch.Tensor | None = None

    def reset_state(self) -> None:
        """Reset the temporal attention window (call between sequences)."""
        self._temporal_window = []
        self.last_g = None
        self.last_router_logits = None
        self.last_feat_alpha = None
        self.last_temporal_attn = None

    def _feature_attention(
        self, x_t: torch.Tensor, h: torch.Tensor,
    ) -> torch.Tensor:
        cat = torch.cat([x_t, h], dim=-1)
        alpha = torch.sigmoid(self.feat_attn(cat))
        self.last_feat_alpha = alpha
        return alpha * x_t

    def _temporal_attention(
        self, h: torch.Tensor,
    ) -> torch.Tensor:
        if not self._temporal_window:
            self.last_temporal_attn = None
            return h
        # Stack past states: [B, W, H]
        past = torch.stack(self._temporal_window, dim=1)
        # Query from current state: [B, 1, H]
        q = self.temporal_query(h).unsqueeze(1)
        # Keys from past states: [B, W, H]
        k = self.temporal_key(past)
        # Scaled dot-product attention.
        scale = float(self.hidden_size) ** 0.5
        scores = (q * k).sum(dim=-1) / scale  # [B, W]
        attn = F.softmax(scores, dim=-1)  # [B, W]
        self.last_temporal_attn = attn
        # Context vector: [B, H]
        context = (attn.unsqueeze(-1) * past).sum(dim=1)
        return context

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        h_new, _ = self.forward_with_aux(x_t, h, dt=dt)
        return h_new

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # 1. Feature-level attention on the input.
        if self.use_dual_attention:
            x_t_gated = self._feature_attention(x_t, h)
            # 2. Temporal attention (context) — use it as a *bias add*
            #    to the gated input.  This lets gradient flow to the
            #    query/key projections while keeping the recurrent
            #    W_h·h force intact (the bias is on x, not h).
            context = self._temporal_attention(h)
            # Project context from hidden_size -> input_size and add.
            context_bias = self.temporal_context_proj(context)
            x_t_gated = x_t_gated + 0.1 * context_bias
        else:
            x_t_gated = x_t

        # 3. Router over [x_t_gated; h].
        g = self.router(x_t_gated, h)
        self.last_g = g.detach()
        # ForecastabilityRouter doesn't expose raw logits (only g);
        # use the post-softmax weights as a proxy for "router_logits"
        # (gradient still flows through g → router weights).
        self.last_router_logits = g

        # 4. Each expert processes (x_t_gated, h) — preserves W_h·h.
        expert_outs = []
        for expert in self.experts:
            h_k = expert(x_t_gated, h, dt=dt)
            expert_outs.append(h_k)
        # 5. Router-weighted mixture.
        stacked = torch.stack(expert_outs, dim=1)  # [B, K, H]
        h_new = (g.unsqueeze(-1) * stacked).sum(dim=1)  # [B, H]

        # 6. Update the temporal window with the *output* hidden state
        #    (detached, to keep the autograd graph acyclic).
        self._temporal_window.append(h_new.detach())
        if len(self._temporal_window) > self.temporal_window:
            self._temporal_window.pop(0)
        return h_new, expert_outs

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"n_experts={self.n_experts}, top_k={self.top_k}, "
            f"tau_inits={self.tau_inits}, "
            f"temporal_window={self.temporal_window}, "
            f"use_dual_attention={self.use_dual_attention}"
        )


class MRMoEDualAttnCfCNetwork(nn.Module):
    """Stacked MR-MoE + Dual Attention CfC network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        return_sequences: bool = True,
        n_experts: int = 3,
        top_k: int = 2,
        tau_inits: tuple = (0.1, 1.0, 10.0),
        temporal_window: int = 4,
        use_dual_attention: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_size = input_size if layer_idx == 0 else hidden_size
            self.cells.append(
                MRMoEDualAttnCfCCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    n_experts=n_experts,
                    top_k=top_k,
                    tau_inits=tau_inits,
                    temporal_window=temporal_window,
                    use_dual_attention=use_dual_attention,
                )
            )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = [
            torch.zeros(B, self.hidden_size, device=x.device)
            for _ in range(self.num_layers)
        ]
        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            for li, cell in enumerate(self.cells):
                # Reset temporal window on the first step of a new
                # sequence (caller is expected to call reset_state
                # between sequences; here we leave the window intact
                # for within-sequence continuity).
                if li == 0:
                    h_new, _ = cell.forward_with_aux(inp, h[li])
                else:
                    h_new, _ = cell.forward_with_aux(h[li - 1], h[li])
                h[li] = h_new
            outputs.append(self.head(h[-1]))
        outputs = torch.stack(outputs, dim=1)
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]
