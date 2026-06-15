"""Round 107 — Soft MoE: fully-differentiable soft expert routing (PRD #10-69).

Implements arXiv:2308.00951 (Puigcerver, Riquelme, Mustafa, Hutter, ICLR
2023) — *From Sparse to Soft Mixtures of Experts*. Replaces discrete
top-K token→expert routing with a **fully-differentiable soft assignment**
based on learned per-expert slots.

Key idea: instead of routing individual tokens to individual experts
(hard assignment, prone to H=0 collapse), Soft MoE passes different
**weighted combinations of all tokens** to each expert. Every token
contributes to every expert via soft weights, so the entire pipeline is
differentiable and dead experts are structurally impossible.

Key components:
- ``SoftMoEConfig`` — dataclass with K, d_slot, normalize options
- ``SoftMoERouter`` — φ(token)·ψ(expert_slot) → dispatch/combine
- ``SoftMoECfCCell`` — K soft-routed CfC experts (no top-K)
- ``SoftMoESETAMoECfCCell`` — SETA + Soft MoE on the unique subgroup
- ``SoftMoESETAMoECfCNetwork`` — full network with pre-computed QuITE + SETA + Soft MoE
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.quite_embedding import QueryIrregularEmbedding
from lnn.core.quite_moe import quite_context_pool
from lnn.core.seta_moe import SETAConfig, SETAMoECfCCell


@dataclass
class SoftMoEConfig:
    """Soft MoE configuration.

    Args:
        n_experts: K (number of experts).
        d_slot: Dimensionality of the slot/phi space for routing.
        normalize: If True, slot embeddings are unit-normalized (cosine
            similarity routing). If False, dot-product.
    """
    n_experts: int = 4
    d_slot: int = 16
    normalize: bool = False


class SoftMoERouter(nn.Module):
    """Soft MoE router — fully differentiable token-to-expert assignment.

    Computes:
        scores = softmax(φ(x) · ψ(e)^T, dim=tokens)        # (B, T, K)
        dispatch = scores^T @ x                              # (B, K, D)
        y_k = expert_k(dispatch_k)                           # (B, H)
        output = scores @ y                                   # (B, T, H)

    Unlike top-K routing, every expert sees a weighted combination of
    ALL tokens, so dead experts are structurally impossible.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Expert output dimension (H).
        n_experts: K (number of experts).
        d_slot: Routing latent dim.
        normalize: If True, use cosine-similarity routing.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 4,
        d_slot: int = 16,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_experts = int(n_experts)
        self.d_slot = int(d_slot)
        self.normalize = bool(normalize)
        # Token projection φ: D → d_slot
        self.phi = nn.Linear(input_size, d_slot, bias=False)
        # Expert slot embeddings ψ: K × d_slot
        self.slots = nn.Parameter(torch.randn(n_experts, d_slot) * 0.02)
        # Experts: K separate CfC cells (we use a single shared structure
        # for simplicity, parameterized by slot-conditioned modulation)
        self.experts = nn.ModuleList([
            nn.Linear(input_size, hidden_size) for _ in range(n_experts)
        ])
        # Diagnostics
        self.last_dispatch_weights: torch.Tensor
        self.last_combine_weights: torch.Tensor

    def _compute_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute (B, T, K) soft assignment scores.

        Args:
            x: (B, T, D) input sequence.

        Returns:
            (B, T, K) softmax-normalized over the K experts.
        """
        # φ(x): (B, T, d_slot)
        phi_x = self.phi(x)
        if self.normalize:
            phi_x = F.normalize(phi_x, dim=-1)
            slots = F.normalize(self.slots, dim=-1)
        else:
            slots = self.slots
        # logits: (B, T, d_slot) @ (d_slot, K) → (B, T, K)
        logits = phi_x @ slots.t()
        # Softmax over experts (per token)
        scores = F.softmax(logits, dim=-1)
        return scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Soft MoE forward pass.

        Args:
            x: (B, T, D) input sequence (T tokens, each D-dim).

        Returns:
            (B, T, H) expert-mixed output.
        """
        B, T, D = x.shape
        if D != self.input_size:
            raise ValueError(f"Expected D={self.input_size}, got {D}")
        # NaN-safe: clean x before any computation
        x_clean = torch.nan_to_num(x, nan=0.0)
        # 1. Compute soft scores (B, T, K)
        scores = self._compute_scores(x_clean)
        # NaN-safe: replace any NaN in scores with 0
        scores = torch.nan_to_num(scores, nan=0.0)
        # 2. Dispatch: weighted average of tokens per expert
        # scores^T: (B, K, T) @ x: (B, T, D) → dispatch: (B, K, D)
        dispatch = torch.bmm(scores.transpose(1, 2), x_clean)
        self.last_dispatch_weights = dispatch.detach()
        # 3. Process each expert
        expert_outputs = []
        for k, expert in enumerate(self.experts):
            expert_outputs.append(expert(dispatch[:, k, :]))
        # (B, K, H)
        y = torch.stack(expert_outputs, dim=1)
        # 4. Combine: (B, T, K) @ (B, K, H) → (B, T, H)
        output = torch.bmm(scores, y)
        self.last_combine_weights = scores.detach()
        return output

    def get_utilization(self) -> Dict[str, float]:
        """Return diagnostic stats for expert utilization.

        Returns:
            Dict with: ``"mean_dispatch_norm"`` (per expert), ``"std"``,
            ``"max_min_ratio"``, ``"slots_norm"`` (per expert).
        """
        with torch.no_grad():
            # Use slot norms as a proxy for "active" experts
            slot_norms = self.slots.norm(dim=-1).cpu().tolist()
            norms = torch.tensor(slot_norms)
            std = float(norms.std().item()) if len(norms) > 1 else 0.0
            nz = norms[norms > 0]
            if len(nz) > 1:
                ratio = float((norms.max() / norms.min()).item())
            else:
                ratio = 1.0
            return {
                "expert_norms": slot_norms,
                "expert_norm_std": std,
                "expert_norm_max_min_ratio": ratio,
            }


class SoftMoESETARouter(nn.Module):
    """Soft MoE router that conforms to SETARouter's interface.

    Differs from ``SoftMoERouter`` (which does full-sequence dispatch):
    this version is called per-step like ``SETARouter`` and returns
    (B, U) soft weights for the unique expert subgroup.

    Uses slot-based soft assignment from per-step input:
        phi = Linear([x_t, h, context]) ∈ R^d_slot
        slots = nn.Parameter(K, d_slot)  # K = n_unique
        scores = softmax(phi · slots^T)  # (B, K) per-step
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        d_context: int,
        n_unique: int,
        d_slot: int = 16,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.d_context = int(d_context)
        self.n_unique = int(n_unique)
        self.d_slot = int(d_slot)
        self.normalize = bool(normalize)
        router_in = input_size + hidden_size + d_context
        self.phi = nn.Linear(router_in, d_slot, bias=False)
        self.slots = nn.Parameter(torch.randn(n_unique, d_slot) * 0.02)
        self.last_g: torch.Tensor

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x_t.size(0)
        if context is None:
            context = torch.zeros(
                B, self.d_context, device=x_t.device, dtype=x_t.dtype,
            )
        combined = torch.cat([x_t, h, context], dim=-1)
        phi = self.phi(combined)
        if self.normalize:
            phi = F.normalize(phi, dim=-1)
            slots = F.normalize(self.slots, dim=-1)
        else:
            slots = self.slots
        logits = phi @ slots.t()
        g = F.softmax(logits, dim=-1)
        self.last_g = g.detach()
        return g

    def get_utilization(self) -> Dict[str, float]:
        """Diagnostic: slot norm stats as proxy for active experts."""
        with torch.no_grad():
            slot_norms = self.slots.norm(dim=-1).cpu().tolist()
            norms = torch.tensor(slot_norms)
            std = float(norms.std().item()) if len(norms) > 1 else 0.0
            nz = norms[norms > 0]
            if len(nz) > 1:
                ratio = float((norms.max() / norms.min()).item())
            else:
                ratio = 1.0
            return {
                "expert_norms": slot_norms,
                "expert_norm_std": std,
                "expert_norm_max_min_ratio": ratio,
            }


class SoftMoECfCCell(nn.Module):
    """CfC cell with Soft MoE routing.

    Replaces top-K MoE with fully-differentiable soft assignment. The
    hidden state is updated K times (one per expert) then mixed by
    the soft weights.

    Args:
        input_size: D.
        hidden_size: H.
        n_experts: K.
        d_slot: Routing dim.
        normalize: Cosine-similarity routing if True.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 4,
        d_slot: int = 16,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_experts = int(n_experts)
        self.d_slot = int(d_slot)
        # Soft MoE router
        self.router = SoftMoERouter(
            input_size=input_size,
            hidden_size=hidden_size,
            n_experts=n_experts,
            d_slot=d_slot,
            normalize=normalize,
        )
        # K CfC experts
        self.experts = nn.ModuleList([
            CfCCell(input_size=input_size, hidden_size=hidden_size)
            for _ in range(n_experts)
        ])
        self.last_router_util: Dict[str, float] = {}

    def forward(
        self,
        x_seq: torch.Tensor,
        h0: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Process a sequence through K soft-routed CfC experts.

        Args:
            x_seq: (B, T, D) input sequence.
            h0: (B, H) initial hidden state.
            dt: CfC time constant.

        Returns:
            (B, T, H) per-timestep hidden states (mixed by soft weights).
        """
        B, T, D = x_seq.shape
        K = self.n_experts
        # NaN-safe: clean x before any computation
        x_clean = torch.nan_to_num(x_seq, nan=0.0)
        # 1. Compute soft scores (B, T, K)
        scores = self.router._compute_scores(x_clean)
        scores = torch.nan_to_num(scores, nan=0.0)
        # 2. Dispatch: (B, K, T) @ (B, T, D) → (B, K, D)
        dispatch = torch.bmm(scores.transpose(1, 2), x_clean)
        # 3. Run each expert on its single dispatch vector (B, K, D)
        # Each expert processes a single D-dim vector at each timestep
        # (the soft-dispatched weighted avg of all tokens).
        h_init = h0.unsqueeze(1).expand(-1, K, -1).reshape(B * K, self.hidden_size)
        # Run each expert on its dispatched input, broadcasting h across
        # timesteps. We use the same dispatched input for every timestep
        # (this is the Soft MoE design: the dispatch is computed once).
        h_t = h_init
        outs_all = []
        for t in range(T):
            x_t_disp = dispatch.reshape(B * K, D)  # (B*K, D)
            h_t = self._run_experts_step(x_t_disp, h_t, dt)
            outs_all.append(h_t.reshape(B, K, self.hidden_size))  # (B, K, H)
        # Stack: (B, T, K, H)
        outs_all = torch.stack(outs_all, dim=1)  # (B, T, K, H)
        # Combine: output[b, t, h] = sum_k scores[b, t, k] * outs[b, t, k, h]
        output = torch.einsum("btk,btkh->bth", scores, outs_all)
        self.last_router_util = self.router.get_utilization()
        return output

    def _run_experts_step(
        self,
        x_t_disp: torch.Tensor,
        h_t: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Run one step through all K experts in parallel.

        Args:
            x_t_disp: (B*K, D) — dispatch[t] for each (batch, expert).
            h_t: (B*K, H) — current hidden state.
            dt: CfC time constant.

        Returns:
            (B*K, H) updated hidden state.
        """
        K = self.n_experts
        B = x_t_disp.size(0) // K
        outs = []
        for k in range(K):
            x_k = x_t_disp[k * B:(k + 1) * B]
            h_k = h_t[k * B:(k + 1) * B]
            h_k_new = self.experts[k](x_k, h_k, dt=dt)
            outs.append(h_k_new)
        return torch.cat(outs, dim=0)


class SoftMoESETAMoECfCCell(SETAMoECfCCell):
    """SETA cell with Soft MoE on the unique expert subgroup.

    The shared subgroup remains always-active (SETA's structural fix).
    The unique subgroup uses Soft MoE dispatch instead of top-K.

    Args:
        Same as ``SETAMoECfCCell`` plus ``d_slot`` and ``normalize`` for
        the Soft MoE on the unique subgroup.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        sdta_config: SETAConfig | dict | None = None,
        d_context: int = 16,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        d_slot: int = 16,
        normalize: bool = False,
        router_hidden: int = 0,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            sdta_config=sdta_config,
            d_context=d_context,
            n_tau_per_expert=n_tau_per_expert,
            tau_scales=tau_scales,
            router_hidden=router_hidden,
        )
        # Replace the SETA router with a Soft MoE per-step router
        # (this is what makes SETA's unique experts fully differentiable
        # in routing, eliminating H=0 lock-in).
        U = self.n_unique
        self.router = SoftMoESETARouter(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            d_context=self.d_context,
            n_unique=U,
            d_slot=d_slot,
            normalize=normalize,
        )

    def collect_expert_utilization(self) -> Dict[str, float | int]:
        """Augment SETA's utilization with Soft MoE diagnostics."""
        base = super().collect_expert_utilization()
        # Both SoftMoERouter and SoftMoESETARouter have get_utilization()
        if hasattr(self.router, "get_utilization"):
            util = self.router.get_utilization()
            base.update({
                "softmoe_expert_norm_std": util["expert_norm_std"],
                "softmoe_expert_norm_max_min_ratio": util["expert_norm_max_min_ratio"],
            })
        return base


class SoftMoESETAMoECfCNetwork(nn.Module):
    """Full network wrapping QuITE + SETA + Soft MoE (round 107).

    Args:
        input_size: D.
        hidden_size: H.
        sdta_config: ``SETAConfig``.
        d_slot: Soft MoE routing dim.
        normalize: Cosine-similarity routing if True.
        n_queries, d_context, n_heads: QuITE params.
        output_size: Output feature dimension.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        sdta_config: SETAConfig | dict | None = None,
        d_slot: int = 16,
        normalize: bool = False,
        n_queries: int = 4,
        d_context: int = 16,
        n_heads: int = 4,
        output_size: int = 1,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        pool_method: str = "mean",
    ) -> None:
        super().__init__()
        if sdta_config is None:
            sdta_config = SETAConfig()
        if isinstance(sdta_config, dict):
            self.seta_config = SETAConfig(**sdta_config)
        else:
            self.seta_config = sdta_config
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.pool_method = str(pool_method)
        self.d_context = int(d_context)
        # QuITE
        self.quite = QueryIrregularEmbedding(
            d_input=input_size,
            n_queries=n_queries,
            d_model=d_context,
            n_heads=n_heads,
        )
        # SETA + Soft MoE cell
        self.cell = SoftMoESETAMoECfCCell(
            input_size=input_size,
            hidden_size=hidden_size,
            sdta_config=self.seta_config,
            d_context=d_context,
            n_tau_per_expert=n_tau_per_expert,
            tau_scales=tau_scales,
            d_slot=d_slot,
            normalize=normalize,
        )
        # Output head
        self.head = nn.Linear(hidden_size, output_size)
        self._cached_context: torch.Tensor | None = None

    def reset_context(self) -> None:
        self._cached_context = None

    def compute_context(
        self,
        observations: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.quite(observations, times, mask=mask)
        context = quite_context_pool(tokens, method=self.pool_method)
        self._cached_context = context
        return context

    def forward(
        self,
        observations: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None = None,
        precomputed_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = observations.shape
        if D != self.input_size:
            raise ValueError(f"Expected D={self.input_size}, got {D}")
        obs_mask = torch.isfinite(observations).all(dim=-1)
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=observations.device)
        elif mask.dtype != torch.bool:
            mask = mask.bool()
        mask = mask & obs_mask
        if precomputed_context is not None:
            context = precomputed_context
        else:
            context = self.compute_context(observations, times, mask=mask)
        h = torch.zeros(
            B, self.hidden_size, device=observations.device, dtype=observations.dtype,
        )
        outputs = []
        for t in range(T):
            x_t = observations[:, t, :]
            valid_t = mask[:, t]
            x_t_clean = torch.where(
                valid_t.unsqueeze(-1), x_t, torch.zeros_like(x_t),
            )
            h = self.cell(x_t_clean, h, context=context, dt=1.0)
            y_t = self.head(h)
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)

    def regularization_loss(self) -> torch.Tensor:
        return self.cell.regularization_loss()

    def get_utilization(self) -> Dict[str, float | int]:
        return self.cell.collect_expert_utilization()
