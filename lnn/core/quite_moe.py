"""Round 103 — QuITE+MoE: Irregularity-Context-Aware Expert Routing (PRD #10-65).

Combines the round 102 QuITE query-based irregular-TS embedding
(arXiv:2605.28166, Lim ICML 2026) with the round 78 FAME-style top-K
sparse MoE routing (arXiv:2606.08896).

Key idea:
  1. Pre-compute a QuITE context vector from the full irregular
     sequence — captures the GLOBAL irregularity pattern
     (which timesteps are missing, the overall distribution, etc.)
  2. Concatenate this context to the per-step [x_t, h_prev] router
     input, so the expert routing decision is informed by both the
     local state AND the global irregularity fingerprint.

This is the FIRST principled combination of two distinct mechanisms
from our 27-layer stack (QuITE embedding + FAME top-K routing).

Functions and classes:
- ``QuiteRouter`` (nn.Module) — router that concatenates QuITE context
- ``QuiteMoECfCCell`` (nn.Module) — K CfCCell experts + QuiteRouter
- ``QuiteMoECfCNetwork`` (nn.Module) — full network with pre-computed QuITE
- ``quite_context_pool`` — mean pool (B, n_queries, d) → (B, d)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.quite_embedding import QueryIrregularEmbedding


def quite_context_pool(
    tokens: torch.Tensor,
    method: str = "mean",
) -> torch.Tensor:
    """Pool QuITE query tokens into a single context vector.

    Args:
        tokens: (B, n_queries, d_model) QuITE output tokens.
        method: one of 'mean', 'max', 'first'.
            - 'mean': average over query tokens → (B, d_model)
            - 'max': max over query tokens → (B, d_model)
            - 'first': take first query token → (B, d_model)

    Returns:
        (B, d_model) pooled context vector.
    """
    if method not in ("mean", "max", "first"):
        raise ValueError(f"method must be one of mean/max/first, got {method!r}")
    if tokens.dim() != 3:
        raise ValueError(
            f"tokens must be (B, n_queries, d_model), got {tuple(tokens.shape)}",
        )
    if method == "mean":
        return tokens.mean(dim=1)
    if method == "max":
        return tokens.max(dim=1).values
    # first
    return tokens[:, 0, :]


class QuiteRouter(nn.Module):
    """FAME-style top-K router augmented with QuITE context.

    Combines the per-step [x_t, h_prev] signal with a pre-computed
    QuITE context vector to produce K logits → top-K mixture weights.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension (H).
        d_context: Dimension of the QuITE context vector.
        n_experts: Number of experts (K ≥ 1).
        top_k: Number of experts activated per step.
        router_hidden: Width of an optional 2-layer router MLP.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        d_context: int,
        n_experts: int,
        top_k: int = 2,
        router_hidden: int = 0,
    ) -> None:
        super().__init__()
        assert n_experts >= 1
        assert 1 <= top_k <= n_experts
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.d_context = int(d_context)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.router_hidden = int(router_hidden)

        router_in = input_size + hidden_size + d_context
        if self.router_hidden > 0:
            self.router = nn.Sequential(
                nn.Linear(router_in, self.router_hidden),
                nn.Tanh(),
                nn.Linear(self.router_hidden, self.n_experts),
            )
        else:
            self.router = nn.Linear(router_in, self.n_experts)
        # Side-channel
        self.last_top_idx: torch.Tensor

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return top-K sparse mixture weights.

        Args:
            x_t: (B, input_size) input at this step.
            h: (B, hidden_size) previous hidden state.
            context: (B, d_context) QuITE context vector (pre-computed).
                If None, falls back to standard FAME-style routing
                with [x_t, h, zeros(d_context)] — context slot is
                zero-filled so the linear layer input dim is consistent.

        Returns:
            g: (B, n_experts) mixture weights with exactly top_k
                non-zero entries per row.
        """
        B = x_t.size(0)
        if context is None:
            # Zero-fill the context slot so linear layer input dim matches.
            context = torch.zeros(
                B, self.d_context, device=x_t.device, dtype=x_t.dtype,
            )
        else:
            if context.size(-1) != self.d_context:
                raise ValueError(
                    f"Expected context dim {self.d_context}, got {context.size(-1)}",
                )
        combined = torch.cat([x_t, h, context], dim=-1)
        logits = self.router(combined)
        if self.top_k == self.n_experts:
            g = F.softmax(logits, dim=-1)
            self.last_top_idx = torch.arange(
                self.n_experts, device=logits.device,
            ).expand(logits.size(0), -1)
            return g
        top_result = logits.topk(self.top_k, dim=-1)
        top_idx = top_result.indices
        del top_result
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(-1, top_idx, 0.0)
        masked_logits = logits + mask
        g = F.softmax(masked_logits, dim=-1)
        self.last_top_idx = top_idx
        return g


class QuiteMoECfCCell(nn.Module):
    """FAME-style top-K sparse MoE cell augmented with QuITE context.

    The cell holds ``n_experts`` independent ``CfCCell`` experts and
    uses a ``QuiteRouter`` for routing.  Each forward step takes a
    pre-computed QuITE context vector that informs the routing
    decision.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension (H).
        n_experts: Number of experts (K).
        top_k: Number of experts activated per step.
        n_tau_per_expert: Per-expert ``n_tau`` (round 76 compatibility).
        tau_scales: Per-branch initial time constants.
        d_context: Dimension of the QuITE context vector.
        router_hidden: Width of the optional 2-layer router MLP.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 3,
        top_k: int = 2,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        d_context: int = 16,
        router_hidden: int = 0,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.d_context = int(d_context)
        self.router_hidden = int(router_hidden)
        # K CfCCell experts (round 77 compatibility)
        self.experts = nn.ModuleList(
            [
                CfCCell(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    n_tau=self.n_tau_per_expert,
                    tau_scales=self.tau_scales,
                )
                for _ in range(self.n_experts)
            ]
        )
        # QuITE-augmented router
        self.router = QuiteRouter(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            d_context=self.d_context,
            n_experts=self.n_experts,
            top_k=self.top_k,
            router_hidden=self.router_hidden,
        )
        # Side-channel: last g for diagnostics
        self.last_g: torch.Tensor

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        context: torch.Tensor | None = None,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One step of QuITE-augmented top-K sparse MoE.

        Args:
            x_t: (B, input_size) input at this step.
            h: (B, hidden_size) previous hidden state.
            context: (B, d_context) QuITE context vector (pre-computed
                once per sequence by ``QuiteMoECfCNetwork``). If None,
                falls back to standard [x_t, h] routing.
            dt: scalar or [B] per-sample time delta.

        Returns:
            h_new: (B, hidden_size) mixed expert output.
        """
        # Compute expert outputs
        expert_outs = []
        for expert in self.experts:
            h_k = expert(x_t, h, dt=dt)
            expert_outs.append(h_k)
        # (B, K, H)
        expert_stack = torch.stack(expert_outs, dim=1)
        # Routing weights (B, K)
        g = self.router(x_t, h, context=context)
        self.last_g = g.detach()
        # Mixture
        h_new = (g.unsqueeze(-1) * expert_stack).sum(dim=1)
        return h_new


class QuiteMoECfCNetwork(nn.Module):
    """Full network that pre-computes QuITE context and routes per step.

    Wraps a ``QuiteMoECfCCell`` with a ``QueryIrregularEmbedding``
    (round 102). The network expects the FULL irregular sequence at
    once and returns the per-step output sequence.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension (H).
        n_experts: Number of experts (K).
        top_k: Number of experts activated per step.
        n_queries: Number of QuITE learnable query tokens.
        d_context: QuITE context dimension (must match QuITE d_model).
        n_heads: Number of attention heads in QuITE.
        output_size: Output feature dimension.
        n_tau_per_expert: Per-expert ``n_tau``.
        tau_scales: Per-branch initial time constants.
        pool_method: How to pool QuITE tokens into context
            ('mean', 'max', 'first').
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 3,
        top_k: int = 2,
        n_queries: int = 8,
        d_context: int = 16,
        n_heads: int = 4,
        output_size: int = 1,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        pool_method: str = "mean",
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.n_queries = int(n_queries)
        self.d_context = int(d_context)
        self.n_heads = int(n_heads)
        self.output_size = int(output_size)
        self.pool_method = str(pool_method)
        # QuITE module (round 102)
        self.quite = QueryIrregularEmbedding(
            d_input=input_size,
            n_queries=n_queries,
            d_model=d_context,
            n_heads=n_heads,
        )
        # QuITE-augmented MoE cell
        self.cell = QuiteMoECfCCell(
            input_size=input_size,
            hidden_size=hidden_size,
            n_experts=n_experts,
            top_k=top_k,
            n_tau_per_expert=n_tau_per_expert,
            tau_scales=tau_scales,
            d_context=d_context,
        )
        # Output projection
        self.head = nn.Linear(hidden_size, output_size)
        # Cached context (computed once per forward)
        self._cached_context: torch.Tensor | None = None

    def reset_context(self) -> None:
        """Clear the cached QuITE context (call at sequence boundaries)."""
        self._cached_context = None

    def compute_context(
        self,
        observations: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pre-compute QuITE context from the full irregular sequence.

        Args:
            observations: (B, T, D) input values. NaN = missing.
            times: (B, T) input time stamps.
            mask: (B, T) bool mask (True = valid). If None, all valid.

        Returns:
            (B, d_context) context vector (also cached for forward()).
        """
        tokens = self.quite(observations, times, mask=mask)  # (B, n_queries, d_context)
        context = quite_context_pool(tokens, method=self.pool_method)  # (B, d_context)
        self._cached_context = context
        return context

    def forward(
        self,
        observations: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None = None,
        precomputed_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the full sequence with QuITE-augmented routing.

        Args:
            observations: (B, T, D) input values. NaN = missing.
            times: (B, T) input time stamps.
            mask: (B, T) bool mask (True = valid). If None, all valid.
            precomputed_context: (B, d_context) optional pre-computed
                QuITE context. If None, computed from observations.

        Returns:
            (B, T, output_size) per-step outputs.
        """
        B, T, D = observations.shape
        if D != self.input_size:
            raise ValueError(
                f"Expected D={self.input_size}, got {D}",
            )
        # Build validity mask: NaN observations are ALWAYS treated as
        # missing, regardless of the user-passed mask.
        obs_mask = torch.isfinite(observations).all(dim=-1)  # (B, T)
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=observations.device)
        elif mask.dtype != torch.bool:
            mask = mask.bool()
        mask = mask & obs_mask  # (B, T) — NaN-aware combined mask
        # Step 1: get context
        if precomputed_context is not None:
            context = precomputed_context
        else:
            context = self.compute_context(observations, times, mask=mask)
        # Step 2: recurrent forward
        h = torch.zeros(B, self.hidden_size, device=observations.device, dtype=observations.dtype)
        outputs = []
        for t in range(T):
            x_t = observations[:, t, :]  # (B, D)
            valid_t = mask[:, t]  # (B,)
            # Replace NaN with 0 to avoid NaN in experts
            x_t_clean = torch.where(
                valid_t.unsqueeze(-1), x_t, torch.zeros_like(x_t),
            )
            h = self.cell(x_t_clean, h, context=context, dt=1.0)
            y_t = self.head(h)  # (B, output_size)
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)  # (B, T, output_size)


__all__ = [
    "QuiteRouter",
    "QuiteMoECfCCell",
    "QuiteMoECfCNetwork",
    "quite_context_pool",
]
