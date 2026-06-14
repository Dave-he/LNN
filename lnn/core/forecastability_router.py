"""Forecastability-Aware Top-K Router (PRD #10-36, 2026-06-14).

Implements the FAME-style sparse top-K mixture routing from
arXiv:2606.08896 (Forecastability-Aware Mixture of Experts for
Heterogeneous Time Series Forecasting, 2026-06-08).

The router maps ``[x_t; h_prev]`` to K logits, then keeps the top-K'
logits, masks the rest to -inf, and applies softmax to obtain a
sparse mixture distribution.  This is the minimum-viable
implementation of the "cost-aware sparse router" idea in FAME §3.3
(``K' = 2`` by default; the paper measured 1.92 experts/series on
average in production).

Invariants:
- ``top_k == K`` is numerically equivalent to dense softmax
  (within float32 eps), so users can tune ``top_k`` without
  changing the rest of the cell.
- ``top_k == 1`` reduces to a router-argmax single expert.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForecastabilityRouter(nn.Module):
    """Sparse top-K mixture router.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        n_experts: Number of experts (K ≥ 1).
        top_k: Number of experts activated per step (K' ∈ [1, K]).
        router_hidden: Width of an optional 2-layer router MLP.
            ``0`` (default) uses a single linear layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int,
        top_k: int = 2,
        router_hidden: int = 0,
    ):
        super().__init__()
        assert n_experts >= 1, f"n_experts must be >= 1, got {n_experts}"
        assert 1 <= top_k <= n_experts, (
            f"top_k must be in [1, n_experts], got top_k={top_k}, n_experts={n_experts}"
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.router_hidden = int(router_hidden)

        router_in = input_size + hidden_size
        if self.router_hidden > 0:
            self.router = nn.Sequential(
                nn.Linear(router_in, self.router_hidden),
                nn.Tanh(),
                nn.Linear(self.router_hidden, self.n_experts),
            )
        else:
            self.router = nn.Linear(router_in, self.n_experts)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Return top-K sparse mixture weights ``g ∈ Δ^K`` with K-K' zeros.

        Args:
            x_t: [B, input_size] input at this step.
            h:   [B, hidden_size] previous hidden state.

        Returns:
            g: [B, n_experts] mixture weights, with exactly ``top_k``
               non-zero entries per row.
            top_idx: [B, top_k] indices of activated experts (side-channel
                     on ``self.last_top_idx`` for diagnostics).
        """
        combined = torch.cat([x_t, h], dim=-1)  # [B, input+hidden]
        logits = self.router(combined)          # [B, K]
        if self.top_k == self.n_experts:
            # No masking needed; pure dense softmax path.
            g = F.softmax(logits, dim=-1)
            self.last_top_idx = torch.arange(self.n_experts, device=logits.device).expand(logits.size(0), -1)
            return g
        # Mask non-top-K positions to -inf so softmax → 0 there.
        top_result = logits.topk(self.top_k, dim=-1)  # [B, K']
        top_idx = top_result.indices
        del top_result  # top_logits not needed; we keep only the indices
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(-1, top_idx, 0.0)
        masked_logits = logits + mask
        g = F.softmax(masked_logits, dim=-1)  # [B, K], exactly K' nonzeros
        self.last_top_idx = top_idx
        return g
