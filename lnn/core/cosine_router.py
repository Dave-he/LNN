"""CosineRouter: parameter-free top-K router via cosine similarity
(PRD #10-41, 2026-06-14).

Implements the "parameter-free online K-Means router" from
arXiv:2605.12476 (*Routers Learn the Geometry of Their Experts*,
Ahrac et al., 2026-05-12):

> "We demonstrate the centrality of geometric coupling for effective
> routing with a parameter-free online K-Means router, in which each
> expert maintains a running average of the hidden states routed to it
> and tokens are assigned based on cosine similarity.  Compared with
> auxiliary-loss and loss-free balancing, this router achieves the
> lowest load imbalance with only a modest perplexity increase."

The router has **zero learned parameters** (``self.parameters()`` is
empty).  It maintains ``expert_means ∈ R^{K × (input+hidden)}`` as
buffers, and assigns each input ``[x_t; h]`` to the top-K experts
based on cosine similarity to those means.  The means are updated
in-place by an EMA of the routed states (no_grad).

This is the **third routing strategy** in the LNN+MoE stack:
- round 77: learned softmax router
- round 78: learned sparse top-K router
- round 81: learned sparse top-K router + φ-bias
- **round 82 (here)**: parameter-free cosine router

The CosineRouter is intentionally a drop-in alternative to
``ForecastabilityRouter`` — same ``forward(x_t, h) → g`` API and
same ``last_top_idx`` side-channel.  Callers must additionally
invoke ``router.update(combined, top_idx)`` in train mode (the
``FAMECfCCell`` does this automatically when ``router_type='cosine'``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineRouter(nn.Module):
    """Parameter-free top-K mixture router via cosine similarity to
    per-expert running hidden-state means.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        n_experts: K (number of experts).
        top_k: K' ∈ [1, K] — number of experts activated per step.
            ``top_k == K`` is numerically equivalent to dense softmax
            (within float32 eps), matching the ``ForecastabilityRouter``
            contract.
        ema_alpha: EMA decay rate for per-expert mean update
            (0 ≤ α ≤ 1).  Default 0.01 matches the paper's "negligible
            overhead" regime.  ``alpha=0`` means "frozen, manual control
            only".
        eps: Numerical floor on the cosine denominator.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int,
        top_k: int = 2,
        ema_alpha: float = 0.01,
        eps: float = 1e-8,
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
        self.ema_alpha = float(ema_alpha)
        self.eps = float(eps)
        # Buffers: per-expert running mean of the COMBINED [x_t; h] state.
        # Init to zeros (deterministic; first batch makes them all equal
        # → uniform softmax; EMA differentiates after a few steps).
        self.register_buffer(
            "expert_means",
            torch.zeros(self.n_experts, self.input_size + self.hidden_size),
        )
        # Update counter (diagnostic).
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        # Side-channel: filled on every forward() call.
        self.last_top_idx: torch.Tensor

    @property
    def num_learned_parameters(self) -> int:
        """Number of trainable parameters — always 0 for this router."""
        return 0

    @torch.no_grad()
    def reset_state(self) -> None:
        """Reset per-expert means to zero and step counter to 0."""
        self.expert_means.zero_()
        self.step.zero_()

    @torch.no_grad()
    def update(self, combined: torch.Tensor, top_idx: torch.Tensor) -> None:
        """Update per-expert running mean from the routed hidden states.

        Args:
            combined: [B, input+hidden] — the ``[x_t; h]`` features
                that were just routed.
            top_idx: [B, top_k] long tensor of activated-expert indices
                per batch element (output of ``topk``).
        """
        if top_idx.numel() == 0:
            return
        K = self.n_experts
        # For each expert, find batch elements routed to it (any of the
        # top_k spots) and update its mean with the per-batch mean.
        for k in range(K):
            # mask: [B] — True if expert k is in this batch element's top-k.
            mask = (top_idx == k).any(dim=-1)
            if mask.any():
                routed = combined[mask]  # [n_routed, D]
                m_batch = routed.mean(dim=0)
                # EMA: m = (1-α) m + α m_batch
                self.expert_means[k].mul_(1.0 - self.ema_alpha).add_(
                    self.ema_alpha * m_batch
                )
        self.step.add_(1)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Return top-K sparse mixture weights ``g ∈ Δ^K`` with K-K' zeros.

        Args:
            x_t: [B, input_size] input at this step.
            h:   [B, hidden_size] previous hidden state.

        Returns:
            g: [B, n_experts] mixture weights, with exactly ``top_k``
               non-zero entries per row.
        """
        combined = torch.cat([x_t, h], dim=-1)  # [B, D]
        # L2-normalize.
        means_norm = F.normalize(self.expert_means, dim=-1, eps=self.eps)  # [K, D]
        comb_norm = F.normalize(combined, dim=-1, eps=self.eps)            # [B, D]
        # Cosine similarity matrix.
        sim = comb_norm @ means_norm.t()  # [B, K]
        if self.top_k == self.n_experts:
            # No masking needed; pure dense softmax path.
            g = F.softmax(sim, dim=-1)
            self.last_top_idx = torch.arange(
                self.n_experts, device=sim.device,
            ).expand(sim.size(0), -1)
            return g
        # Mask non-top-K positions to -inf so softmax → 0 there.
        top_result = sim.topk(self.top_k, dim=-1)  # [B, K']
        top_idx = top_result.indices
        del top_result
        mask = torch.full_like(sim, float("-inf"))
        mask.scatter_(-1, top_idx, 0.0)
        g = F.softmax(sim + mask, dim=-1)  # [B, K], exactly K' nonzeros
        self.last_top_idx = top_idx
        return g
