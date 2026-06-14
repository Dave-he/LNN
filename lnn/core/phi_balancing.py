"""φ-Balancing: EMA-based expert load balancing (PRD #10-40, 2026-06-14).

Implements the minimum-viable LNN-flavored version of the φ-Balancing
framework (arXiv:2605.15403, 2026-05-14, Chen et al.):

> "Mixture-of-Experts (MoE) models rely on balanced expert utilization
> to fully realize their scalability. However, existing load-balancing
> methods are largely heuristic and operate on noisy mini-batch
> assignment statistics, introducing bias relative to population-level
> objectives. We propose φ-balancing, a principled framework that
> directly targets population-level expert balance by minimizing a
> strictly convex, symmetric, and differentiable potential of the
> expected routing distribution. Using convex duality, we derive an
> equivalent min-max formulation and obtain a simple online algorithm
> via mirror descent, yielding an efficient EMA-based routing
> adjustment with negligible overhead."

We use the simplest strictly-convex symmetric differentiable potential:

    φ(f) = -Σ_k f_k log f_k          (negative entropy)

The mirror-descent gradient is dφ/df_k = -log f_k - 1, and the
corresponding bias that should be ADDED to the router logits is
proportional to -dφ/df_k ∝ log f_k.  We use a simple sign convention:
**high f_k → negative bias (demote), low f_k → positive bias (promote)**.
This encourages uniform utilization without introducing any extra
gradient — the bias is a no_grad buffer.

The interface is intentionally minimal: ``PhiBalancer.update(...)`` is
called once per step with the hard assignment indicators
(e.g. ``last_top_idx``), and ``forward(logits)`` returns the biased
logits.  This composes with the existing ``ForecastabilityRouter``
without changing its public API.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PhiBalancer(nn.Module):
    """EMA-tracked per-expert bias for φ-balancing.

    Args:
        n_experts: K, the number of experts.
        ema_alpha: EMA decay (0 < α ≤ 1).  Smaller α = slower update.
            Default 0.01 matches the paper's "negligible overhead" regime.
        step_size: Mirror-descent step size η.  Default 0.01.
        eps: Numerical floor on the assignment fraction f_k, to keep
            ``log(f_k)`` finite when an expert is rarely used.
    """

    def __init__(
        self,
        n_experts: int,
        ema_alpha: float = 0.01,
        step_size: float = 0.01,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert n_experts >= 1, f"n_experts must be >= 1, got {n_experts}"
        assert 0.0 <= ema_alpha <= 1.0, f"ema_alpha must be in [0, 1], got {ema_alpha}"
        assert step_size >= 0.0, f"step_size must be >= 0, got {step_size}"
        self.n_experts = int(n_experts)
        self.ema_alpha = float(ema_alpha)
        self.step_size = float(step_size)
        self.eps = float(eps)

        # Buffers: f = EMA of assignment rate, b = mirror-descent bias.
        # Both are non-parameter state (move with .to(device) etc.) but
        # are not in self.parameters() so they don't show up in
        # parameter-count or get optimized.
        self.register_buffer(
            "f", torch.full((self.n_experts,), 1.0 / self.n_experts)
        )
        self.register_buffer("b", torch.zeros(self.n_experts))

    @torch.no_grad()
    def reset_state(self) -> None:
        """Reset EMA and bias to the initial uniform state."""
        self.f.copy_(torch.full((self.n_experts,), 1.0 / self.n_experts))
        self.b.zero_()

    @torch.no_grad()
    def update(self, top_idx: torch.Tensor) -> None:
        """Update EMA from per-batch hard top-K assignment indices.

        Args:
            top_idx: [B, K'] long tensor of activated-expert indices per
                batch element.  K' is ``router.top_k``.  Each value
                must be in ``[0, n_experts)``.
        """
        if top_idx.numel() == 0:
            return
        K = self.n_experts
        # Build a one-hot assignment matrix [B, K].
        # Note: for top_k > 1, each batch element activates K' experts,
        # so we use scatter_(1, ..., 1.0) and divide by K' to get the
        # average activation per expert per batch element.
        bsize, kprime = top_idx.shape
        one_hot = torch.zeros(bsize, K, device=top_idx.device, dtype=self.f.dtype)
        one_hot.scatter_(1, top_idx, 1.0)
        if kprime > 1:
            one_hot = one_hot / float(kprime)
        # Per-expert fraction in the batch.
        f_batch = one_hot.mean(dim=0)  # [K]
        # EMA update: f = (1 - α) f + α f_batch.
        self.f.mul_(1.0 - self.ema_alpha).add_(self.ema_alpha * f_batch)
        # Mirror-descent bias: b_k = -η * log f_k (clamped).
        # We use the convention that ADDING b_k to logits PROMOTES expert k.
        # We want high f_k → negative b_k (demote), so b = -η * log f.
        self.b.copy_(-self.step_size * torch.log(self.f.clamp_min(self.eps)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Add the bias to the router logits.

        Args:
            logits: [B, K] raw router logits.

        Returns:
            [B, K] biased logits, ready for the top-K mask + softmax
            downstream.  Bias is broadcast over the batch dimension.
        """
        if self.step_size == 0.0:
            return logits
        return logits + self.b.unsqueeze(0)  # broadcast over batch
