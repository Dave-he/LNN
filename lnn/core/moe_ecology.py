"""MoE Ecology Diagnostic: E = T*H/(O+B) (PRD #10-42, 2026-06-14).

Implements the minimum-viable LNN-flavored version of the MoE
ecology framework from arXiv:2605.06415 (*E = T*H/(O+B): A
Dimensionless Control Parameter for Mixture-of-Experts Ecology*,
Zhang, 2026):

> "We introduce E = T*H/(O+B), a dimensionless control parameter
> that predicts whether Mixture-of-Experts (MoE) models will develop
> a healthy expert ecology or collapse into dead experts. E combines
> four hyperparameters -- routing temperature T, routing entropy
> weight H, oracle weight O, and balance weight B -- into a single
> quantity. Through 12 controlled experiments (8 vision, 4 language)
> totaling over 11,000 training epochs, we establish that E >= 0.5
> alone is sufficient to guarantee zero dead experts, removing the
> necessity for handcrafted load-balancing auxiliary losses. ... Six
> additional findings emerge: (1) dead experts can resuscitate ...
> (2) ortho toxicity is dataset-dependent, not universal ..."

Two practical tools:
- ``moe_ecology_number(router_logits, last_g, T, H, O, B)`` returns
  the dimensionless E for a given cell state.
- ``MoEEcologyMonitor`` tracks per-expert utilization EMA and
  dead-expert count over training.

Mapping to our FAME stack (round 78-82):
- T = 1.0 (we don't use temperature scaling)
- H = -Σ g_mean log g_mean (empirical routing entropy, normalised
  by the natural log of K — this is a 0-th order approximation of
  the paper's gradient-based H but is sufficient for diagnostic
  purposes on toy data)
- O = 0.0 (we don't use oracle loss)
- B = λ (orthogonality) or η (φ-balancing) or 0 (plain)

**The paper's claim**: E ≥ 0.5 ⇒ no dead experts.
**Our extension**: E is also useful as a continuous diagnostic even
when the paper's threshold doesn't hold (e.g., our toy data).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def moe_ecology_number(
    router_logits: torch.Tensor,
    last_g: torch.Tensor,
    T: float = 1.0,
    H: float | None = None,
    O: float = 0.0,
    B: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute E = T·H/(O+B) — the MoE ecology diagnostic (Zhang 2026).

    Args:
        router_logits: [B, K] raw router logits (or any [B, K] tensor).
            Kept for API symmetry with the paper but not used directly
            in the empirical H approximation.
        last_g: [B, K] mixture weights (post top-K mask + softmax).
            Used to compute the empirical routing entropy H.
        T: Routing temperature.  Default 1.0 (no temperature scaling
            in our FAME stack).
        H: Routing entropy weight.  If ``None`` (default), computed
            empirically from ``last_g`` as ``-Σ g_mean log g_mean``,
            normalised by ``log(K)`` so the value is in [0, 1].
        O: Oracle weight.  Default 0.0 (no oracle loss in our stack).
        B: Balance weight — typically ``lambda_coeff`` (orthogonality)
            or ``phi_step_size`` (φ-balancing) or 0 (plain learned).
        eps: Numerical floor for log and denominator.

    Returns:
        Scalar ``E ∈ [0, ∞)``.  E ≥ 0.5 in the paper implies a healthy
        ecology with no dead experts.
    """
    K = last_g.shape[-1]
    if H is None:
        # Empirical routing entropy: H = -Σ g_mean log g_mean, normalised
        # by log(K) so it's in [0, 1].  When g is uniform, H = 1.
        g_mean = last_g.mean(dim=0).clamp_min(eps)  # [K]
        H_val = -(g_mean * torch.log(g_mean)).sum() / max(torch.log(torch.tensor(float(K))).item(), eps)
    else:
        H_val = float(H)
    denom = O + B
    return T * H_val / (denom + eps)


class MoEEcologyMonitor(nn.Module):
    """Track MoE cell health over training.

    Records per-expert utilization EMA, dead-expert count, and E
    trajectory.  Use ``step(g, B)`` to update; ``summary()`` for the
    current snapshot.

    Args:
        n_experts: K.
        dead_threshold: Per-expert utilization below this is considered
            "dead".  Default 0.01 (1% of routing mass).
        ema_alpha: EMA decay for utilization tracking.
    """

    def __init__(
        self,
        n_experts: int,
        dead_threshold: float = 0.01,
        ema_alpha: float = 0.01,
    ):
        super().__init__()
        assert n_experts >= 1
        self.n_experts = int(n_experts)
        self.dead_threshold = float(dead_threshold)
        self.ema_alpha = float(ema_alpha)
        # Per-expert utilization EMA.
        self.register_buffer(
            "util_ema", torch.full((self.n_experts,), 1.0 / self.n_experts)
        )
        # E trajectory (list of floats, capped at last 1000 entries).
        self.E_history: list[float] = []
        self.dead_history: list[int] = []

    @torch.no_grad()
    def step(
        self,
        g: torch.Tensor,
        T: float = 1.0,
        O: float = 0.0,
        B: float = 0.0,
    ) -> dict:
        """Update monitor with one batch's mixture weights.

        Args:
            g: [B, K] mixture weights (post top-K mask + softmax).
            T: Routing temperature.
            O: Oracle weight.
            B: Balance weight.

        Returns:
            Dict with current ``E``, ``dead_experts`` count, and
            ``utilization`` (per-expert EMA).
        """
        g_mean = g.mean(dim=0)  # [K]
        # Update utilization EMA.
        self.util_ema.mul_(1.0 - self.ema_alpha).add_(self.ema_alpha * g_mean)
        # Compute E using current state.
        E = moe_ecology_number(
            router_logits=g,
            last_g=g,
            T=T, H=None, O=O, B=B,
        )
        dead = int((self.util_ema < self.dead_threshold).sum().item())
        e_val = float(E.item())
        self.E_history.append(e_val)
        self.dead_history.append(dead)
        # Cap history at 1000 entries.
        if len(self.E_history) > 1000:
            self.E_history = self.E_history[-1000:]
            self.dead_history = self.dead_history[-1000:]
        return {"E": e_val, "dead_experts": dead, "utilization": self.util_ema.tolist()}

    def summary(self) -> dict:
        """Return current state (no step)."""
        dead = int((self.util_ema < self.dead_threshold).sum().item())
        return {
            "utilization": self.util_ema.tolist(),
            "dead_experts": dead,
            "E_last": self.E_history[-1] if self.E_history else float("nan"),
            "E_mean": float(sum(self.E_history) / len(self.E_history)) if self.E_history else float("nan"),
        }

    @torch.no_grad()
    def reset(self) -> None:
        """Reset EMA and history."""
        self.util_ema.copy_(torch.full((self.n_experts,), 1.0 / self.n_experts))
        self.E_history.clear()
        self.dead_history.clear()
