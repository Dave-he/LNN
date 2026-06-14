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

**Round 87 (PRD #10-49)**: added ``H_mode="gradient"`` to replace
the empirical H with a **gradient-based H** that measures the loss
sensitivity to routing.  This addresses arXiv:2606.10703 (Causal
Audit: observational ≠ causal).  Default remains
``H_mode="empirical"`` for back-compat.

**Round 88 (PRD #10-50)**: extends the gradient H from
**aggregated** to **per-expert**.  Adds
``H_mode="per_expert_gradient"`` (returns per-expert E as [K]
tensor) and ``per_expert_gradient_norms()`` function.  The
motivation is GRIN (arXiv:2409.12136): aggregated signals average
out per-expert pathologies, so a dead expert can be masked by
healthy ones in the aggregate.  Per-expert H_grad catches
per-expert collapse.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def gradient_routing_sensitivity(
    router_logits: torch.Tensor,
    task_loss: torch.Tensor | None,
    normalize: bool = True,
) -> float:
    """Compute H_grad = ||∂L_task/∂router_logits|| (Frobenius norm).

    Round 87 (PRD #10-49).  This is the **causal** counterpart to the
    empirical routing entropy H used in round 83.  Where empirical H
    asks "how uniform does the routing look?", gradient H asks
    "how sensitive is the loss to changes in the routing?".

    A small H_grad means the loss is **insensitive** to routing
    changes — the MoE has functionally collapsed (even if the
    routing distribution looks diverse).  This is the Causal Audit
    (arXiv:2606.10703) concern that observational E can mask causal
    collapse.

    Args:
        router_logits: [B, K] raw router logits (must require_grad
            for the gradient to be defined; returns 0.0 if not).
        task_loss: Scalar task loss.  If None, returns 0.0 (no
            gradient can be computed).
        normalize: If True, divide by B·log(K) for scale-invariance
            (so the value is roughly in [0, 1] when routing matters).

    Returns:
        Scalar H_grad ≥ 0.  Large ⇒ routing matters (healthy).  Small
        ⇒ routing doesn't matter (collapse imminent or already
        happened).
    """
    if task_loss is None or not router_logits.requires_grad:
        return 0.0
    try:
        grads = torch.autograd.grad(
            task_loss, router_logits,
            retain_graph=True, create_graph=False,
            allow_unused=True,
        )[0]
    except RuntimeError:
        return 0.0
    if grads is None:
        return 0.0
    h = float(grads.norm().item())
    if normalize:
        B = router_logits.shape[0]
        K = router_logits.shape[-1]
        h = h / max(B, 1) / max(float(np.log(max(K, 2))), 1e-8)
    return h


def per_expert_gradient_norms(
    router_logits: torch.Tensor,  # [B, K], requires_grad
    task_loss: torch.Tensor | None,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute per-expert gradient norms (round 88, PRD #10-50).

    For each expert k, compute ``||∂L_task/∂g_k||`` — the gradient
    norm of the loss with respect to expert k's gate logits, summed
    over the batch.  This is the **per-expert** counterpart to the
    aggregated ``gradient_routing_sensitivity`` (round 87).

    Where aggregated H_grad averages over experts (and can mask
    per-expert pathologies), per-expert H_grad exposes **which
    specific experts** are dead or alive.  An expert with all-zero
    gradient magnitude is functionally dead (its gate probability
    doesn't matter for the loss).

    Args:
        router_logits: [B, K] raw router logits (must require_grad
            for the gradient to be defined; returns zeros if not).
        task_loss: Scalar task loss.  Returns zeros if None.
        normalize: If True, divide by B for scale-invariance.

    Returns:
        [K] tensor of non-negative values, one per expert.  Large
        ⇒ expert k matters.  Small ⇒ expert k is functionally
        dead (its gate probability doesn't matter for the loss).
    """
    K = router_logits.shape[-1]
    if task_loss is None or not router_logits.requires_grad:
        return torch.zeros(K)
    try:
        grads = torch.autograd.grad(
            task_loss, router_logits,
            retain_graph=True, create_graph=False, allow_unused=True,
        )[0]
    except RuntimeError:
        return torch.zeros(K)
    if grads is None:
        return torch.zeros(K)
    # grads: [B, K]. Per-expert norm = ||g_b,k|| over batch dim.
    per_expert = grads.norm(dim=0)  # [K]
    if normalize:
        B = router_logits.shape[0]
        per_expert = per_expert / max(B, 1)
    return per_expert


def moe_ecology_number(
    router_logits: torch.Tensor,
    last_g: torch.Tensor,
    T: float = 1.0,
    H: float | None = None,
    O: float = 0.0,
    B: float = 0.0,
    eps: float = 1e-8,
    H_mode: str = "empirical",
    alpha: float = 0.5,
    task_loss: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute E = T·H/(O+B) — the MoE ecology diagnostic (Zhang 2026).

    Args:
        router_logits: [B, K] raw router logits (or any [B, K] tensor).
            Kept for API symmetry with the paper but not used directly
            in the empirical H approximation.  Used by ``H_mode="gradient"``
            and ``H_mode="blend"`` to compute H_grad.
        last_g: [B, K] mixture weights (post top-K mask + softmax).
            Used to compute the empirical routing entropy H.
        T: Routing temperature.  Default 1.0 (no temperature scaling
            in our FAME stack).
        H: Routing entropy weight.  If ``None`` (default), computed
            from ``last_g`` and/or ``task_loss`` per ``H_mode``.
        O: Oracle weight.  Default 0.0 (no oracle loss in our stack).
        B: Balance weight — typically ``lambda_coeff`` (orthogonality)
            or ``phi_step_size`` (φ-balancing) or 0 (plain learned).
        eps: Numerical floor for log and denominator.
        H_mode: ``"empirical"`` (default, round 83) uses
            ``-Σ g_mean log g_mean / log(K)``; ``"gradient"`` (round 87)
            uses ``||∂L_task/∂router_logits||``; ``"blend"`` uses
            ``alpha·H_emp + (1-alpha)·H_grad``;
            ``"per_expert_gradient"`` (round 88) returns per-expert
            E as a [K] tensor (per-expert gradient magnitude).
        alpha: Blend weight for ``H_mode="blend"`` (ignored otherwise).
        task_loss: Scalar task loss.  Required for
            ``H_mode="gradient"``, ``H_mode="blend"``, and
            ``H_mode="per_expert_gradient"``; silently falls back to
            empirical when None.

    Returns:
        Scalar ``E ∈ [0, ∞)`` for ``H_mode="empirical"|"gradient"|"blend"``.
        ``E ≥ 0.5`` in the paper implies a healthy ecology with no
        dead experts.

        Tensor ``[K]`` for ``H_mode="per_expert_gradient"`` —
        per-expert E values, one per expert.
    """
    K = last_g.shape[-1]
    if H is not None:
        H_val = float(H)  # user override
    elif H_mode == "empirical":
        # Empirical routing entropy: H = -Σ g_mean log g_mean, normalised
        # by log(K) so it's in [0, 1].  When g is uniform, H = 1.
        g_mean = last_g.mean(dim=0).clamp_min(eps)  # [K]
        H_val = -(g_mean * torch.log(g_mean)).sum() / max(torch.log(torch.tensor(float(K))).item(), eps)
    elif H_mode == "gradient":
        if task_loss is None:
            # Fall back to empirical if no task_loss (silent).
            g_mean = last_g.mean(dim=0).clamp_min(eps)
            H_val = -(g_mean * torch.log(g_mean)).sum() / max(torch.log(torch.tensor(float(K))).item(), eps)
        else:
            H_val = gradient_routing_sensitivity(router_logits, task_loss, normalize=True)
    elif H_mode == "blend":
        g_mean = last_g.mean(dim=0).clamp_min(eps)
        h_emp = float((-(g_mean * torch.log(g_mean)).sum() / max(torch.log(torch.tensor(float(K))).item(), eps)).item())
        h_grad = gradient_routing_sensitivity(router_logits, task_loss, normalize=True)
        H_val = alpha * h_emp + (1.0 - alpha) * h_grad
    elif H_mode == "per_expert_gradient":
        if task_loss is None:
            # Fall back to per-expert empirical (uniform H over experts).
            h_per_expert = torch.ones(K) / float(K)
        else:
            h_per_expert = per_expert_gradient_norms(
                router_logits, task_loss, normalize=True,
            )
        # Per-expert E_k = T · H_grad_k / (O + B), shape [K].
        denom = O + B
        return T * h_per_expert / (denom + eps)
    else:
        raise ValueError(
            f"H_mode must be 'empirical', 'gradient', 'blend', "
            f"or 'per_expert_gradient', got {H_mode!r}"
        )
    denom = O + B
    # Ensure H_val is a tensor (gradient mode returns a Python float).
    if not isinstance(H_val, torch.Tensor):
        H_val = torch.tensor(float(H_val))
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

    def compute_gradient_H(
        self,
        router_logits: torch.Tensor,
        task_loss: torch.Tensor,
        normalize: bool = True,
    ) -> dict:
        """Compute gradient-based H on demand (round 87, PRD #10-49).

        Args:
            router_logits: [B, K] raw router logits (must require_grad).
            task_loss: Scalar task loss.
            normalize: See ``gradient_routing_sensitivity``.

        Returns:
            Dict with ``H_grad`` (float), ``H_emp`` (float),
            ``E_emp``, ``E_grad``, and the recommended E (blend if
            ``H_mode="blend"`` else empirical).
        """
        K = router_logits.shape[-1]
        eps = 1e-8
        g_mean = router_logits.mean(dim=0).clamp_min(eps)
        h_emp = float(
            (-(g_mean * torch.log(g_mean)).sum()
             / max(torch.log(torch.tensor(float(K))).item(), eps)).item()
        )
        h_grad = gradient_routing_sensitivity(
            router_logits, task_loss, normalize=normalize,
        )
        return {
            "H_emp": h_emp,
            "H_grad": h_grad,
            "E_emp": h_emp,  # when B=0, E=H
            "E_grad": h_grad,
        }

    def per_expert_gradient_diagnostic(
        self,
        router_logits: torch.Tensor,
        task_loss: torch.Tensor,
        dead_grad_threshold: float = 1e-6,
    ) -> dict:
        """Per-expert gradient magnitude diagnostic (round 88, PRD #10-50).

        Identifies experts whose gate probabilities **don't matter**
        for the loss (i.e., functionally dead).  This is a
        per-expert refinement of the round 87 aggregated H_grad:
        while the aggregated H_grad averages over experts (and can
        mask per-expert pathologies), this method exposes
        **which specific experts** are dead.

        Args:
            router_logits: [B, K] raw router logits (requires_grad).
            task_loss: Scalar task loss.
            dead_grad_threshold: Per-expert gradient norm below this
                is considered "dead by gradient".  Default 1e-6.

        Returns:
            Dict with:
            - ``per_expert_grad``: [K] tensor of gradient norms
            - ``per_expert_grad_list``: [K] list of floats (JSON-safe)
            - ``dead_by_grad``: int count of dead experts
            - ``alive_by_grad``: list[int] of alive expert indices
            - ``dead_by_grad_indices``: list[int] of dead expert indices
            - ``max_grad``: float max per-expert gradient
            - ``min_grad``: float min per-expert gradient
            - ``max_min_ratio``: float max_grad / (min_grad + eps)
        """
        per_expert = per_expert_gradient_norms(
            router_logits, task_loss, normalize=True,
        )
        K = per_expert.shape[0]
        dead_mask = per_expert < dead_grad_threshold
        alive_mask = ~dead_mask
        max_g = float(per_expert.max().item()) if K > 0 else 0.0
        min_g = float(per_expert.min().item()) if K > 0 else 0.0
        eps = 1e-8
        ratio = max_g / (min_g + eps)
        return {
            "per_expert_grad": per_expert,
            "per_expert_grad_list": per_expert.tolist(),
            "dead_by_grad": int(dead_mask.sum().item()),
            "alive_by_grad": torch.where(alive_mask)[0].tolist(),
            "dead_by_grad_indices": torch.where(dead_mask)[0].tolist(),
            "max_grad": max_g,
            "min_grad": min_g,
            "max_min_ratio": float(ratio),
        }

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


# ---------------------------------------------------------------------------
# Round 90 (PRD #10-52): weights-vs-activations orthogonality audit
# (response to arXiv:2601.00457, Kim 2026).
# ---------------------------------------------------------------------------


def weight_space_overlap(expert_weights: list[torch.Tensor]) -> float:
    """Mean pairwise cosine similarity between flattened expert weight matrices.

    For a stack of K expert weight matrices W_0, ..., W_{K-1} (any shape,
    flattened to 1D), returns::

        (1 / (K*(K-1))) * sum_{i != j} |<W_i, W_j>| / (||W_i|| * ||W_j||)

    with sign-abs so that anti-parallel weights still count as "overlap".

    Returns 0.0 if K < 2.

    **Kim 2026 finding**: this metric INCREASES under weight-space
    geometric regularization (up to +114% in their experiments), the
    "disconnect" — minimizing the loss does not minimize this.
    """
    K = len(expert_weights)
    if K < 2:
        return 0.0
    flats = [w.detach().reshape(-1).to(torch.float32) for w in expert_weights]
    norms = torch.stack([f.norm() for f in flats])
    if torch.any(norms < 1e-12):
        # Treat zero-norm expert as having 0 overlap with all others.
        return 0.0
    sims = []
    for i in range(K):
        for j in range(i + 1, K):
            cos = (flats[i] * flats[j]).sum() / (norms[i] * norms[j])
            sims.append(cos.abs().item())
    return float(sum(sims) / len(sims))


def activation_space_overlap(expert_outs: list[torch.Tensor]) -> float:
    """Mean pairwise cosine similarity between per-expert activation tensors.

    ``expert_outs[k]`` has shape (B, T, D) (or (N, D) for unbatched).
    We flatten all non-expert dims, then compute the same pairwise
    absolute-cosine as :func:`weight_space_overlap`.

    Returns 0.0 if K < 2.

    **Kim 2026 finding**: this metric stays at ~0.6 across weight-
    space orth strengths (no effect), the "disconnect" target.

    **Our claim (round 80)**: our ``orthogonality_loss`` operates
    on these activations directly, so this metric SHOULD decrease
    under our orth loss — by construction.
    """
    K = len(expert_outs)
    if K < 2:
        return 0.0
    flats = [h.detach().reshape(-1).to(torch.float32) for h in expert_outs]
    norms = torch.stack([f.norm() for f in flats])
    if torch.any(norms < 1e-12):
        return 0.0
    sims = []
    for i in range(K):
        for j in range(i + 1, K):
            cos = (flats[i] * flats[j]).sum() / (norms[i] * norms[j])
            sims.append(cos.abs().item())
    return float(sum(sims) / len(sims))
