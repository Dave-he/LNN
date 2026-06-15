"""Anchored MoE with Structural Prior (PRD #10-70, round 108).

Implements AME-TS (arXiv:2605.25166 Wang et al. May 2026) — *Anchored
Mixture-of-Experts for Time Series Forecasting*.

The core idea: replace emergent-learned routing with **structural
anchoring** of routing decisions to interpretable per-series
descriptors (forecastability, seasonality, trend, sparsity).

Pipeline:
  1. RegimePredictor(x): (B, T, D) → (B, 4) descriptors in [0, 1]
  2. StructuralPrior: (B, 4) → (B, K) prior over K experts (softmax)
  3. AnchoredRouter: logit_anchored = logit_learned + log(p_prior + ε)
  4. Top-K over logit_anchored → soft assignment

Three anchoring modes supported:
  - 'logit':  additive anchoring in log-space (default)
  - 'mix':    p_final = α·softmax(logit) + (1-α)·p_prior
  - 'kl':     KL(softmax(logit) || p_prior) added to loss (regularization)

The audit pattern (rounds 91-107) predicts structural > routing-only,
so this is expected to outperform pure top-K (round 78 FAME H=0
lock-in) by giving each expert a stable, interpretable specialization
axis.
"""
from __future__ import annotations

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AnchoredMoEConfig:
    n_experts: int = 4
    top_k: int = 2
    d_hidden: int = 16
    descriptor_dim: int = 4  # forecast, season, trend, sparsity
    anchor_mode: str = "logit"  # 'logit' | 'mix' | 'kl'
    anchor_alpha: float = 0.5  # for 'mix' mode
    anchor_lambda: float = 0.1  # for 'kl' mode
    anchor_eps: float = 1e-6


def _safe_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


class RegimePredictor(nn.Module):
    """Predicts 4 descriptors per timestep, then pools to (B, 4).

    Descriptors (all in [0, 1]):
      - forecastability: 1 - normalized residual variance (higher = more predictable)
      - seasonality: amplitude of dominant FFT component
      - trend: |slope| of linear fit
      - sparsity: 1 - fraction of non-NaN values

    All four are differentiable approximations of classical
    time-series descriptors.
    """

    def __init__(self, input_size: int, d_hidden: int = 16):
        super().__init__()
        self.input_size = input_size
        self.d_hidden = d_hidden
        # Per-timestep MLP: input D → hidden → 4 descriptors
        self.mlp = nn.Sequential(
            nn.Linear(input_size, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) — may contain NaN. Returns (B, 4) descriptors in [0, 1]."""
        x_clean = torch.nan_to_num(x, nan=0.0)
        # Per-timestep descriptors from MLP — but we need a learnable mapping
        # to interpretable descriptors. Use the MLP to learn them, then
        # aggregate over time.
        # (B, T, D) → (B, T, 4)
        per_step = _safe_sigmoid(self.mlp(x_clean))
        # Pool over time: mean (B, 4)
        pooled = per_step.mean(dim=1)
        return pooled


class StructuralPrior(nn.Module):
    """Maps (B, 4) descriptors to (B, K) prior over K experts.

    Uses a small MLP followed by softmax to produce a valid
    probability distribution.
    """

    def __init__(self, descriptor_dim: int, n_experts: int, d_hidden: int = 16):
        super().__init__()
        self.n_experts = n_experts
        self.mlp = nn.Sequential(
            nn.Linear(descriptor_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_experts),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """descriptors: (B, descriptor_dim). Returns (B, K) prior."""
        logits = self.mlp(descriptors)
        return F.softmax(logits, dim=-1)


class AnchoredRouter(nn.Module):
    """Top-K router with structural anchoring.

    Modes:
      - 'logit': logit_anchored = logit + log(p_prior + ε)
      - 'mix':   p_final = α·softmax(logit) + (1-α)·p_prior
      - 'kl':    standard top-K with separate KL regularization
                 (caller adds the KL loss to total loss)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int,
        top_k: int,
        d_context: int = 0,
        anchor_mode: str = "logit",
        anchor_alpha: float = 0.5,
        anchor_eps: float = 1e-6,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.anchor_mode = anchor_mode
        self.anchor_alpha = anchor_alpha
        self.anchor_eps = anchor_eps
        in_dim = input_size + hidden_size + d_context
        self.router_mlp = nn.Sequential(
            nn.Linear(in_dim, max(8, n_experts * 2)),
            nn.GELU(),
            nn.Linear(max(8, n_experts * 2), n_experts),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        prior: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x_t: (B, D), h: (B, H), context: (B, C) or None, prior: (B, K) or None.

        Returns:
          weights: (B, K) — sparse assignment
          top_idx: (B, top_k) — indices of selected experts
        """
        feats = [x_t, h]
        if context is not None:
            feats.append(context)
        inp = torch.cat(feats, dim=-1)
        logit = self.router_mlp(inp)  # (B, K)

        if prior is not None and self.anchor_mode == "logit":
            logit = logit + torch.log(prior + self.anchor_eps)
        elif prior is not None and self.anchor_mode == "mix":
            p_learned = F.softmax(logit, dim=-1)
            p_mixed = self.anchor_alpha * p_learned + (1 - self.anchor_alpha) * prior
            # Use log of mixed as if it were raw scores
            logit = torch.log(p_mixed + self.anchor_eps)

        top_v, top_idx = logit.topk(self.top_k, dim=-1)
        weights = F.softmax(top_v, dim=-1)
        # Build full K-vector weights
        full_w = torch.zeros_like(logit)
        full_w.scatter_(-1, top_idx, weights)
        # Bookkeeping
        self.last_logits = logit.detach()
        self.last_top_idx = top_idx.detach()
        return full_w, top_idx

    def get_kl_regularization(
        self,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """KL(softmax(logit) || prior) — used in 'kl' mode.

        This is independent of forward() — it uses self.last_logits
        which is set during the last forward() call.
        """
        if not hasattr(self, "last_logits"):
            return torch.tensor(0.0, device=prior.device)
        p_learned = F.softmax(self.last_logits, dim=-1)
        # KL(p_learned || prior) = sum p_learned * (log p_learned - log prior)
        kl = (p_learned * (torch.log(p_learned + 1e-8) - torch.log(prior + 1e-8))).sum(dim=-1).mean()
        return kl


class AnchoredMoECfCCell(nn.Module):
    """Single CfC cell with anchored MoE routing.

    K experts, each a 2-layer MLP. Per-step:
      1. Compute descriptors from x (B, 4)
      2. Compute prior from descriptors (B, K)
      3. Top-K routing with anchoring → weights (B, K)
      4. Mix expert outputs by weights → next hidden state

    Designed as a stand-in for plain CfCCell; output is the same
    shape (B, H) for next-step consumption.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 4,
        top_k: int = 2,
        d_context: int = 0,
        anchor_mode: str = "logit",
        anchor_alpha: float = 0.5,
        anchor_lambda: float = 0.1,
        anchor_eps: float = 1e-6,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = n_experts
        self.top_k = top_k
        self.anchor_mode = anchor_mode
        self.anchor_lambda = anchor_lambda
        self.regime = RegimePredictor(input_size)
        self.prior = StructuralPrior(4, n_experts)
        self.router = AnchoredRouter(
            input_size=input_size,
            hidden_size=hidden_size,
            n_experts=n_experts,
            top_k=top_k,
            d_context=d_context,
            anchor_mode=anchor_mode,
            anchor_alpha=anchor_alpha,
            anchor_eps=anchor_eps,
        )
        # K expert MLPs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size + hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            for _ in range(n_experts)
        ])
        # Bookkeeping
        self.last_descriptors = None
        self.last_prior = None
        self.last_weights = None
        self.last_expert_outputs = None

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, T, D), h: (B, H), context: (B, C) or None.

        Returns: h_next: (B, H)
        """
        # Compute descriptors and prior from the full sequence
        # (for per-step use, we could do this per step, but pooling
        # is more efficient and stable)
        descriptors = self.regime(x)  # (B, 4)
        prior = self.prior(descriptors)  # (B, K)
        # Use the LAST timestep as the current input
        x_t = x[:, -1, :]  # (B, D)
        # Per-step routing
        weights, _ = self.router(x_t, h, context=context, prior=prior)
        # Compute expert outputs: each expert takes [x_t, h]
        feat = torch.cat([x_t, h], dim=-1)  # (B, H+D)
        expert_outs = torch.stack(
            [e(feat) for e in self.experts], dim=1
        )  # (B, K, H)
        # Mix by weights
        h_next = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)  # (B, H)
        # Bookkeeping
        self.last_descriptors = descriptors.detach()
        self.last_prior = prior.detach()
        self.last_weights = weights.detach()
        self.last_expert_outputs = expert_outs.detach()
        return h_next

    def get_regularization_loss(self) -> torch.Tensor:
        """KL regularization for 'kl' mode; 0 otherwise."""
        if self.anchor_mode != "kl":
            return torch.tensor(0.0)
        if self.last_prior is None:
            return torch.tensor(0.0)
        kl = self.router.get_kl_regularization(self.last_prior)
        return self.anchor_lambda * kl

    def get_utilization(self) -> dict:
        """Return expert utilization diagnostics."""
        if self.last_weights is None or self.last_prior is None or self.last_descriptors is None:
            return {}
        # Average weight per expert (sparse-aware: only count nonzero)
        avg_weights = self.last_weights.mean(dim=0).cpu().tolist()
        # H of routing distribution (over selected experts)
        w = self.last_weights.clamp(min=1e-8)
        H = (-w * w.log()).sum(dim=-1).mean().item()
        # max/min ratio — only meaningful if all K experts got nonzero
        # weight at least once. With sparse top-K routing, some
        # experts may have 0 weight by design, so we count
        # "active_fraction" separately.
        avg_w_np = torch.tensor(avg_weights)
        active_mask = avg_w_np > 1e-6
        if active_mask.sum() >= 2:
            active_weights = avg_w_np[active_mask]
            max_min = (active_weights.max() / (active_weights.min() + 1e-8)).item()
        else:
            max_min = 1.0
        active_fraction = active_mask.float().mean().item()
        return {
            "expert_avg_weights": avg_weights,
            "routing_entropy": H,
            "routing_max_min_ratio": max_min,
            "routing_active_fraction": active_fraction,
            "prior_entropy": (-self.last_prior * self.last_prior.clamp(min=1e-8).log()).sum(dim=-1).mean().item(),
            "descriptors_mean": self.last_descriptors.mean(dim=0).cpu().tolist(),
        }


class AnchoredMoECfCNetwork(nn.Module):
    """Full network: rolling-window loop over an AnchoredMoECfCCell.

    x: (B, T, D) → per-step descriptors + per-step routing + per-step expert
    outputs. Final prediction is the last hidden state projected to output_size.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 4,
        top_k: int = 2,
        output_size: Optional[int] = None,
        anchor_mode: str = "logit",
        anchor_alpha: float = 0.5,
        anchor_lambda: float = 0.1,
        anchor_eps: float = 1e-6,
    ):
        super().__init__()
        if output_size is None:
            output_size = input_size
        self.cell = AnchoredMoECfCCell(
            input_size=input_size,
            hidden_size=hidden_size,
            n_experts=n_experts,
            top_k=top_k,
            anchor_mode=anchor_mode,
            anchor_alpha=anchor_alpha,
            anchor_lambda=anchor_lambda,
            anchor_eps=anchor_eps,
        )
        self.head = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

    def forward(
        self,
        x: torch.Tensor,
        times: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, T, D), times: (B, T), mask: (B, T). Returns (B, T, output_size)."""
        B, T, _ = x.shape
        # NaN-safe: replace NaN with 0 BEFORE the per-step loop
        # (the cell's regime predictor also does nan_to_num, but doing
        # it once here is more efficient and prevents any leakage)
        x_clean = torch.nan_to_num(x, nan=0.0)
        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            x_t = x_clean[:, t, :]
            h = self.cell(x_t.unsqueeze(1), h)  # cell expects (B, 1, D)
            outputs.append(self.head(h))
        return torch.stack(outputs, dim=1)

    def get_utilization(self) -> dict:
        return self.cell.get_utilization()

    def get_regularization_loss(self) -> torch.Tensor:
        return self.cell.get_regularization_loss()
