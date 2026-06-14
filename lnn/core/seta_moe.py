"""Round 105 — SETA: Sparse Shared + Unique Experts (PRD #10-67).

Implements arXiv:2606.07500 (Siddika, Hossen, Mallick, Jannesari,
June 2026) *SETA: Sparse Subspace-to-Expert Sharing for Task-Agnostic
Continual Learning*, re-interpreted for time-series MoE.

The paper decomposes K experts into:
  - S = n_shared: SHARED experts, always active, output averaged
  - U = n_unique: UNIQUE experts, top-k routed among themselves

SETA's two regularizers:
  - Adaptive elastic anchoring: penalize shared expert weight drift
    from EMA-snapshotted anchors (preserves "common knowledge")
  - Routing-aware regularization: keep unique router entropy near a
    target (avoid the H=0 lock-in we observed in rounds 103-104)

Functions and classes:
- ``SETAConfig`` — dataclass with S, U, top_k, anchoring, routing reg
- ``elastic_anchoring_loss`` — L2 between current and anchor weights
- ``routing_regularization`` — penalize H deviation from target
- ``SETARouter`` — top-k router for the unique expert subgroup
- ``SETAMoECfCCell`` — S shared + U unique CfC experts
- ``SETAMoECfCNetwork`` — full network with pre-computed QuITE + SETA
- ``update_ema_anchors`` — exponential moving average of shared weights
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.quite_embedding import QueryIrregularEmbedding
from lnn.core.quite_moe import quite_context_pool


@dataclass
class SETAConfig:
    """SETA configuration.

    Args:
        n_shared: Number of always-active shared experts (S >= 1).
        n_unique: Number of unique experts to be top-k routed (U >= 1).
        top_k: How many of the U unique experts to activate per step.
        elastic_lambda: Coefficient for the elastic anchoring loss.
        routing_lambda: Coefficient for the routing-aware regularization.
        target_routing_entropy: Target entropy (nats) for the unique
            router. The default ``log(top_k)`` is a sensible choice.
        use_ema_anchor: If True, anchor weights are EMA of recent
            weights; if False, anchors are the initial weights.
        ema_decay: EMA decay rate (only used if use_ema_anchor).
    """
    n_shared: int = 2
    n_unique: int = 3
    top_k: int = 2
    elastic_lambda: float = 1e-3
    routing_lambda: float = 1e-2
    target_routing_entropy: float = -1.0  # -1 = use log(top_k)
    use_ema_anchor: bool = True
    ema_decay: float = 0.99

    def __post_init__(self):
        if self.n_shared < 1:
            raise ValueError(f"n_shared must be >= 1, got {self.n_shared}")
        if self.n_unique < 1:
            raise ValueError(f"n_unique must be >= 1, got {self.n_unique}")
        if not (1 <= self.top_k <= self.n_unique):
            raise ValueError(
                f"top_k must be in [1, n_unique] = [1, {self.n_unique}], "
                f"got {self.top_k}",
            )


def elastic_anchoring_loss(
    shared_experts: nn.ModuleList,
    anchor_state: Dict[str, torch.Tensor],
    lambda_val: float = 1e-3,
) -> torch.Tensor:
    """L2 distance between current shared expert weights and anchors.

    Args:
        shared_experts: ModuleList of shared expert modules.
        anchor_state: Dict mapping ``"expert_i.param_name"`` → tensor.
            Use ``snapshot_expert_weights(shared_experts)`` to build this
            at training time, or ``update_ema_anchors`` to maintain it.
        lambda_val: Coefficient applied to the L2 sum.

    Returns:
        Scalar loss: ``lambda_val * sum_i ||theta_i - theta_i^anchor||^2``.
    """
    loss = torch.tensor(0.0, device=next(shared_experts.parameters()).device)
    n = 0
    for i, expert in enumerate(shared_experts):
        for name, p in expert.named_parameters():
            key = f"expert_{i}.{name}"
            if key not in anchor_state:
                # Anchor missing — use 0 (still produces a valid loss).
                anchor = torch.zeros_like(p)
            else:
                anchor = anchor_state[key]
            loss = loss + ((p - anchor) ** 2).sum()
            n += 1
    if lambda_val != 1.0:
        loss = loss * lambda_val
    return loss


def routing_regularization(
    router: "SETARouter",
    target_entropy: float,
    lambda_val: float = 1e-2,
) -> torch.Tensor:
    """Penalize deviation of unique-router entropy from target.

    Args:
        router: The unique expert router (with ``last_g`` populated).
        target_entropy: Target entropy in nats. Use ``math.log(top_k)``
            for a balanced router.
        lambda_val: Coefficient applied to the squared-deviation loss.

    Returns:
        Scalar loss: ``lambda_val * (H_actual - target)^2``.
    """
    if not hasattr(router, "last_g"):
        return torch.tensor(0.0)
    g = router.last_g  # (B, top_k) — already softmaxed
    if g is None or g.numel() == 0:
        return torch.tensor(0.0)
    p_safe = g.clamp(min=1e-8)
    # 0 * log(0) is NaN; replace NaN with 0 contribution
    terms = p_safe * p_safe.log()
    terms = torch.where(torch.isfinite(terms), terms, torch.zeros_like(terms))
    H_actual = -terms.sum(dim=-1).mean()
    dev = H_actual - float(target_entropy)
    return lambda_val * dev * dev


def snapshot_expert_weights(shared_experts: nn.ModuleList) -> Dict[str, torch.Tensor]:
    """Snapshot current shared expert weights as a fresh anchor dict.

    Returns:
        Dict with keys ``"expert_i.<param_name>"`` and tensor values.
    """
    snap: Dict[str, torch.Tensor] = {}
    for i, expert in enumerate(shared_experts):
        for name, p in expert.named_parameters():
            snap[f"expert_{i}.{name}"] = p.detach().clone()
    return snap


def update_ema_anchors(
    current_anchors: Dict[str, torch.Tensor],
    shared_experts: nn.ModuleList,
    decay: float = 0.99,
) -> Dict[str, torch.Tensor]:
    """Update EMA anchors in-place: ``a <- decay * a + (1 - decay) * p``.

    Args:
        current_anchors: Existing anchor dict (or {} to initialize).
        shared_experts: The shared experts ModuleList.
        decay: EMA decay (0.99 = anchors move slowly toward current).

    Returns:
        The same dict (updated in place).
    """
    for i, expert in enumerate(shared_experts):
        for name, p in expert.named_parameters():
            key = f"expert_{i}.{name}"
            new_val = p.detach().clone()
            if key in current_anchors:
                current_anchors[key].mul_(decay).add_(new_val, alpha=1.0 - decay)
            else:
                current_anchors[key] = new_val
    return current_anchors


class SETARouter(nn.Module):
    """Top-k router for the UNIQUE expert subgroup.

    Same architecture as ``QuiteRouter`` (round 103) but operates on
    only the unique experts.  This separates the always-active shared
    path from the sparse top-k unique path.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension (H).
        d_context: Dimension of the QuITE context vector (0 = no context).
        n_unique: Number of unique experts (U).
        top_k: How many of the U unique experts to activate.
        router_hidden: Width of an optional 2-layer router MLP.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        d_context: int,
        n_unique: int,
        top_k: int = 2,
        router_hidden: int = 0,
    ) -> None:
        super().__init__()
        assert n_unique >= 1
        assert 1 <= top_k <= n_unique
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.d_context = int(d_context)
        self.n_unique = int(n_unique)
        self.top_k = int(top_k)
        self.router_hidden = int(router_hidden)
        router_in = input_size + hidden_size + d_context
        if self.router_hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(router_in, self.router_hidden),
                nn.Tanh(),
                nn.Linear(self.router_hidden, self.n_unique),
            )
        else:
            self.net = nn.Linear(router_in, self.n_unique)
        self.last_top_idx: torch.Tensor
        self.last_g: torch.Tensor

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return top-k sparse mixture weights for unique experts.

        Args:
            x_t: (B, input_size) input at this step.
            h: (B, hidden_size) previous hidden state.
            context: (B, d_context) QuITE context, or None to zero-fill.

        Returns:
            g: (B, n_unique) mixture weights with exactly top_k
                non-zero entries per row.
        """
        B = x_t.size(0)
        if context is None:
            context = torch.zeros(
                B, self.d_context, device=x_t.device, dtype=x_t.dtype,
            )
        combined = torch.cat([x_t, h, context], dim=-1)
        logits = self.net(combined)
        if self.top_k == self.n_unique:
            g = F.softmax(logits, dim=-1)
            self.last_top_idx = torch.arange(
                self.n_unique, device=logits.device,
            ).expand(B, -1)
            self.last_g = g.detach()
            return g
        top_result = logits.topk(self.top_k, dim=-1)
        top_idx = top_result.indices
        del top_result
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(-1, top_idx, 0.0)
        masked_logits = logits + mask
        g = F.softmax(masked_logits, dim=-1)
        self.last_top_idx = top_idx
        self.last_g = g.detach()
        return g


class SETAMoECfCCell(nn.Module):
    """SETA: S shared + U unique CfC experts (PRD #10-67).

    The shared experts (S) are ALWAYS active — their outputs are
    averaged to form ``shared_out``.  The unique experts (U) are
    top-k routed via ``SETARouter`` and their weighted sum forms
    ``unique_out``.  Final output is ``shared_out + unique_out``.

    The shared experts act as a **structural baseline** of multi-expert
    utilization, addressing the H=0 lock-in observed in rounds 103-104.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension (H).
        sdta_config: ``SETAConfig`` (or a plain dict with the same keys).
        d_context: QuITE context dim (0 disables context).
        n_tau_per_expert: Per-expert ``n_tau`` (round 76 compatibility).
        tau_scales: Per-branch initial time constants.
        router_hidden: Width of the optional 2-layer unique-router MLP.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        sdta_config: SETAConfig | dict | None = None,
        d_context: int = 16,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        router_hidden: int = 0,
    ) -> None:
        super().__init__()
        if sdta_config is None:
            sdta_config = SETAConfig()
        if isinstance(sdta_config, dict):
            self.config = SETAConfig(**sdta_config)
        else:
            self.config = sdta_config
        S = self.config.n_shared
        U = self.config.n_unique
        # Validation already done in SETAConfig.__post_init__

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_shared = int(S)
        self.n_unique = int(U)
        self.n_experts = S + U  # for compatibility with bench scripts
        self.top_k = int(self.config.top_k)
        self.d_context = int(d_context)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.router_hidden = int(router_hidden)

        # Shared experts — ALWAYS active
        self.shared_experts = nn.ModuleList(
            [
                CfCCell(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    n_tau=self.n_tau_per_expert,
                    tau_scales=self.tau_scales,
                )
                for _ in range(S)
            ]
        )
        # Unique experts — top-k routed
        self.unique_experts = nn.ModuleList(
            [
                CfCCell(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    n_tau=self.n_tau_per_expert,
                    tau_scales=self.tau_scales,
                )
                for _ in range(U)
            ]
        )
        # Unique router (top-k)
        self.router = SETARouter(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            d_context=self.d_context,
            n_unique=U,
            top_k=self.config.top_k,
            router_hidden=self.router_hidden,
        )
        # EMA anchors (initialized as init weights)
        self.register_buffer("_ema_initialized", torch.tensor(False))
        self._ema_anchors: Dict[str, torch.Tensor] = {}
        # Side-channel
        self.last_g_unique: torch.Tensor
        self.last_shared_stack: torch.Tensor
        self.last_unique_stack: torch.Tensor

    @property
    def target_entropy(self) -> float:
        if self.config.target_routing_entropy > 0:
            return float(self.config.target_routing_entropy)
        return math.log(self.top_k)

    def __post_init__(self):
        # Hard validation (raises ValueError, not assertion)
        if self.n_shared < 1:
            raise ValueError(f"n_shared must be >= 1, got {self.n_shared}")
        if self.n_unique < 1:
            raise ValueError(f"n_unique must be >= 1, got {self.n_unique}")
        if not (1 <= self.top_k <= self.n_unique):
            raise ValueError(
                f"top_k must be in [1, n_unique] = [1, {self.n_unique}], "
                f"got {self.top_k}",
            )

    def init_anchors(self) -> None:
        """Snapshot current shared expert weights as the initial anchor."""
        self._ema_anchors = snapshot_expert_weights(self.shared_experts)
        self._ema_initialized.fill_(True)

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        context: torch.Tensor | None = None,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One step of SETA sparse-shared-unique MoE.

        Args:
            x_t: (B, input_size) input at this step.
            h: (B, hidden_size) previous hidden state.
            context: (B, d_context) QuITE context (or None to zero-fill).
            dt: scalar or [B] per-sample time delta.

        Returns:
            h_new: (B, hidden_size) = shared_mean + unique_top_k_weighted.
        """
        # Always-active shared experts
        shared_outs = [expert(x_t, h, dt=dt) for expert in self.shared_experts]
        shared_stack = torch.stack(shared_outs, dim=1)  # (B, S, H)
        self.last_shared_stack = shared_stack.detach()
        shared_mean = shared_stack.mean(dim=1)  # (B, H)
        # Top-k routed unique experts
        unique_outs = [expert(x_t, h, dt=dt) for expert in self.unique_experts]
        unique_stack = torch.stack(unique_outs, dim=1)  # (B, U, H)
        self.last_unique_stack = unique_stack.detach()
        g = self.router(x_t, h, context=context)  # (B, U) — sparse
        self.last_g_unique = g.detach()
        unique_contrib = (g.unsqueeze(-1) * unique_stack).sum(dim=1)  # (B, H)
        return shared_mean + unique_contrib

    def regularization_loss(self) -> torch.Tensor:
        """Combined SETA regularization: elastic anchoring + routing reg.

        Returns:
            Scalar loss to be ADDED to the task loss.  Call after each
            forward pass so ``router.last_g`` is populated.
        """
        # Lazy init anchors on first call
        if not bool(self._ema_initialized.item()):
            self.init_anchors()
        elif self.config.use_ema_anchor:
            # Refresh EMA
            update_ema_anchors(
                self._ema_anchors, self.shared_experts, decay=self.config.ema_decay,
            )
        e_loss = elastic_anchoring_loss(
            self.shared_experts,
            self._ema_anchors,
            lambda_val=self.config.elastic_lambda,
        )
        r_loss = routing_regularization(
            self.router,
            self.target_entropy,
            lambda_val=self.config.routing_lambda,
        )
        return e_loss + r_loss

    def collect_expert_utilization(
        self,
    ) -> Dict[str, float | int]:
        """Snapshot utilization of shared and unique experts (diagnostic).

        Returns:
            Dict with keys ``"shared_entropy"``, ``"unique_entropy"``,
            ``"unique_top_idx"``, ``"unique_dead"``.
        """
        with torch.no_grad():
            # Shared entropy is meaningless (always active) — report 1
            shared_H = math.log(max(self.n_shared, 1))
            # Unique entropy from router last_g
            g = self.last_g_unique
            if g is None or g.numel() == 0:
                unique_H = 0.0
            else:
                p = g.clamp(min=1e-8)
                unique_H = float(-(p * p.log()).sum(dim=-1).mean().item())
            return {
                "shared_entropy": shared_H,
                "unique_entropy": unique_H,
                "shared_n_active": self.n_shared,
                "unique_n_active": int((g > 1e-6).sum(dim=-1).float().mean().item()) if g is not None else 0,
            }


class SETAMoECfCNetwork(nn.Module):
    """Full network wrapping QuITE embedding with SETA cell.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension (H).
        sdta_config: ``SETAConfig``.
        n_queries: QuITE learnable query tokens (round 102).
        d_context: QuITE context dim (must match cell d_context).
        n_heads: QuITE attention heads.
        output_size: Output feature dimension.
        pool_method: How to pool QuITE tokens ('mean', 'max', 'first').
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        sdta_config: SETAConfig | dict | None = None,
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
            self.config = SETAConfig(**sdta_config)
        else:
            self.config = sdta_config
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.pool_method = str(pool_method)
        self.d_context = int(d_context)
        # QuITE module (round 102) — must be at __init__ so Adam finds params
        self.quite = QueryIrregularEmbedding(
            d_input=input_size,
            n_queries=n_queries,
            d_model=d_context,
            n_heads=n_heads,
        )
        # SETA cell
        self.cell = SETAMoECfCCell(
            input_size=input_size,
            hidden_size=hidden_size,
            sdta_config=self.config,
            d_context=d_context,
            n_tau_per_expert=n_tau_per_expert,
            tau_scales=tau_scales,
        )
        # Output head
        self.head = nn.Linear(hidden_size, output_size)
        # Cached context
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
        h = torch.zeros(B, self.hidden_size, device=observations.device, dtype=observations.dtype)
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
        """Aggregate SETA regularizers from the cell."""
        return self.cell.regularization_loss()

    def get_utilization(self) -> Dict[str, float | int]:
        return self.cell.collect_expert_utilization()
