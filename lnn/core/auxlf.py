"""Round 106 — AuxLF: Auxiliary-Loss-Free Load Balancing (PRD #10-68).

Implements arXiv:2408.15664 (Wang et al. Aug 2024) *Auxiliary-Loss-Free
Load Balancing Strategy for Mixture-of-Experts* — the load-balancing
mechanism used in DeepSeek-V3. The idea: replace the standard auxiliary
loss with a per-expert **bias term** added to the routing scores before
top-K selection. The bias is updated based on recent expert load
**outside** the gradient computation, so it doesn't interfere with
training.

Key components:
- ``AuxLFRouter`` — top-k router with bias-adjusted scores
- ``update_load_balancing_bias`` — adjust bias based on recent load
- ``AuxLFSETAMoECfCCell`` — SETA cell with AuxLF on the unique router
- ``AuxLFSETAMoECfCNetwork`` — full network with pre-computed QuITE +
  SETA + AuxLF
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.quite_embedding import QueryIrregularEmbedding
from lnn.core.quite_moe import quite_context_pool
from lnn.core.seta_moe import SETAConfig, SETAMoECfCCell, SETARouter


@dataclass
class AuxLFConfig:
    """AuxLF configuration.

    Args:
        bias_lr: Learning rate for bias updates (typical: 0.001 to 0.1).
        target_load_fraction: Target fraction of tokens per expert
            (default 1/n_experts = uniform).
        bias_clamp: Maximum absolute bias value (prevents drift).
        warmup_steps: Number of forward passes before bias updates begin
            (lets gradients stabilize first).
        use_update: If True, bias is updated each forward. If False,
            bias is fixed (acts as a static prior).
    """
    bias_lr: float = 0.01
    target_load_fraction: float = -1.0  # -1 = 1/n_experts
    bias_clamp: float = 2.0
    warmup_steps: int = 10
    use_update: bool = True


def update_load_balancing_bias(
    bias: torch.Tensor,
    top_idx_counts: torch.Tensor,
    config: AuxLFConfig,
    n_experts: int,
) -> torch.Tensor:
    """Adjust per-expert bias based on recent load counts.

    Args:
        bias: (n_experts,) current bias values.
        top_idx_counts: (n_experts,) how many times each expert was
            selected in the recent forward pass.
        config: ``AuxLFConfig`` with hyperparameters.
        n_experts: Total number of experts (K).

    Returns:
        Updated bias tensor (same shape). Updates are in-place.
    """
    if not config.use_update:
        return bias
    target = config.target_load_fraction
    if target <= 0:
        target = 1.0 / n_experts
    # Expected count per expert: target * total
    total = float(top_idx_counts.sum().item())
    expected = target * total
    # diff > 0 means expert k is over-loaded → DECREASE its bias.
    # diff < 0 means expert k is under-loaded → INCREASE its bias.
    diff = top_idx_counts.float() - expected
    with torch.no_grad():
        bias.sub_(config.bias_lr * diff)
        if config.bias_clamp > 0:
            bias.clamp_(-config.bias_clamp, config.bias_clamp)
    return bias


class AuxLFRouter(SETARouter):
    """AuxLF-augmented top-k router for the unique expert subgroup.

    Inherits ``SETARouter`` (round 105) and adds a per-expert bias term
    to the routing scores before top-K. The bias is updated based on
    recent load counts via ``update_load_balancing_bias``.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension (H).
        d_context: QuITE context dim (0 disables).
        n_unique: Number of unique experts (U).
        top_k: How many of U to activate.
        auxlf_config: ``AuxLFConfig`` or dict.
        router_hidden: Optional 2-layer router MLP width.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        d_context: int,
        n_unique: int,
        top_k: int = 2,
        auxlf_config: AuxLFConfig | dict | None = None,
        router_hidden: int = 0,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            d_context=d_context,
            n_unique=n_unique,
            top_k=top_k,
            router_hidden=router_hidden,
        )
        if auxlf_config is None:
            auxlf_config = AuxLFConfig()
        if isinstance(auxlf_config, dict):
            self.auxlf_config = AuxLFConfig(**auxlf_config)
        else:
            self.auxlf_config = auxlf_config
        # Per-expert bias (initialized to 0)
        self.bias = nn.Parameter(torch.zeros(n_unique), requires_grad=False)
        # Side-channel: last top-K indices (for bias update)
        self.last_top_idx: torch.Tensor
        self.last_g: torch.Tensor
        # Diagnostic: load counts since last update
        self._load_counts = torch.zeros(n_unique)
        self._step_count = 0

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return top-k mixture weights with AuxLF bias adjustment.

        Args:
            x_t: (B, input_size) input.
            h: (B, hidden_size) previous hidden.
            context: (B, d_context) QuITE context (or None).

        Returns:
            g: (B, n_unique) softmax over biased top-K.
        """
        B = x_t.size(0)
        if context is None:
            context = torch.zeros(
                B, self.d_context, device=x_t.device, dtype=x_t.dtype,
            )
        combined = torch.cat([x_t, h, context], dim=-1)
        logits = self.net(combined)
        # AuxLF: add per-expert bias BEFORE top-K
        biased = logits + self.bias
        if self.top_k == self.n_unique:
            g = F.softmax(biased, dim=-1)
            self.last_top_idx = torch.arange(
                self.n_unique, device=biased.device,
            ).expand(B, -1)
        else:
            top_result = biased.topk(self.top_k, dim=-1)
            top_idx = top_result.indices
            del top_result
            mask = torch.full_like(biased, float("-inf"))
            mask.scatter_(-1, top_idx, 0.0)
            masked_logits = biased + mask
            g = F.softmax(masked_logits, dim=-1)
            self.last_top_idx = top_idx
        self.last_g = g.detach()
        # Update load counts (on device, will sync at end of sequence)
        with torch.no_grad():
            counts = torch.bincount(
                self.last_top_idx.flatten(), minlength=self.n_unique,
            ).float()
            self._load_counts = self._load_counts.to(counts.device) + counts
            self._step_count += 1
            # Auto-update bias if past warmup
            if (
                self.auxlf_config.use_update
                and self._step_count >= self.auxlf_config.warmup_steps
            ):
                self.update_bias_now()
        return g

    def update_bias_now(self) -> None:
        """Apply bias update using accumulated load counts.

        Resets the load counts after the update. Call this manually if
        you want to control the update cadence.
        """
        update_load_balancing_bias(
            self.bias, self._load_counts, self.auxlf_config, self.n_unique,
        )
        self._load_counts.zero_()
        self._step_count = 0

    def get_load_stats(self) -> Dict[str, float]:
        """Return current load statistics for diagnostics.

        Returns:
            Dict with keys: ``"util_per_expert"`` (list), ``"std"``,
            ``"max_min_ratio"``, ``"bias_per_expert"`` (list).
        """
        with torch.no_grad():
            counts = self._load_counts.cpu()
            if counts.sum() == 0:
                return {
                    "util_per_expert": [0.0] * self.n_unique,
                    "std": 0.0,
                    "max_min_ratio": 1.0,
                    "bias_per_expert": self.bias.cpu().tolist(),
                }
            util = (counts / counts.sum()).tolist()
            std = float(counts.std().item())
            nz = counts[counts > 0]
            if len(nz) > 1:
                ratio = float((counts.max() / counts.min()).item())
            else:
                ratio = 1.0
            return {
                "util_per_expert": util,
                "std": std,
                "max_min_ratio": ratio,
                "bias_per_expert": self.bias.cpu().tolist(),
            }


class AuxLFSETAMoECfCCell(SETAMoECfCCell):
    """SETA cell with AuxLF bias on the unique router.

    Args:
        Same as ``SETAMoECfCCell`` plus ``auxlf_config``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        sdta_config: SETAConfig | dict | None = None,
        d_context: int = 16,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        auxlf_config: AuxLFConfig | dict | None = None,
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
        # Replace the SETA router with AuxLFRouter
        U = self.n_unique
        self.router = AuxLFRouter(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            d_context=self.d_context,
            n_unique=U,
            top_k=self.top_k,
            auxlf_config=auxlf_config,
            router_hidden=router_hidden,
        )

    def collect_expert_utilization(self) -> Dict[str, float | int]:
        """Augment SETA's utilization with load-balance stats."""
        base = super().collect_expert_utilization()
        if isinstance(self.router, AuxLFRouter):
            load_stats = self.router.get_load_stats()
            base.update({
                "auxlf_util_std": load_stats["std"],
                "auxlf_max_min_ratio": load_stats["max_min_ratio"],
                "auxlf_bias_norm": float(
                    torch.norm(self.router.bias).item(),
                ),
            })
        return base


class AuxLFSETAMoECfCNetwork(nn.Module):
    """Full network wrapping QuITE + SETA + AuxLF.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension (H).
        sdta_config: ``SETAConfig``.
        auxlf_config: ``AuxLFConfig`` (or None to disable).
        n_queries, d_context, n_heads: QuITE params.
        output_size: Output feature dimension.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        sdta_config: SETAConfig | dict | None = None,
        auxlf_config: AuxLFConfig | dict | None = None,
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
        if auxlf_config is None:
            auxlf_config = AuxLFConfig()
        if isinstance(auxlf_config, dict):
            self.auxlf_config = AuxLFConfig(**auxlf_config)
        else:
            self.auxlf_config = auxlf_config
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
        # SETA + AuxLF cell
        self.cell = AuxLFSETAMoECfCCell(
            input_size=input_size,
            hidden_size=hidden_size,
            sdta_config=self.seta_config,
            d_context=d_context,
            n_tau_per_expert=n_tau_per_expert,
            tau_scales=tau_scales,
            auxlf_config=self.auxlf_config,
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
        return self.cell.regularization_loss()

    def get_utilization(self) -> Dict[str, float | int]:
        return self.cell.collect_expert_utilization()
