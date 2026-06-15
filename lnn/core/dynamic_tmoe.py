"""Round 109 — Drift-Aware Dynamic MoE (PRD #10-71).

Implements Dynamic TMoE (arXiv:2605.20678 Zhu, Liu, Weng, Wu — May 2026,
ICML 2026) — *Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts
Framework for Non-Stationary Time Series Forecasting*.

Three structural mechanisms:
1. **MMD drift detector** — Maximum Mean Discrepancy between two
   consecutive windows; fires when distribution shifts.
2. **Dynamic expert pool** — expert pool grows (on drift) or prunes
   (on redundancy); size is no longer fixed.
3. **Temporal memory router** — recurrent state + anomaly repository
   for context-aware expert selection, NO test-time updates.

Audit prediction (rounds 91-108): **structural > routing-only**. This
is the most structural fix yet — it changes the expert pool itself.
Should be the 6th structural winner if the implementation is correct.

Key components:
- ``mmd_rbf`` — Maximum Mean Discrepancy with Gaussian RBF kernel
- ``DriftDetector`` — sliding-window MMD detector with threshold
- ``DynamicExpertPool`` — list of experts, add/prune operations
- ``TemporalMemoryRouter`` — recurrent router with anomaly memory
- ``DynamicTMoECfCCell`` — K experts + drift detection + dynamic add/prune
- ``DynamicTMoECfCNetwork`` — full network with rolling loop
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# MMD drift detection
# ----------------------------------------------------------------------------


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Maximum Mean Discrepancy with Gaussian RBF kernel.

    MMD^2(P, Q) = E[k(x, x')] + E[k(y, y')] - 2·E[k(x, y)]

    Args:
        x: (N, D) samples from distribution P
        y: (M, D) samples from distribution Q
        sigma: RBF bandwidth. If 0, use median heuristic.

    Returns:
        scalar MMD^2 (>= 0)
    """
    # Replace NaN with 0 (treating missing as "no signal")
    x = torch.nan_to_num(x, nan=0.0)
    y = torch.nan_to_num(y, nan=0.0)

    n = x.size(0)
    m = y.size(0)

    if n == 0 or m == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    # Median heuristic for sigma if not provided
    if sigma <= 0:
        xy = torch.cat([x, y], dim=0)
        dists = torch.cdist(xy, xy)
        sigma = float(dists.median().item())
        if sigma < 1e-6:
            sigma = 1.0

    def rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(a, b)
        return torch.exp(-(d ** 2) / (2 * sigma ** 2 + 1e-8))

    kxx = rbf(x, x).mean()
    kyy = rbf(y, y).mean()
    kxy = rbf(x, y).mean()

    return kxx + kyy - 2 * kxy


class DriftDetector(nn.Module):
    """Sliding-window MMD-based drift detector.

    Maintains a rolling reference window. When new samples arrive,
    compute MMD(reference, new). If MMD > threshold, drift is detected.

    Args:
        window_size: Number of samples to keep in reference window.
        threshold: MMD threshold for drift detection (default 0.1).
        sigma: RBF bandwidth (default 0.0 = median heuristic).
    """

    def __init__(
        self,
        window_size: int = 32,
        threshold: float = 0.1,
        sigma: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.threshold = float(threshold)
        self.sigma = float(sigma)

        # Reference window: dynamic shape, filled on first update
        self.register_buffer("ref_window", torch.zeros(window_size, 1))
        self.register_buffer("is_filled", torch.tensor(False))
        self.register_buffer("n_seen", torch.tensor(0))
        self.register_buffer("feature_dim", torch.tensor(0))

    def reset(self) -> None:
        """Reset the detector to empty state."""
        self.ref_window.zero_()
        self.is_filled.zero_()
        self.n_seen.zero_()
        self.feature_dim.zero_()

    def update(self, x: torch.Tensor) -> None:
        """Update the reference window with new samples.

        x: (N, D) new samples. Replaces oldest samples (FIFO).
        """
        n = x.size(0)
        if n == 0:
            return
        x = torch.nan_to_num(x, nan=0.0)
        D = x.size(1)
        # Initialize ref_window feature dim on first non-empty input
        if int(self.feature_dim.item()) != D:
            self.feature_dim = torch.tensor(D)
            self.ref_window = torch.zeros(self.window_size, D,
                                          device=x.device, dtype=x.dtype)
            self.n_seen.zero_()
        # FIFO replacement
        if n >= self.window_size:
            self.ref_window = x[-self.window_size:].clone()
        else:
            cur_n = int(self.n_seen.item())
            keep = self.ref_window[cur_n % self.window_size:]
            self.ref_window = torch.cat([keep, x], dim=0)[-self.window_size:]
        self.is_filled = torch.tensor(True)
        self.n_seen += n

    def detect(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Compute MMD(ref, x) and return (mmd_score, is_drift).

        x: (N, D) new samples
        """
        if not bool(self.is_filled.item()) or self.n_seen < self.window_size:
            return torch.zeros(()), False
        x = torch.nan_to_num(x, nan=0.0)
        score = mmd_rbf(self.ref_window, x, sigma=self.sigma)
        is_drift = bool((score > self.threshold).item())
        return score, is_drift


# ----------------------------------------------------------------------------
# Dynamic Expert Pool
# ----------------------------------------------------------------------------


@dataclass
class DynamicExpertPoolConfig:
    """Configuration for the dynamic expert pool."""
    init_size: int = 4           # Initial number of experts
    max_size: int = 8            # Cap on pool growth
    min_size: int = 2            # Floor: never prune below this
    expert_hidden: int = 16      # Expert hidden dim
    add_strategy: str = "copy"   # 'copy' | 'random' | 'noise' (how to init new experts)
    prune_strategy: str = "least_used"  # 'least_used' | 'oldest' | 'random'
    add_eps: float = 1e-6


class ExpertModule(nn.Module):
    """Single expert — a 2-layer MLP from input_size to hidden_size."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DynamicExpertPool(nn.Module):
    """Expert pool that grows and shrinks in response to drift.

    Uses nn.ModuleList of experts. ``add_expert()`` and
    ``prune_expert()`` modify the list. The router and downstream
    code read ``self.experts`` each forward.

    Args:
        input_size: Input feature dimension.
        hidden_size: Expert output dimension.
        config: Pool configuration.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        config: DynamicExpertPoolConfig,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.config = config
        self.experts = nn.ModuleList([
            ExpertModule(input_size, hidden_size)
            for _ in range(config.init_size)
        ])
        # Usage statistics for pruning
        self.register_buffer("usage_count", torch.zeros(config.max_size))
        self.register_buffer("usage_last_step", torch.zeros(config.max_size))
        self.register_buffer("n_adds", torch.tensor(0))
        self.register_buffer("n_prunes", torch.tensor(0))

    @property
    def size(self) -> int:
        return len(self.experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all experts on x: returns (size, B, hidden_size)."""
        outs = [e(x) for e in self.experts]
        return torch.stack(outs, dim=0)

    def add_expert(self, reference: Optional[nn.Module] = None) -> int:
        """Add a new expert. Returns the new size."""
        if self.size >= self.config.max_size:
            return self.size  # at cap
        new = ExpertModule(self.input_size, self.hidden_size)
        if reference is not None and self.config.add_strategy == "copy":
            # Copy from reference, with small noise
            new.fc1.weight.data = reference.fc1.weight.data.clone()
            new.fc1.bias.data = reference.fc1.bias.data.clone()
            new.fc2.weight.data = reference.fc2.weight.data.clone()
            new.fc2.bias.data = reference.fc2.bias.data.clone()
        elif self.config.add_strategy == "noise":
            new.fc1.weight.data += torch.randn_like(new.fc1.weight.data) * 0.01
            new.fc2.weight.data += torch.randn_like(new.fc2.weight.data) * 0.01
        # "random" is just default init
        self.experts.append(new)
        self.n_adds += 1
        return self.size

    def prune_expert(self) -> int:
        """Remove the least-used expert (or the oldest). Returns new size."""
        if self.size <= self.config.min_size:
            return self.size
        # Pick index to remove
        if self.config.prune_strategy == "least_used":
            # Use usage_last_step (most recent window)
            counts = []
            for i in range(self.size):
                counts.append(float(self.usage_last_step[i].item()))
            prune_idx = int(torch.tensor(counts).argmin().item())
        elif self.config.prune_strategy == "oldest":
            prune_idx = 0
        else:  # random
            prune_idx = int(torch.randint(self.size, (1,)).item())
        del self.experts[prune_idx]
        self.n_prunes += 1
        return self.size

    def update_usage(self, weights: torch.Tensor) -> None:
        """Update usage statistics.

        weights: (size,) — average router weights for this forward pass
        """
        if weights.dim() == 0:
            return
        for i in range(min(self.size, weights.size(0))):
            self.usage_count[i] += weights[i]
            self.usage_last_step[i] = weights[i]
        # Zero out pruned slots
        for i in range(self.size, self.config.max_size):
            self.usage_count[i] = 0
            self.usage_last_step[i] = 0


# ----------------------------------------------------------------------------
# Temporal Memory Router
# ----------------------------------------------------------------------------


@dataclass
class TemporalMemoryRouterConfig:
    """Configuration for the temporal memory router."""
    memory_dim: int = 8       # Recurrent memory dimension
    anomaly_dim: int = 4      # Anomaly repository size
    use_anomaly: bool = True  # Whether to use anomaly repository
    top_k: int = 2            # Top-K experts to route to


class TemporalMemoryRouter(nn.Module):
    """Router with recurrent state + anomaly repository.

    logit = Router_MLP([x_t, h, memory, anomaly_history])

    Args:
        input_size: Input feature dim.
        hidden_size: Cell hidden state dim.
        n_experts: Current pool size (updated externally).
        config: Router configuration.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int,
        config: TemporalMemoryRouterConfig,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_experts = int(n_experts)
        self.config = config

        # Router MLP input: [x_t, h, memory, anomaly]
        ctx_dim = config.memory_dim + (config.anomaly_dim if config.use_anomaly else 0)
        in_dim = input_size + hidden_size + ctx_dim
        self.router_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_experts),
        )

        # Memory update: GRU cell for memory
        self.memory_cell = nn.GRUCell(in_dim, config.memory_dim)

        # Anomaly repository: a small learned buffer updated externally
        if config.use_anomaly:
            self.anomaly_buffer = nn.Parameter(
                torch.zeros(config.anomaly_dim), requires_grad=False
            )
        else:
            self.anomaly_buffer = None

        # Init memory state
        self.register_buffer("memory_state", torch.zeros(1, config.memory_dim))
        self.register_buffer("memory_initialized", torch.tensor(False))

    def reset_memory(self) -> None:
        """Reset recurrent memory (between sequences)."""
        self.memory_state.zero_()
        self.memory_initialized.zero_()

    def set_anomaly(self, anomaly: torch.Tensor) -> None:
        """Update the anomaly repository (called externally per forward)."""
        if self.config.use_anomaly and self.anomaly_buffer is not None:
            self.anomaly_buffer.data = anomaly.detach().clone()

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights.

        Args:
            x_t: (B, D) current input.
            h: (B, H) current hidden state.
        Returns:
            weights: (B, n_experts) softmax over experts
            top_idx: (B, top_k) top-K expert indices
            top_w: (B, top_k) top-K normalized weights
        """
        B = x_t.size(0)

        # Build context vector
        ctx_parts = [self.memory_state.expand(B, -1)]
        if self.config.use_anomaly and self.anomaly_buffer is not None:
            ctx_parts.append(self.anomaly_buffer.unsqueeze(0).expand(B, -1))
        ctx = torch.cat(ctx_parts, dim=-1)

        # Build router input
        router_in = torch.cat([x_t, h, ctx], dim=-1)

        # Update memory (GRU step) — memory state batch must match
        if self.memory_state.size(0) != B:
            self.memory_state = self.memory_state[:1].expand(B, -1).contiguous()
        new_mem = self.memory_cell(router_in, self.memory_state)
        self.memory_state = new_mem.detach()  # detach to avoid BPTT through full history

        # Compute logits
        logit = self.router_mlp(router_in)
        weights = F.softmax(logit, dim=-1)

        # Top-K
        top_v, top_idx = weights.topk(min(self.config.top_k, self.n_experts), dim=-1)
        top_w = F.softmax(top_v, dim=-1)  # re-normalize

        return weights, top_idx, top_w


# ----------------------------------------------------------------------------
# Dynamic TMoE Cell
# ----------------------------------------------------------------------------


@dataclass
class DynamicTMoEConfig:
    """Top-level config for Dynamic TMoE."""
    input_size: int = 2
    hidden_size: int = 16
    output_size: int = 1
    pool: DynamicExpertPoolConfig = field(default_factory=DynamicExpertPoolConfig)
    router: TemporalMemoryRouterConfig = field(default_factory=TemporalMemoryRouterConfig)
    drift_window: int = 32
    drift_threshold: float = 0.1
    drift_sigma: float = 0.0
    prune_every: int = 50    # Prune check every N steps
    min_usage_threshold: float = 0.01  # Below this → candidate for prune


class DynamicTMoECfCCell(nn.Module):
    """CfC-style cell with dynamic MoE (drift-aware expert pool).

    Per-step:
      1. Run current expert pool on [x_t, h] → (size, B, H)
      2. Update drift detector with x_t
      3. If drift detected, add a new expert (initialized from current usage-weighted mean)
      4. Periodically prune least-used expert
      5. Update anomaly repository with MMD score
      6. Route via temporal memory router
      7. Weighted mix of expert outputs
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        config: Optional[DynamicTMoEConfig] = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = DynamicTMoEConfig()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.config = config

        # Dynamic expert pool
        self.expert_pool = DynamicExpertPool(
            input_size + hidden_size,  # input to experts is [x_t, h]
            hidden_size,
            config.pool,
        )
        # Drift detector
        self.drift_detector = DriftDetector(
            window_size=config.drift_window,
            threshold=config.drift_threshold,
            sigma=config.drift_sigma,
        )
        # Temporal memory router
        self.router = TemporalMemoryRouter(
            input_size=input_size,
            hidden_size=hidden_size,
            n_experts=config.pool.init_size,
            config=config.router,
        )
        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size)

        # Step counter for prune cadence
        self.register_buffer("step_count", torch.tensor(0))
        self.register_buffer("last_drift_score", torch.tensor(0.0))
        self.register_buffer("drift_count", torch.tensor(0))

    @property
    def pool_size(self) -> int:
        return self.expert_pool.size

    def _expand_router_if_needed(self) -> None:
        """If pool grew, expand the router's output dim via re-init.

        Note: we never shrink the router's output dim — if pool shrinks
        (after prune), the router still has the old larger size and
        the unused outputs are simply not selected by top-K. This is
        cleaner than rebuilding the layer on every shrink/grow.
        """
        target_n = self.expert_pool.size
        old_n = self.router.n_experts
        if target_n <= old_n:
            return  # already big enough
        # Expand last layer to target_n
        old_layer = self.router.router_mlp[-1]
        new_layer = nn.Linear(old_layer.in_features, target_n)
        new_layer.weight.data.zero_()
        new_layer.bias.data.zero_()
        # Copy old weights/bias into the first old_n rows
        with torch.no_grad():
            new_layer.weight[:old_n] = old_layer.weight
            new_layer.bias[:old_n] = old_layer.bias
        self.router.router_mlp[-1] = new_layer
        self.router.n_experts = target_n

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """One step.

        Args:
            x_t: (B, D) current input (may contain NaN)
            h: (B, H) current hidden state
        Returns:
            new_h: (B, H) updated hidden
            output: (B, O) output projection
            info: dict with diagnostics
        """
        x_clean = torch.nan_to_num(x_t, nan=0.0)
        B = x_clean.size(0)

        # --- Drift detection ---
        score, is_drift = self.drift_detector.detect(x_clean)
        self.last_drift_score = score.detach()
        if is_drift:
            self.drift_count += 1
            # Use the most-used expert as reference for the new one
            if self.expert_pool.size > 0:
                counts = self.expert_pool.usage_count[:self.expert_pool.size]
                ref_idx = int(counts.argmax().item())
                self.expert_pool.add_expert(reference=self.expert_pool.experts[ref_idx])
            else:
                self.expert_pool.add_expert()
            self._expand_router_if_needed()

        # --- Periodic pruning ---
        if int(self.step_count.item()) % self.config.prune_every == 0 and int(self.step_count.item()) > 0:
            if self.expert_pool.size > self.expert_pool.config.min_size:
                # Prune the least-used expert (above min)
                self.expert_pool.prune_expert()
                self._expand_router_if_needed()

        # --- Update drift detector with current sample ---
        self.drift_detector.update(x_clean)

        # --- Run all experts ---
        # Input to experts: [x_t, h] → (B, D+H)
        expert_in = torch.cat([x_clean, h], dim=-1)
        expert_outs = self.expert_pool(expert_in)  # (size, B, H)

        # --- Update anomaly repository (rolling buffer of recent MMD scores) ---
        if self.router.config.use_anomaly:
            # Take last few drift scores
            cur = float(score.item())
            buf = self.router.anomaly_buffer.data
            buf = torch.roll(buf, -1)
            buf[-1] = cur
            self.router.set_anomaly(buf)

        # --- Route ---
        weights, top_idx, top_w = self.router(x_clean, h)
        # weights: (B, n_experts), top_idx: (B, top_k), top_w: (B, top_k)

        # Compute output: weighted mix of top-K expert outputs
        # expert_outs: (size, B, H) → (B, size, H)
        expert_outs_t = expert_outs.transpose(0, 1)  # (B, size, H)
        # Clamp top_idx to within [0, pool_size) — router may have grown
        # larger than pool if prune shrunk it
        top_idx_safe = top_idx.clamp(max=self.expert_pool.size - 1)
        B_idx = torch.arange(B, device=expert_outs_t.device).unsqueeze(-1).expand(-1, self.config.router.top_k)
        top_outs = expert_outs_t[B_idx, top_idx_safe]  # (B, top_k, H)
        mixed = (top_w.unsqueeze(-1) * top_outs).sum(dim=1)  # (B, H)

        # --- Update usage stats ---
        # avg weight across batch
        avg_weights = weights.detach().mean(dim=0)
        self.expert_pool.update_usage(avg_weights)

        # --- Output projection ---
        new_h = mixed
        output = self.output_proj(mixed)

        self.step_count += 1

        info = {
            "drift_score": float(score.item()),
            "is_drift": is_drift,
            "pool_size": self.expert_pool.size,
            "drift_count": int(self.drift_count.item()),
            "n_adds": int(self.expert_pool.n_adds.item()),
            "n_prunes": int(self.expert_pool.n_prunes.item()),
        }
        return new_h, output, info


# ----------------------------------------------------------------------------
# Dynamic TMoE Network
# ----------------------------------------------------------------------------


class DynamicTMoECfCNetwork(nn.Module):
    """Rolling-window network using DynamicTMoECfCCell.

    Handles NaN inputs, maintains hidden state across timesteps,
    and supports reset between sequences.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        config: Optional[DynamicTMoEConfig] = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = DynamicTMoEConfig()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.config = config
        self.cell = DynamicTMoECfCCell(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            config=config,
        )

    def forward(
        self,
        x: torch.Tensor,
        times: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        reset_state: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """Run the network on a sequence.

        Args:
            x: (B, T, D) input sequence (may contain NaN)
            times: unused (kept for API compatibility)
            mask: unused (kept for API compatibility)
            reset_state: if True, reset cell state at start
        Returns:
            outputs: (B, T, O) output sequence
            info: aggregated diagnostics
        """
        if reset_state:
            self.cell.drift_detector.reset()
            self.cell.router.reset_memory()
            self.cell.step_count.zero_()
            self.cell.drift_count.zero_()
            self.cell.last_drift_score.zero_()
            # Reset expert pool usage
            self.cell.expert_pool.usage_count.zero_()
            self.cell.expert_pool.usage_last_step.zero_()

        B, T, D = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        drift_scores = []
        pool_sizes = []
        is_drifts = []

        for t in range(T):
            x_t = x[:, t, :]
            h, out, info = self.cell(x_t, h)
            outputs.append(out)
            drift_scores.append(info["drift_score"])
            pool_sizes.append(info["pool_size"])
            is_drifts.append(info["is_drift"])

        outputs = torch.stack(outputs, dim=1)  # (B, T, O)
        agg_info = {
            "drift_score_mean": float(sum(drift_scores) / max(len(drift_scores), 1)),
            "drift_score_max": float(max(drift_scores) if drift_scores else 0.0),
            "n_drifts": sum(1 for d in is_drifts if d),
            "pool_size_initial": pool_sizes[0] if pool_sizes else 0,
            "pool_size_final": pool_sizes[-1] if pool_sizes else 0,
            "n_adds": info["n_adds"],
            "n_prunes": info["n_prunes"],
        }
        return outputs, agg_info

    def get_utilization(self) -> Dict:
        """Get utilization statistics."""
        size = self.cell.expert_pool.size
        counts = self.cell.expert_pool.usage_count[:size]
        total = counts.sum().item()
        if total < 1e-8:
            return {
                "pool_size": size,
                "routing_H": 0.0,
                "max_min": 1.0,
                "active_fraction": 0.0,
                "usage_count": counts.detach().cpu().tolist(),
            }
        probs = (counts / total).cpu().tolist()
        # Entropy
        import math
        H = -sum(p * math.log(p + 1e-12) for p in probs)
        H = H / max(math.log(max(size, 2)), 1)  # normalize
        # Max/min over active (count > 0)
        active = [c for c in probs if c > 1e-8]
        if not active:
            max_min = 1.0
            active_frac = 0.0
        else:
            max_min = max(active) / (min(active) + 1e-8)
            active_frac = len(active) / size
        return {
            "pool_size": size,
            "routing_H": H,
            "max_min": max_min,
            "active_fraction": active_frac,
            "usage_count": counts.detach().cpu().tolist(),
        }


__all__ = [
    "mmd_rbf",
    "DriftDetector",
    "DynamicExpertPoolConfig",
    "ExpertModule",
    "DynamicExpertPool",
    "TemporalMemoryRouterConfig",
    "TemporalMemoryRouter",
    "DynamicTMoEConfig",
    "DynamicTMoECfCCell",
    "DynamicTMoECfCNetwork",
]
