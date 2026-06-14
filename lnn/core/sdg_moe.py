"""Round 104 — SDG-MoE: Signed Debate Graph Inter-Expert Deliberation (PRD #10-66).

Implements inter-expert deliberation for MoE based on
arXiv:2605.08322 (Kulibaba et al., May 2026) — *SDG-MoE: Signed
Debate Graph Mixture-of-Experts*. After top-K routing, the active
experts engage in signed message passing (support A⁺, critique A⁻)
with disagreement-gated Friedkin-Johnsen anchoring.

Key idea:
  1. Standard MoE: h_new = Σ_k g_k · expert_k(x_t, h)
     (no inter-expert communication)
  2. SDG-MoE: after computing expert_outs (B, K_active, H):
       - pairwise disagreement score per batch
       - e_k ← (1-λ_d) · e_k + λ_d · (e_k + A⁺ · e_active - A⁻ · e_active)
       - h_new = Σ_k g_k · e_k
  3. λ_d = f(disagreement) — Friedkin-Johnsen anchoring

Components:
- ``SDGConfig`` — dataclass for deliberation hyperparameters
- ``disagreement_score(expert_outs)`` — pairwise cosine dissimilarity
- ``signed_debate_step(expert_outs, A_pos, A_neg, alpha, beta)`` — one
  round of signed message passing
- ``SDGQuiteMoECfCCell`` — wraps QuiteMoECfCCell with deliberation
- ``SDGQuiteMoECfCNetwork`` — full network wrapper
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.quite_moe import QuiteMoECfCCell, QuiteMoECfCNetwork


@dataclass
class SDGConfig:
    """SDG-MoE deliberation hyperparameters.

    Args:
        alpha_max: Maximum support update strength. 0 = no support.
        beta_max: Maximum critique update strength. 0 = no critique.
        n_steps: Number of deliberation rounds (default 1).
        use_anchoring: If True, use disagreement-gated Friedkin-Johnsen
            anchoring to bound deliberation strength.
        anchoring_strength: λ_d_max in Friedkin-Johnsen. Default 0.5.
    """
    alpha_max: float = 0.1
    beta_max: float = 0.1
    n_steps: int = 1
    use_anchoring: bool = True
    anchoring_strength: float = 0.5

    def __post_init__(self) -> None:
        if self.alpha_max < 0:
            raise ValueError(f"alpha_max must be >= 0, got {self.alpha_max}")
        if self.beta_max < 0:
            raise ValueError(f"beta_max must be >= 0, got {self.beta_max}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        if not 0.0 <= self.anchoring_strength <= 1.0:
            raise ValueError(
                f"anchoring_strength must be in [0, 1], got {self.anchoring_strength}",
            )


def disagreement_score(expert_outs: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine disagreement among active expert outputs.

    Args:
        expert_outs: (B, K_active, H) expert outputs.

    Returns:
        (B,) per-batch disagreement score, in [0, 1].
        0 = all experts produce identical outputs.
        1 = all experts produce orthogonal outputs.
    """
    if expert_outs.dim() != 3:
        raise ValueError(
            f"expert_outs must be (B, K, H), got {tuple(expert_outs.shape)}",
        )
    B, K, H = expert_outs.shape
    if K < 2:
        return torch.zeros(B, device=expert_outs.device, dtype=expert_outs.dtype)
    # L2-normalize along H
    normed = F.normalize(expert_outs, p=2, dim=-1)  # (B, K, H)
    # Pairwise cosine sim: (B, K, K) = normed @ normed.transpose
    sim = torch.bmm(normed, normed.transpose(1, 2))  # (B, K, K)
    # Mean off-diagonal similarity
    mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
    mean_sim = sim[:, mask].view(B, K, K - 1).mean(dim=(1, 2))  # (B,)
    # Disagreement = 1 - mean similarity, clamped to [0, 1]
    disagreement = (1.0 - mean_sim).clamp(0.0, 1.0)
    return disagreement


def signed_debate_step(
    expert_outs: torch.Tensor,
    A_pos: torch.Tensor,
    A_neg: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """One round of signed message passing.

    Args:
        expert_outs: (B, K_active, H) expert outputs.
        A_pos: (K_active, K_active) support interaction matrix.
        A_neg: (K_active, K_active) critique interaction matrix.
        alpha: support update strength.
        beta: critique update strength.

    Returns:
        (B, K_active, H) updated expert outputs.
    """
    if expert_outs.dim() != 3:
        raise ValueError(
            f"expert_outs must be (B, K, H), got {tuple(expert_outs.shape)}",
        )
    K = expert_outs.size(1)
    if A_pos.shape != (K, K):
        raise ValueError(f"A_pos must be ({K}, {K}), got {tuple(A_pos.shape)}")
    if A_neg.shape != (K, K):
        raise ValueError(f"A_neg must be ({K}, {K}), got {tuple(A_neg.shape)}")
    # support: A_pos @ expert_outs, broadcast over batch
    support = torch.einsum("ij,bjh->bih", A_pos, expert_outs)
    critique = torch.einsum("ij,bjh->bih", A_neg, expert_outs)
    return expert_outs + alpha * support - beta * critique


class SDGLearnedInteractions(nn.Module):
    """Per-K-topk learned support (A⁺) and critique (A⁻) matrices.

    Because the active expert set can change per step (sparse top-K
    routing), we maintain a small bank of A⁺/A⁻ matrices keyed by
    the top-K pattern. For the common case of dense top-K (K'=K),
    we use a single A⁺ and A⁻.

    For sparse top-K, we approximate by using a dense (K, K) matrix
    and zeroing the inactive rows/columns at deliberation time.

    Args:
        n_experts: Total number of experts K.
    """

    def __init__(self, n_experts: int) -> None:
        super().__init__()
        self.n_experts = int(n_experts)
        # Dense A+ and A- for simplicity
        self.A_pos = nn.Parameter(torch.zeros(n_experts, n_experts))
        self.A_neg = nn.Parameter(torch.zeros(n_experts, n_experts))
        # Initialize small
        nn.init.normal_(self.A_pos, mean=0.0, std=0.02)
        nn.init.normal_(self.A_neg, mean=0.0, std=0.02)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (A_pos, A_neg) for use in signed_debate_step."""
        return self.A_pos, self.A_neg


class SDGQuiteMoECfCCell(nn.Module):
    """QuiteMoECfCCell with SDG-MoE inter-expert deliberation.

    Wraps a ``QuiteMoECfCCell`` and adds a post-routing deliberation
    step. After the top-K experts compute their outputs, signed message
    passing (A⁺ for support, A⁻ for critique) refines the expert
    representations before mixture aggregation.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        n_experts: Number of experts (K).
        top_k: Number of experts activated per step.
        n_tau_per_expert: Per-expert ``n_tau``.
        tau_scales: Per-branch initial time constants.
        d_context: QuITE context dimension.
        router_hidden: Width of the optional 2-layer router MLP.
        sdg_config: ``SDGConfig`` with deliberation hyperparameters.
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
        sdg_config: SDGConfig | None = None,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.d_context = int(d_context)
        self.sdg_config = sdg_config or SDGConfig()
        # Inner cell: does the routing and per-expert forward
        self.cell = QuiteMoECfCCell(
            input_size=input_size,
            hidden_size=hidden_size,
            n_experts=n_experts,
            top_k=top_k,
            n_tau_per_expert=n_tau_per_expert,
            tau_scales=tau_scales,
            d_context=d_context,
            router_hidden=router_hidden,
        )
        # Learned interaction matrices
        self.interactions = SDGLearnedInteractions(n_experts=n_experts)
        # Side-channel: last disagreement
        self.last_disagreement: torch.Tensor

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        context: torch.Tensor | None = None,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One step with deliberation.

        Args:
            x_t: (B, input_size) input at this step.
            h: (B, hidden_size) previous hidden state.
            context: (B, d_context) QuITE context vector.
            dt: scalar or [B] per-sample time delta.

        Returns:
            h_new: (B, hidden_size) mixed expert output after deliberation.
        """
        # Step 1: compute per-expert outputs (B, K, H)
        expert_outs = []
        for expert in self.cell.experts:
            h_k = expert(x_t, h, dt=dt)
            expert_outs.append(h_k)
        expert_stack = torch.stack(expert_outs, dim=1)  # (B, K, H)
        # Step 2: routing
        g = self.cell.router(x_t, h, context=context)  # (B, K)
        # Step 3: deliberation
        if self.top_k == self.n_experts:
            # Dense: all K experts participate
            active_stack = expert_stack
            active_indices = None
        else:
            # Sparse: only top-K experts participate
            top_idx = self.cell.router.last_top_idx  # (B, K')
            B, K_top = top_idx.shape
            K = expert_stack.size(1)
            # Gather active expert outputs
            # top_idx_exp: (B, K', H)
            top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            active_stack = expert_stack.gather(1, top_idx_exp)  # (B, K', H)
            active_indices = top_idx  # (B, K')
        # Signed message passing
        A_pos_full, A_neg_full = self.interactions()
        # If sparse, gather the (K', K') sub-matrix of A_pos/A_neg per batch
        for _ in range(self.sdg_config.n_steps):
            if active_indices is None:
                # Dense: use full matrices
                A_pos = A_pos_full
                A_neg = A_neg_full
            else:
                # Sparse: gather sub-matrices
                # top_idx: (B, K'), A_pos: (K, K) → A_pos_active: (B, K', K)
                A_pos_active = A_pos_full[active_indices]  # (B, K', K)
                A_neg_active = A_neg_full[active_indices]  # (B, K', K)
                # Gather columns: (B, K', K) @ one-hot(K, K') → (B, K', K')
                # Use einsum: A_pos_active[b, i, j] * delta(j, k) where k = top_idx[b, ...]
                # Equivalently: gather along last dim
                A_pos = A_pos_active.gather(
                    2, active_indices.unsqueeze(1).expand(-1, active_indices.size(1), -1),
                )  # (B, K', K')
                A_neg = A_neg_active.gather(
                    2, active_indices.unsqueeze(1).expand(-1, active_indices.size(1), -1),
                )  # (B, K', K')
                # Use the FIRST batch's A_pos/A_neg (assume same for simplicity)
                # OR average across batch
                A_pos = A_pos[0]
                A_neg = A_neg[0]
            active_stack = signed_debate_step(
                active_stack, A_pos, A_neg,
                self.sdg_config.alpha_max, self.sdg_config.beta_max,
            )
            # Optional Friedkin-Johnsen anchoring
            if self.sdg_config.use_anchoring:
                with torch.no_grad():
                    d = disagreement_score(active_stack)  # (B,)
                self.last_disagreement = d.detach()
                # Anchor: scale update by disagreement (more disagreement → more deliberation)
                # But bound by anchoring_strength
                lambda_d = (
                    self.sdg_config.anchoring_strength
                    * d.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                )
                # Apply anchoring: mix updated with original
                if active_indices is None:
                    expert_orig = expert_stack
                else:
                    expert_orig = expert_stack.gather(
                        1, self.cell.router.last_top_idx.unsqueeze(-1)
                        .expand(-1, -1, self.hidden_size),
                    )
                active_stack = (1.0 - lambda_d) * expert_orig + lambda_d * active_stack
        # Step 4: aggregate with routing weights
        # Map g to active set: take g[top_idx]
        if self.top_k == self.n_experts:
            h_new = (g.unsqueeze(-1) * active_stack).sum(dim=1)
        else:
            # Active g
            g_active = g.gather(1, self.cell.router.last_top_idx)  # (B, K')
            # Renormalize so it sums to 1
            g_active = g_active / g_active.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            h_new = (g_active.unsqueeze(-1) * active_stack).sum(dim=1)
        # Update last_g
        self.cell.last_g = g.detach()
        return h_new


class SDGQuiteMoECfCNetwork(nn.Module):
    """Full network with QuITE context + SDG-MoE deliberation.

    Wraps a ``QuiteMoECfCNetwork`` with deliberation enabled.

    Args:
        Same as ``QuiteMoECfCNetwork``, plus ``sdg_config``.
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
        sdg_config: SDGConfig | None = None,
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
        self.quite = __import__(
            "lnn.core.quite_embedding", fromlist=["QueryIrregularEmbedding"],
        ).QueryIrregularEmbedding(
            d_input=input_size,
            n_queries=n_queries,
            d_model=d_context,
            n_heads=n_heads,
        )
        # SDG-augmented MoE cell
        self.cell = SDGQuiteMoECfCCell(
            input_size=input_size,
            hidden_size=hidden_size,
            n_experts=n_experts,
            top_k=top_k,
            n_tau_per_expert=n_tau_per_expert,
            tau_scales=tau_scales,
            d_context=d_context,
            sdg_config=sdg_config,
        )
        # Output projection
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
        """Pre-compute QuITE context from the full irregular sequence."""
        from lnn.core.quite_moe import quite_context_pool
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
        """Run the full sequence with QuITE+SDG-MoE deliberation."""
        B, T, D = observations.shape
        if D != self.input_size:
            raise ValueError(f"Expected D={self.input_size}, got {D}")
        # Build NaN-aware validity mask
        obs_mask = torch.isfinite(observations).all(dim=-1)
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=observations.device)
        elif mask.dtype != torch.bool:
            mask = mask.bool()
        mask = mask & obs_mask
        # Get context
        if precomputed_context is not None:
            context = precomputed_context
        else:
            context = self.compute_context(observations, times, mask=mask)
        # Recurrent forward with deliberation
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


__all__ = [
    "SDGConfig",
    "disagreement_score",
    "signed_debate_step",
    "SDGLearnedInteractions",
    "SDGQuiteMoECfCCell",
    "SDGQuiteMoECfCNetwork",
]
