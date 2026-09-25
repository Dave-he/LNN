"""JEPA-style world model with CfC / ParallelCfC backbones.

Phase C of PRD-MSD-LNN. Implements a lightweight Joint Embedding
Predictive Architecture (JEPA) on top of the in-house liquid neural
network primitives, with three components:

    JEPAEncoder       — obs_t  ->  z_t (latent)
    JEPAWorldModel    — (z_t, a_t)  ->  ẑ_{t+1} (predicted next latent)
    JEPAPolicyHead    — z_t    ->  a_t   (single forward pass policy)

Plus two utility classes used by the training / inference scripts:

    StateSnapshot     — serialise / restore parallel recurrent state
    LatentPlanner     — K-step rollout in latent space (train-time
                        supervision signal for distillation)

Architectural notes
-------------------
* JEPAEncoder and JEPAWorldModel both use CfCNetwork (closed-form,
  NPU-friendly). The world-model uses ``n_tau > 1`` for fast / slow
  time-scale split when available, otherwise defaults to vanilla CfC.
* JEPAPolicyHead is intentionally simple (Linear or 2-dim Gaussian).
  MDN is NOT supported (see r307 honest negative on single-mode targets).
* StateSnapshot serialises hidden-state tensors via ``torch.save`` /
  ``torch.load``. Designed for ``CfCNetwork.cells`` and
  ``ParallelCfCNetwork.h_layers`` which both expose external state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from lnn.core.cfc import CfCNetwork


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


DEFAULT_LATENT_DIM: int = 8
DEFAULT_HIDDEN_SIZE: int = 16
DEFAULT_N_TAU: int = 1  # set > 1 for multi-timescale world model
DEFAULT_HORIZON: int = 4  # latent planner rollout depth


# ---------------------------------------------------------------------------
# StateSnapshot utility
# ---------------------------------------------------------------------------


class StateSnapshot:
    """Serialize / restore recurrent state tensors.

    Works for any nn.Module whose state we want to capture. We deliberately
    avoid deepcopying the entire module — only the tensors in
    ``module.state_dict()`` plus an optional bag of extra buffers (e.g.
    ``ParallelCfCNetwork.h_layers`` which lives outside the standard
    state_dict).

    Args:
        module: the source module whose state we want to save.
        extra_buffers: optional dict of named tensors (e.g. parallel
            hidden states) to capture alongside ``state_dict``.
    """

    def __init__(self, module: nn.Module, extra_buffers: Optional[dict[str, torch.Tensor]] = None) -> None:
        self.state = {k: v.detach().clone() for k, v in module.state_dict().items()}
        self.extra = {k: v.detach().clone() for k, v in (extra_buffers or {}).items()}

    def restore(self, module: nn.Module, strict: bool = True) -> None:
        module.load_state_dict(self.state, strict=strict)
        # caller is responsible for re-attaching extra buffers if needed.

    def to(self, device: torch.device) -> "StateSnapshot":
        self.state = {k: v.to(device) for k, v in self.state.items()}
        self.extra = {k: v.to(device) for k, v in self.extra.items()}
        return self

    def cpu(self) -> "StateSnapshot":
        return self.to(torch.device("cpu"))


# ---------------------------------------------------------------------------
# JEPAEncoder
# ---------------------------------------------------------------------------


class JEPAEncoder(nn.Module):
    """Map an observation to a latent vector.

    Architecture:
        obs_t -> Linear -> LayerNorm -> SiLU -> CfCNetwork -> Linear -> z_t

    The recurrent core uses ``CfCNetwork`` so the encoder has a static
    compute graph and is NPU-friendly. The output is a single vector
    ``z_t`` of dimension ``latent_dim`` (no sequence axis).
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        n_tau: int = DEFAULT_N_TAU,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.recurrent = CfCNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            num_layers=1,
            n_tau=n_tau,
            return_sequences=False,
        )
        self.output_proj = nn.Linear(hidden_size, latent_dim)

    def forward(self, obs_seq: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Encode a ``[B, T, obs_dim]`` sequence to a ``[B, latent_dim]`` latent.

        Returns only the *final* latent — the latent state at the end of
        the input window. Hidden state is internal; use :class:`StateSnapshot`
        if you need to restore a recurrent state across calls.
        """
        x = self.input_proj(obs_seq)
        h_seq = self.recurrent(x, dt=dt)
        # recurrent returns either [B, hidden] or [B, T, hidden]
        if h_seq.dim() == 3:
            h_seq = h_seq[:, -1, :]
        return self.output_proj(h_seq)

    @torch.no_grad()
    def encode_step(self, obs_t: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Encode a single observation ``[B, obs_dim]`` -> ``[B, latent_dim]``.

        Useful for online decision loops where T=1.
        """
        x = self.input_proj(obs_t.unsqueeze(1))
        h = self.recurrent(x, dt=dt)
        if h.dim() == 3:
            h = h[:, -1, :]
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# JEPAWorldModel
# ---------------------------------------------------------------------------


class JEPAWorldModel(nn.Module):
    """Predict next latent from current latent + action.

    Architecture:
        concat([z_t, a_t]) -> Linear -> CfC -> Linear -> ẑ_{t+1}

    This is a deterministic forward model (no sampling in the world
    model itself — Gaussian noise is reserved for the SDE-CfC variant
    in :mod:`lnn.core.sde_cfc`). Training uses MSE on the predicted
    latent vs the encoder's actual next latent.
    """

    def __init__(
        self,
        latent_dim: int = DEFAULT_LATENT_DIM,
        action_dim: int = 2,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        n_tau: int = DEFAULT_N_TAU,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.recurrent = CfCNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            num_layers=1,
            n_tau=n_tau,
            return_sequences=False,
        )
        self.output_proj = nn.Linear(hidden_size, latent_dim)

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Predict next latent.

        Args:
            z_t: ``[B, latent_dim]`` current latent (or ``[B, T, latent_dim]``).
            a_t: ``[B, action_dim]`` action taken at t (broadcast across T if needed).
            dt: time delta (default 1.0).
        Returns:
            ẑ_{t+1}: ``[B, latent_dim]`` predicted next latent.
        """
        if z_t.dim() == 2:
            z_t = z_t.unsqueeze(1)  # [B, 1, latent_dim]
        T = z_t.shape[1]
        # broadcast action across time
        if a_t.dim() == 2:
            a_t = a_t.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([z_t, a_t], dim=-1)
        x = self.input_proj(x)
        h = self.recurrent(x, dt=dt)
        if h.dim() == 3:
            h = h[:, -1, :]
        return self.output_proj(h)

    @torch.no_grad()
    def predict_next(self, z_t: torch.Tensor, a_t: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        return self.forward(z_t, a_t, dt=dt)


# ---------------------------------------------------------------------------
# JEPAPolicyHead
# ---------------------------------------------------------------------------


class JEPAPolicyHead(nn.Module):
    """Map latent (and optionally raw obs) to action.

    Two modes:
        'mse'   — single Linear(input -> action_dim), deterministic
        'gauss' — Gaussian head: Linear(input -> mu) + learnable log_std
                  (clamped to [-5, 2]). Returns (action, log_prob, entropy)
                  for compatibility with PPO-style rollouts.

    Skip connection: if ``obs_dim > 0`` at construction time, the head
    consumes ``concat([z_t, obs_t])`` (input dim = ``latent_dim + obs_dim``).
    This dramatically helps on tasks where the latent needs to retain
    raw task-relevant state (e.g. goal-relative position for PointMassNav).

    MDN is NOT supported here — r307 honest negative on single-mode
    targets (PD-expert action distributions are unimodal Gaussian).
    """

    MIN_LOG_STD: float = -5.0
    MAX_LOG_STD: float = 2.0

    def __init__(
        self,
        latent_dim: int = DEFAULT_LATENT_DIM,
        action_dim: int = 2,
        head_type: str = "mse",
        obs_dim: int = 0,
    ) -> None:
        super().__init__()
        head_type = head_type.lower()
        if head_type not in {"mse", "gauss"}:
            raise ValueError(f"head_type must be mse or gauss, got {head_type}")
        self.head_type = head_type
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.input_dim = latent_dim + obs_dim
        self.mu = nn.Linear(self.input_dim, action_dim)
        if head_type == "gauss":
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.69))

    def forward(self, z_t: torch.Tensor, obs_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Deterministic action prediction.

        Args:
            z_t: ``[B, latent_dim]`` latent.
            obs_t: optional ``[B, obs_dim]`` raw observation for skip
                connection. If ``None``, falls back to plain ``z_t -> a``.
        """
        if self.obs_dim > 0 and obs_t is not None:
            x = torch.cat([z_t, obs_t], dim=-1)
        else:
            x = z_t
        return self.mu(x)

    def sample(self, z_t: torch.Tensor, obs_t: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action with reparameterisation; returns (action, log_prob, entropy)."""
        if self.head_type != "gauss":
            raise RuntimeError("sample() requires head_type='gauss'")
        if self.obs_dim > 0 and obs_t is not None:
            x = torch.cat([z_t, obs_t], dim=-1)
        else:
            x = z_t
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std, self.MIN_LOG_STD, self.MAX_LOG_STD)
        log_std = log_std.expand_as(mu)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(-1)
        entropy = normal.entropy().sum(-1)
        return action, log_prob, entropy


# ---------------------------------------------------------------------------
# LatentPlanner
# ---------------------------------------------------------------------------


@dataclass
class PlannerResult:
    """Output of :class:`LatentPlanner.rollout`."""

    best_action: torch.Tensor  # [B, action_dim]
    best_return: torch.Tensor  # [B]
    rollout_returns: torch.Tensor  # [B, n_candidates]


class LatentPlanner(nn.Module):
    """K-step latent rollout + reward surrogate for distillation.

    Wraps a :class:`JEPAWorldModel` and a reward surrogate
    (default: identity — the predicted latent L2 distance is used as
    a negative "energy" score). Used at train time to score candidate
    action sequences; the best one is distilled into the policy head
    via supervised behaviour cloning.

    Per-step diverse actions: for each candidate we sample a full
    K-step action trajectory (so different candidates use different
    actions at each step), then roll out the world model with the
    candidate's own trajectory.

    Args:
        world_model: trained :class:`JEPAWorldModel`.
        horizon: K — number of rollout steps per candidate action.
        n_candidates: how many candidate trajectories to score per state.
    """

    def __init__(
        self,
        world_model: JEPAWorldModel,
        horizon: int = DEFAULT_HORIZON,
        n_candidates: int = 8,
        action_dim: int = 2,
        action_low: float = -0.1,
        action_high: float = 0.1,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

    def _sample_trajectories(self, batch: int, device: torch.device) -> torch.Tensor:
        """Sample ``n_candidates`` full K-step trajectories per batch element.

        Returns:
            ``[B, n_candidates, horizon, action_dim]`` — each candidate is
            a complete K-step action plan. Actions are uniform in
            ``[action_low, action_high]``.
        """
        return torch.empty(
            batch, self.n_candidates, self.horizon, self.action_dim, device=device,
        ).uniform_(self.action_low, self.action_high)

    def rollout(
        self,
        z0: torch.Tensor,
        reward_fn: Optional[callable] = None,
    ) -> PlannerResult:
        """Score ``n_candidates`` K-step trajectories starting from ``z0``.

        For each candidate we roll out ``horizon`` steps in latent space
        using ``world_model`` with that candidate's per-step actions.
        The reward surrogate is the *sum of negative predicted-latent
        norms* (encouraging the latent to stay small / smooth). If
        ``reward_fn`` is provided, it overrides the default.

        Returns:
            :class:`PlannerResult` with ``best_action`` = the first-step
            action of the best-scoring trajectory per batch element,
            and the rollout returns for diagnostics.
        """
        batch = z0.shape[0]
        device = z0.device
        trajectories = self._sample_trajectories(batch, device)
        # z is shared at the start: [B, n_candidates, latent_dim]
        z = z0.unsqueeze(1).expand(-1, self.n_candidates, -1).contiguous()
        returns = torch.zeros(batch, self.n_candidates, device=device)
        with torch.no_grad():
            for k in range(self.horizon):
                a_k = trajectories[:, :, k, :]  # [B, n_candidates, action_dim]
                # Predict next latent for each candidate.
                z_flat = z.reshape(-1, z.shape[-1])
                a_flat = a_k.reshape(-1, a_k.shape[-1])
                z_next_flat = self.world_model.predict_next(z_flat, a_flat)
                z_next = z_next_flat.reshape(batch, self.n_candidates, -1)
                # Reward surrogate.
                if reward_fn is None:
                    r = -z_next.norm(dim=-1)
                else:
                    r = reward_fn(z_next)
                returns += r
                z = z_next
        best_idx = returns.argmax(dim=1)
        # best_action = the *first-step* action of the best trajectory.
        best_action = trajectories[torch.arange(batch, device=device), best_idx, 0, :]
        return PlannerResult(
            best_action=best_action,
            best_return=returns.max(dim=1).values,
            rollout_returns=returns,
        )

    @torch.no_grad()
    def distill_action(self, z0: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: returns only the best first-step action per state."""
        return self.rollout(z0).best_action


# ---------------------------------------------------------------------------
# Composite: full JEPA decision policy (encoder + world model + policy head)
# ---------------------------------------------------------------------------


class JEPAPolicy(nn.Module):
    """End-to-end JEPA decision policy.

    Training mode: train world model with ``forward_train(...)``
    (returns next-latent prediction + MSE loss vs target latent).
    Inference mode: call ``forward_inference(obs)`` to get an action in
    a single forward pass through encoder + policy head.

    Args:
        obs_dim: observation dimensionality (e.g. 14 for PointMassNavLite).
        action_dim: action dimensionality (e.g. 2 for PointMassNavLite).
        latent_dim: bottleneck size for the JEPA latent.
        hidden_size: hidden size of all CfC cores inside.
        head_type: 'mse' or 'gauss' for :class:`JEPAPolicyHead`.
        n_tau: number of time-scale splits for :class:`CfCNetwork`.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        head_type: str = "mse",
        n_tau: int = DEFAULT_N_TAU,
        head_obs_skip: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.head_obs_skip = head_obs_skip
        self.encoder = JEPAEncoder(obs_dim=obs_dim, latent_dim=latent_dim,
                                   hidden_size=hidden_size, n_tau=n_tau)
        self.world_model = JEPAWorldModel(latent_dim=latent_dim, action_dim=action_dim,
                                          hidden_size=hidden_size, n_tau=n_tau)
        self.policy_head = JEPAPolicyHead(
            latent_dim=latent_dim, action_dim=action_dim,
            head_type=head_type,
            obs_dim=obs_dim if head_obs_skip else 0,
        )

    def forward_train(
        self,
        obs_seq: torch.Tensor,
        next_obs_seq: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """Compute world-model MSE loss for a batch of transitions.

        Args:
            obs_seq:      ``[B, T, obs_dim]``
            next_obs_seq: ``[B, T, obs_dim]``
            actions:      ``[B, T, action_dim]``
        Returns:
            dict with keys ``z``, ``z_hat``, ``mse``.
        """
        # Encode each step into a latent (treat each step as length-1 sequence
        # for simplicity here; for long sequences use batched encoding).
        B, T, _ = obs_seq.shape
        z_list = []
        z_next_list = []
        for t in range(T):
            z_list.append(self.encoder.encode_step(obs_seq[:, t, :]))
            z_next_list.append(self.encoder.encode_step(next_obs_seq[:, t, :]))
        z = torch.stack(z_list, dim=1)         # [B, T, latent_dim]
        z_next = torch.stack(z_next_list, dim=1)  # [B, T, latent_dim]
        # World model: predict z_{t+1} from z_t and a_t for every step.
        z_hat = []
        for t in range(T):
            z_hat.append(self.world_model(z[:, t, :], actions[:, t, :]))
        z_hat = torch.stack(z_hat, dim=1)  # [B, T, latent_dim]
        mse = torch.nn.functional.mse_loss(z_hat, z_next)
        return {"z": z, "z_hat": z_hat, "mse": mse}

    def forward_inference(self, obs_t: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Single-step decision: obs_t -> action."""
        z = self.encoder.encode_step(obs_t)
        if deterministic or self.policy_head.head_type == "mse":
            if self.head_obs_skip:
                return self.policy_head(z, obs_t)
            return self.policy_head(z)
        if self.head_obs_skip:
            action, _, _ = self.policy_head.sample(z, obs_t)
        else:
            action, _, _ = self.policy_head.sample(z)
        return action

    def snapshot(self) -> StateSnapshot:
        """Capture the full JEPA policy state."""
        return StateSnapshot(self)