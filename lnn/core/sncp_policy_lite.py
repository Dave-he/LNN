"""SNCPPolicyLite — minimal LTC + actor-critic for crowd-aware navigation.

This is a *minimal* reproduction / re-implementation of the architecture in
heimdilon/sncp-ppo-crowdnav (a PPO + LTC crowd-nav repo, see
``docs/reports/SNCP-PPO_Crowdnav_LTC_深度研读报告.md``).

The full SNCP policy uses three LTC encoders (per-pedestrian spatial,
temporal robot motion, node fusion) plus an attention pooling module over
pedestrians. This lite version is the *core* idea:

* A single in-house ``LTCNetwork`` (from ``lnn/core/ltc.py``) encodes the
  **temporal** axis (the history of robot motion ``[v, w]`` + a few
  global state features).
* A lightweight MLP encodes the **spatial** axis (a fixed-size vector of
  summary statistics over pedestrians).
* A fused trunk produces a shared feature ``sf``.
* An actor head outputs a 2-dim Gaussian (mean for [v, w]); the action is
  clipped to the robot's valid action range downstream.
* A critic head outputs V(s).

The aim is to demonstrate that ``LTCNetwork`` is a drop-in recurrent
encoder for PPO actor-critic, not to match the original repo's 86% hard
scenario success rate. The full env is replaced by a simple
``PointMassCrowdLite`` test wrapper; see ``tests/test_sncp_policy_lite.py``.

References
----------
* heimdilon/sncp-ppo-crowdnav (2026) — LTC + PPO crowd-nav
* Hasani et al. (2021) — Liquid Time-Constant Networks
* Schulman et al. (2017) — Proximal Policy Optimization
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from lnn.core.ltc import LTCNetwork


class _TritanGaussianHead(nn.Module):
    """A 2-dim Gaussian actor head producing (mean, log_std)."""

    MIN_LOG_STD: float = -5.0
    MAX_LOG_STD: float = 2.0

    def __init__(self, input_size: int, action_dim: int = 2) -> None:
        super().__init__()
        self.mu = nn.Linear(input_size, action_dim)
        # Per-dim learnable log-std initialised to log(0.5) ~ -0.69.
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.69))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std, self.MIN_LOG_STD, self.MAX_LOG_STD)
        log_std = log_std.expand_as(mu)
        return mu, log_std

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action, log_prob, entropy)."""
        mu, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        action = normal.rsample()  # reparameterised for PPO
        log_prob = normal.log_prob(action).sum(-1)
        entropy = normal.entropy().sum(-1)
        return action, log_prob, entropy


class SNCPPolicyLite(nn.Module):
    """LTC encoder + actor-critic heads.

    Args:
        temporal_input_size: Dimension of the per-step temporal feature
            (default 2 = ``[v, w]``). The lite version treats the
            *summary* spatial features as part of the temporal input,
            so the policy can be exercised with a single
            ``[B, T, temporal_input_size]`` tensor.
        spatial_summary_size: Dimension of the *per-step* spatial
            summary (e.g. ``[min_d, mean_d, num_pedestrians, goal_dx, goal_dy]``).
        ltc_hidden_size: Hidden size of the LTC encoder.
        trunk_hidden_size: Hidden size of the post-LTC shared trunk.
        action_dim: 2 by default (linear + angular velocity).
        ode_method: 'euler' / 'rk4' / 'dopri5' — euler is fastest, fine
            for a tiny smoke.
    """

    def __init__(
        self,
        temporal_input_size: int = 2,
        ltc_hidden_size: int = 64,
        trunk_hidden_size: int = 64,
        spatial_summary_size: int = 0,
        action_dim: int = 2,
        ode_method: str = "euler",
    ) -> None:
        super().__init__()
        self.temporal_input_size = temporal_input_size
        self.ltc_hidden_size = ltc_hidden_size
        self.trunk_hidden_size = trunk_hidden_size
        self.action_dim = action_dim
        self.spatial_summary_size = spatial_summary_size

        effective_input = temporal_input_size + spatial_summary_size
        # We use the in-house LTCNetwork as a temporal encoder. The
        # output projection is a no-op (output_size == ltc_hidden_size)
        # so the post-LTC trunk can take the hidden states directly.
        self.ltc = LTCNetwork(
            input_size=effective_input,
            hidden_size=ltc_hidden_size,
            output_size=ltc_hidden_size,
            num_layers=1,
            ode_method=ode_method,
            return_sequences=True,
        )
        self.trunk = nn.Sequential(
            nn.Linear(ltc_hidden_size, trunk_hidden_size),
            nn.Tanh(),
            nn.Linear(trunk_hidden_size, trunk_hidden_size),
            nn.Tanh(),
        )
        self.actor = _TritanGaussianHead(trunk_hidden_size, action_dim)
        self.critic = nn.Linear(trunk_hidden_size, 1)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # LTCNetwork expects h0 with shape [num_layers, batch, hidden].
        return torch.zeros(self.ltc.num_layers, batch_size, self.ltc_hidden_size, device=device)

    def encode(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the LTC encoder over ``[B, T, F]`` and return (last-step features, h_T).

        We use the LTC *cell* directly (instead of ``LTCNetwork.forward``)
        so that the final hidden state is observable. ``LTCNetwork`` itself
        is fine for sequence classification, but for an RL actor-critic we
        need the per-step hidden state to seed the next episode.
        """
        if h0 is None:
            h0 = self.initial_hidden(x.shape[0], x.device)
        batch_size, seq_len, _ = x.shape
        # Project the output through the LTC's output_proj per step.
        outputs = []
        h = h0
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_layer = h[0]
            h_layer = self.ltc.cells[0](x_t, h_layer)
            h = h_layer.unsqueeze(0)
            outputs.append(self.ltc.output_proj(h_layer))
        seq = torch.stack(outputs, dim=1)  # [B, T, ltc_hidden_size]
        last_feat = seq[:, -1, :]
        return last_feat, h

    def encode_sequence(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the LTC encoder over ``[B, T, F]`` and return (per-step features, h_T).

        Same as ``encode`` but returns the full [B, T, H] sequence instead of
        the last step. This is what PPO needs when evaluating per-step
        actions from a rollout buffer.
        """
        if h0 is None:
            h0 = self.initial_hidden(x.shape[0], x.device)
        outputs = []
        h = h0
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            h_layer = h[0]
            h_layer = self.ltc.cells[0](x_t, h_layer)
            h = h_layer.unsqueeze(0)
            outputs.append(self.ltc.output_proj(h_layer))
        seq = torch.stack(outputs, dim=1)  # [B, T, H]
        return seq, h

    def forward_trunk(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (shared feature ``sf`` at last step, final hidden state)."""
        last_feat, h_T = self.encode(x, h0=h0)
        sf = self.trunk(last_feat)
        return sf, h_T

    def forward_trunk_sequence(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (shared feature ``sf`` at every step [B, T, trunk_hidden], h_T)."""
        seq, h_T = self.encode_sequence(x, h0=h0)
        B, T, H = seq.shape
        sf = self.trunk(seq.reshape(B * T, H)).reshape(B, T, -1)
        return sf, h_T

    def act(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action, log_prob, entropy, value, h_T)."""
        sf, h_T = self.forward_trunk(x, h0=h0)
        action, log_prob, entropy = self.actor.sample(sf)
        value = self.critic(sf).squeeze(-1)
        return action, log_prob, entropy, value, h_T

    def evaluate_actions(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used by PPO update — return (log_prob, entropy, value) per step.

        ``x`` and ``actions`` are [B, T, F] and [B, T, A] respectively; the
        returned tensors are [B, T].
        """
        sf, _ = self.forward_trunk_sequence(x)
        B, T, _ = sf.shape
        sf_flat = sf.reshape(B * T, -1)
        actions_flat = actions.reshape(B * T, -1)
        mu, log_std = self.actor(sf_flat)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        log_prob = normal.log_prob(actions_flat).sum(-1).reshape(B, T)
        entropy = normal.entropy().sum(-1).reshape(B, T)
        value = self.critic(sf_flat).squeeze(-1).reshape(B, T)
        return log_prob, entropy, value


__all__ = ["SNCPPolicyLite", "_TritanGaussianHead"]
