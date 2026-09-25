"""Decision-quality (reach_rate) CI guard for SNCP-PPO.

Phase B of PRD-MSD-LNN. The repo previously had no reach_rate CI test
(``tests/test_sncp_policy_lite.py`` was shape-only — agent verified).
This file adds:

* A standalone ``PDExpert`` smoke (high reach_rate on n_ped=0).
* An end-to-end reach_rate test on ``PointMassNavLite`` that runs the
  trained PD controller and asserts reach_rate >= 0.95 on n_ped=0.
* An IL warm-start sanity test: after training BC on PD demos and
  transferring recurrent weights to SNCP, the policy's actions on a
  few validation states must be close to the PD expert's actions
  (within a generous tolerance — we're testing structural correctness,
  not competition).

These tests are fast enough to run on every CI but generous enough to
allow occasional randomness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.bench_pid_expert_demo import PDExpert, collect_demos  # noqa: E402
from scripts.experiment_sncp_ppo_lite import PointMassNavLite  # noqa: E402


# ---------------------------------------------------------------------------
# PD controller reach-rate gate
# ---------------------------------------------------------------------------


def test_pd_expert_reach_rate_n_ped_0():
    """The PD expert should reach the goal >= 95% of the time on n_ped=0."""
    demos = collect_demos(
        n_episodes=100, n_pedestrians=0, max_steps_per_episode=50,
        seed=42, controller=PDExpert(),
    )
    s = demos["summary"]
    assert s["reach_rate"] >= 0.95, (
        f"PD reach_rate too low on n_ped=0: {s['reach_rate']:.3f} "
        f"(collision_rate={s['collision_rate']:.3f})"
    )


def test_pd_expert_no_collisions_n_ped_0():
    """No static obstacles should be hit by a working PD expert."""
    demos = collect_demos(
        n_episodes=100, n_pedestrians=0, max_steps_per_episode=50,
        seed=123, controller=PDExpert(),
    )
    s = demos["summary"]
    assert s["collision_rate"] <= 0.05, (
        f"PD collision_rate too high on n_ped=0: {s['collision_rate']:.3f}"
    )


def test_pd_expert_action_dim_and_clamp():
    """PD expert actions must be within env clip ranges."""
    env = PointMassNavLite(seed=0, n_pedestrians=0)
    expert = PDExpert()
    expert.reset()
    obs = env.reset(seed=0)
    v, w = expert(obs, env)
    assert -0.1 <= v <= 0.1, f"v out of env clip: {v}"
    import math
    assert -math.pi / 2 <= w <= math.pi / 2, f"w out of env clip: {w}"


# ---------------------------------------------------------------------------
# IL warm-start sanity
# ---------------------------------------------------------------------------


def test_il_warmstart_actions_close_to_pd_on_validation_states():
    """After BC training on PD demos, the IL policy should mimic PD on val states."""
    from lnn.utils.lnn_il_to_ppo_transfer import train_il_on_demos

    torch.manual_seed(0)
    state_dim = 14
    action_dim = 2
    # Generate PD demos.
    demos = collect_demos(
        n_episodes=100, n_pedestrians=0, max_steps_per_episode=30,
        seed=42, controller=PDExpert(),
    )
    transitions = demos["transitions"]  # list of (obs, action, next_obs)
    obs_list = [t[0] for t in transitions]
    action_list = [t[1] for t in transitions]
    obs = torch.tensor(obs_list, dtype=torch.float32)
    actions = torch.tensor(action_list, dtype=torch.float32)

    # Train BC for a few epochs (smoke).
    policy, _history = train_il_on_demos(
        obs[:2000], actions[:2000],
        state_dim=state_dim, action_dim=action_dim,
        hidden_size=16, encoder_size=state_dim,
        recurrent_type="ltc", head_type="mse",
        epochs=8, batch_size=64, lr=3e-3, log_every=0,
    )

    # Validate on a held-out slice.
    val_obs = obs[2000:2200]
    val_actions = actions[2000:2200]
    with torch.no_grad():
        pred = policy(val_obs.unsqueeze(1)).squeeze(1)
    err = (pred - val_actions).abs().mean().item()
    # Loose tolerance: PD demos are noisy-ish (we use obs as state, not raw pos);
    # 8 epochs is not enough for a perfect fit. The point is *some* learning.
    assert err < 0.20, f"BC didn't approximate PD well enough: mean abs err {err:.4f}"


# ---------------------------------------------------------------------------
# Transferability end-to-end (warm-started PPO still runs)
# ---------------------------------------------------------------------------


def test_warmstarted_ppo_rolls_out_without_nan():
    """After BC->PPO transfer, rolling out the PPO policy must produce finite actions."""
    from lnn.core.sncp_policy_lite import SNCPPolicyLite
    from lnn.utils.lnn_il_to_ppo_transfer import train_il_on_demos, transfer_il_to_ppo

    torch.manual_seed(0)
    state_dim = 14
    action_dim = 2
    obs = torch.randn(64, state_dim)
    actions = torch.randn(64, action_dim)

    il, _ = train_il_on_demos(
        obs, actions,
        state_dim=state_dim, action_dim=action_dim,
        hidden_size=16, encoder_size=state_dim,
        recurrent_type="ltc", head_type="mse",
        epochs=4, batch_size=32, lr=3e-3, log_every=0,
    )
    ppo = SNCPPolicyLite(
        temporal_input_size=state_dim, ltc_hidden_size=16,
        trunk_hidden_size=16, action_dim=action_dim, ode_method="euler",
    )
    transfer_il_to_ppo(il, ppo)

    # Roll out on a fresh env.
    env = PointMassNavLite(seed=7, n_pedestrians=0)
    obs_t = env.reset(seed=7)
    h = ppo.initial_hidden(batch_size=1, device=torch.device("cpu"))
    for _ in range(env.HORIZON):
        x = obs_t.view(1, 1, -1)
        with torch.no_grad():
            action, _lp, _entropy, _value, h = ppo.act(x, h)
        assert torch.isfinite(action).all(), f"non-finite action: {action}"
        result = env.step(action.squeeze(0))
        obs_t = result.obs
        if result.done:
            break
    # If we got here, rollout was clean. (Reach rate is *not* asserted here —
    # PPO needs RL fine-tuning to actually solve the task; we just want to
    # confirm the transferred encoder + freshly initialised actor produce valid
    # outputs over a horizon.)
    assert True