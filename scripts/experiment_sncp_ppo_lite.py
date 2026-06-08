"""PPO + SNCPPolicyLite smoke experiment on a 2D point-mass nav task.

This is a *minimal* reproducible smoke (not a real crowdnav). The goal is to
demonstrate that ``lnn.core.sncp_policy_lite.SNCPPolicyLite`` (LTC encoder +
actor-critic) can be trained end-to-end with PPO on a simple task, not to
match heimdilon/sncp-ppo-crowdnav's 86% hard scenario success rate.

Task (2D point-mass nav):
- Start at origin (0, 0). Goal at (1, 1).
- 2 static obstacles at (-0.2, 0.3) and (0.5, -0.3) (radius 0.2)
  — only when ``--n-pedestrians 0`` (the default, for backward compat).
- When ``--n-pedestrians N >= 1``: replace static obstacles with N moving
  pedestrians. Each pedestrian walks on a deterministic seeded circular path
  around its origin. The policy must learn to avoid them while reaching the
  goal. Observation is augmented to ``[pos.x, pos.y, goal_dx, goal_dy,
  ped1_dx, ped1_dy, ..., pedN_dx, pedN_dy]``.
- Action: [v ∈ [-0.1, 0.1], w ∈ [-π/2, π/2]] (linear/angular velocity).
- Dynamics: x_{t+1} = x_t + v·[cos θ_t, sin θ_t]·dt; θ_{t+1} = θ_t + w·dt
  where dt=1.0, horizon T=20.
- Reward: +10 on goal; -10 on collision (static obstacle or pedestrian);
  -0.1·|p_t - goal| dense.

PPO:
- Roll out N=8 episodes per update, K=4 epochs of minibatch updates.
- clip=0.2, vf_coef=0.5, ent_coef=0.01.
- 30 PPO updates ≈ 240 episodes.

Curriculum support (``--curriculum``): when set, train sequentially on
``n_pedestrians=1, 2, 3`` with ``--ppo-updates-per-stage`` updates per stage.
The model is reused across stages (parameters carry over).

This is a smoke — the script is designed to complete in a few minutes on
CPU and produce a JSON+MD with learning curve + final win rate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lnn.core.sncp_policy_lite import SNCPPolicyLite  # noqa: E402


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class Step:
    obs: torch.Tensor  # [obs_dim]
    reward: float
    done: bool
    info: dict


class PointMassNavLite:
    """2D point-mass navigation with ``n_pedestrians`` moving obstacles (or 0 static).

    When ``n_pedestrians=0`` (default), the env uses 2 static obstacles
    (backward compatible with iter#26). When ``n_pedestrians >= 1``, the
    static obstacles are *replaced* by ``n_pedestrians`` walking agents on
    deterministic circular paths, and the observation is augmented with
    ``2 * MAX_PED_SLOTS`` dims (per-ped [dx, dy] from the agent). The first
    ``n_pedestrians`` slots are filled with real relative positions; the
    remaining ``MAX_PED_SLOTS - n_pedestrians`` slots are zero-padded (a
    sentinel meaning "no pedestrian here"). This keeps ``obs_dim`` constant
    across all curriculum stages (n=1, 2, 3) so a single policy can be reused.
    """

    BASE_OBS_DIM: int = 4  # [pos.x, pos.y, goal_dx, goal_dy]
    MAX_PED_SLOTS: int = 5  # matches len(PEDESTRIAN_ORIGINS)
    ACTION_DIM: int = 2
    HORIZON: int = 20
    DT: float = 1.0
    GOAL_REWARD: float = 10.0
    COLLISION_REWARD: float = -10.0
    DENSE_COEF: float = 0.1
    GOAL_THRESHOLD: float = 0.15
    OBSTACLE_RADIUS: float = 0.2
    PEDESTRIAN_RADIUS: float = 0.15
    PEDESTRIAN_SPEED: float = 0.06  # per step on a unit circle
    PEDESTRIAN_ORIGINS: list = [(-0.2, 0.3), (0.5, -0.3), (0.7, 0.5), (-0.4, -0.2), (0.3, 0.7)]

    def __init__(self, seed: int = 0, n_pedestrians: int = 0) -> None:
        if n_pedestrians < 0:
            raise ValueError(f"n_pedestrians must be >= 0, got {n_pedestrians}")
        if n_pedestrians > self.MAX_PED_SLOTS:
            raise ValueError(
                f"n_pedestrians={n_pedestrians} exceeds MAX_PED_SLOTS={self.MAX_PED_SLOTS}; "
                f"expand PEDESTRIAN_ORIGINS and MAX_PED_SLOTS first."
            )
        self.goal = torch.tensor([1.0, 1.0])
        self.n_pedestrians = n_pedestrians
        # Static obstacles are used only in the n=0 case (backward compat).
        self.obstacles = [(-0.2, 0.3), (0.5, -0.3)] if n_pedestrians == 0 else []
        # Pedestrians are pre-seeded with deterministic circular params.
        self._ped_params: list[tuple[float, float, float, float, float]] = []
        # Per-ped: (cx, cy, radius, omega, phase). Phase=0 by default; omega
        # alternates sign so the cohort spreads out around their orbits.
        for i in range(n_pedestrians):
            cx, cy = self.PEDESTRIAN_ORIGINS[i]
            self._ped_params.append((cx, cy, 0.20, 0.12 if i % 2 == 0 else -0.12, 0.0))
        self._step_count = 0
        # obs_dim is constant: BASE_OBS_DIM + 2 * MAX_PED_SLOTS (= 4 + 10 = 14).
        # This keeps policy input size fixed across curriculum stages.
        self.obs_dim = self.BASE_OBS_DIM + 2 * self.MAX_PED_SLOTS
        self.reset(seed=seed)

    def _ped_positions(self, t: int) -> list[tuple[float, float]]:
        """Return per-pedestrian (x, y) at global step ``t``."""
        out = []
        for (cx, cy, r, omega, _) in self._ped_params:
            phase = omega * t
            out.append((cx + r * math.cos(phase), cy + r * math.sin(phase)))
        return out

    def reset(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        self.pos = torch.zeros(2)
        self.theta = 0.0
        self._step_count = 0
        return self._current_obs()

    def _current_obs(self) -> torch.Tensor:
        base = [
            self.pos[0].item(),
            self.pos[1].item(),
            (self.goal - self.pos)[0].item(),
            (self.goal - self.pos)[1].item(),
        ]
        # Real ped relative positions, then zero-fill the unused slots.
        active_peds = self._ped_positions(self._step_count)
        ped_obs: list[float] = []
        for px, py in active_peds:
            ped_obs.extend([self.pos[0].item() - px, self.pos[1].item() - py])
        # Pad to 2 * MAX_PED_SLOTS with zeros (sentinel: "no ped in this slot").
        pad_n = self.MAX_PED_SLOTS - self.n_pedestrians
        ped_obs.extend([0.0] * (2 * pad_n))
        return torch.tensor(base + ped_obs)

    def step(self, action: torch.Tensor) -> Step:
        v = float(action[0].clamp(-0.1, 0.1))
        w = float(action[1].clamp(-math.pi / 2, math.pi / 2))
        self.pos[0] += v * math.cos(self.theta) * self.DT
        self.pos[1] += v * math.sin(self.theta) * self.DT
        self.theta += w * self.DT
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))  # wrap
        self._step_count += 1

        # Collision check: static obstacles (n=0) or pedestrians (n>=1).
        collision = False
        for ox, oy in self.obstacles:
            if (self.pos[0].item() - ox) ** 2 + (self.pos[1].item() - oy) ** 2 < self.OBSTACLE_RADIUS ** 2:
                collision = True
                break
        if not collision:
            for px, py in self._ped_positions(self._step_count):
                if (self.pos[0].item() - px) ** 2 + (self.pos[1].item() - py) ** 2 < self.PEDESTRIAN_RADIUS ** 2:
                    collision = True
                    break

        dist = float((self.goal - self.pos).norm())
        reached = dist < self.GOAL_THRESHOLD
        reward = -self.DENSE_COEF * dist
        done = False
        info: dict = {"dist": dist, "reached": reached, "collision": collision}
        if collision:
            reward += self.COLLISION_REWARD
            done = True
        elif reached:
            reward += self.GOAL_REWARD
            done = True
        return Step(obs=self._current_obs(), reward=reward, done=done, info=info)


# ---------------------------------------------------------------------------
# PPO rollout + update
# ---------------------------------------------------------------------------


def gae_advantages(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns for a single trajectory.

    rewards, values, dones: shape [T]
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T)
    last_adv = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: SNCPPolicyLite,
    optimiser: torch.optim.Optimizer,
    obs_seq: torch.Tensor,  # [N, T, F]
    actions: torch.Tensor,  # [N, T, A]
    old_log_probs: torch.Tensor,  # [N, T]
    advantages: torch.Tensor,  # [N, T]
    returns: torch.Tensor,  # [N, T]
    clip: float,
    vf_coef: float,
    ent_coef: float,
    epochs: int,
) -> dict:
    """Run K epochs of PPO minibatch updates on the rollout buffer."""
    N, T, _ = obs_seq.shape
    policy_loss_acc = []
    value_loss_acc = []
    entropy_acc = []
    for _ in range(epochs):
        # Flatten [N, T] → [N*T] for one big minibatch update.
        x = obs_seq.reshape(N * T, 1, -1)  # treat each step as length-1
        # The lite policy encodes [B, T, F] — but T=1 means no real recurrence.
        # We feed the full T-length sequence per env instead:
        x = obs_seq  # [N, T, F]
        new_log_prob, entropy, value = policy.evaluate_actions(x, actions)
        ratio = (new_log_prob - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = (value - returns).pow(2).mean()
        entropy_loss = entropy.mean()
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimiser.step()
        policy_loss_acc.append(float(policy_loss.item()))
        value_loss_acc.append(float(value_loss.item()))
        entropy_acc.append(float(entropy_loss.item()))
    return {
        "policy_loss": statistics.fmean(policy_loss_acc),
        "value_loss": statistics.fmean(value_loss_acc),
        "entropy": statistics.fmean(entropy_acc),
    }


def collect_rollout(
    policy: SNCPPolicyLite,
    env: PointMassNavLite,
    episodes: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Collect ``episodes`` trajectories, each of length env.HORIZON (or shorter if done).

    Returns:
        obs_seq: [N, T_max, F]  (T_max = env.HORIZON; padded if early termination)
        actions: [N, T_max, A]
        log_probs: [N, T_max]
        advantages: [N, T_max]
        returns: [N, T_max]
        stats: dict of per-episode metrics
    """
    T = PointMassNavLite.HORIZON
    obs_dim = env.obs_dim
    A = PointMassNavLite.ACTION_DIM
    obs_buf = torch.zeros(episodes, T, obs_dim)
    action_buf = torch.zeros(episodes, T, A)
    log_prob_buf = torch.zeros(episodes, T)
    reward_buf = torch.zeros(episodes, T)
    value_buf = torch.zeros(episodes, T)
    done_buf = torch.zeros(episodes, T)
    ep_returns = []
    ep_reached = []
    ep_collision = []

    for ep in range(episodes):
        env_seed = seed * 1000 + ep
        torch.manual_seed(env_seed)
        obs = env.reset(seed=env_seed)
        h = None
        ep_reward = 0.0
        ep_steps = 0
        for t in range(T):
            x = obs.view(1, 1, -1)
            with torch.no_grad():
                action, log_prob, _, value, h = policy.act(x, h0=h)
            obs_buf[ep, t] = obs
            action_buf[ep, t] = action.squeeze(0)
            log_prob_buf[ep, t] = log_prob.squeeze(0)
            value_buf[ep, t] = value.squeeze(0)
            step = env.step(action.squeeze(0))
            reward_buf[ep, t] = step.reward
            done_buf[ep, t] = float(step.done)
            ep_reward += step.reward
            ep_steps += 1
            obs = step.obs
            if step.done:
                # Zero out the rest of the trajectory to avoid spurious gradients.
                for t_pad in range(t + 1, T):
                    done_buf[ep, t_pad] = 1.0
                break
        ep_returns.append(ep_reward)
        ep_reached.append(step.info.get("reached", False) if step else False)
        ep_collision.append(step.info.get("collision", False) if step else False)

    # Compute GAE per episode.
    advantages_buf = torch.zeros(episodes, T)
    returns_buf = torch.zeros(episodes, T)
    for ep in range(episodes):
        adv, ret = gae_advantages(
            reward_buf[ep],
            value_buf[ep],
            done_buf[ep],
            gamma=0.99,
            lam=0.95,
        )
        advantages_buf[ep] = adv
        returns_buf[ep] = ret

    # Normalise advantages across the rollout.
    advantages_buf = (advantages_buf - advantages_buf.mean()) / (advantages_buf.std() + 1e-8)

    stats = {
        "n_episodes": episodes,
        "mean_return": statistics.fmean(ep_returns) if ep_returns else 0.0,
        "std_return": statistics.stdev(ep_returns) if len(ep_returns) > 1 else 0.0,
        "reach_rate": sum(ep_reached) / max(1, len(ep_reached)),
        "collision_rate": sum(ep_collision) / max(1, len(ep_collision)),
        "ep_returns": ep_returns,
    }
    return obs_buf, action_buf, log_prob_buf, advantages_buf, returns_buf, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--ppo-updates", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--ltc-hidden", type=int, default=32)
    parser.add_argument("--trunk-hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="analysis/sncp_ppo_lite")
    parser.add_argument(
        "--n-pedestrians", type=int, default=0,
        help="Number of moving pedestrian obstacles (0 = static obstacles, backward compat with iter#26).",
    )
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Train sequentially over --ped-curriculum-list with parameters carried over.",
    )
    parser.add_argument(
        "--ped-curriculum-list", type=str, default="1,2,3",
        help="Comma-separated n_pedestrians per curriculum stage (only when --curriculum is set).",
    )
    parser.add_argument(
        "--ppo-updates-per-stage", type=int, default=20,
        help="PPO updates per curriculum stage (only when --curriculum is set).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Resolve curriculum stages.
    if args.curriculum:
        try:
            stages = [int(s.strip()) for s in args.ped_curriculum_list.split(",") if s.strip()]
        except ValueError:
            raise SystemExit(f"--ped-curriculum-list must be comma-separated ints, got {args.ped_curriculum_list!r}")
        if not stages:
            raise SystemExit("--ped-curriculum-list is empty; pass e.g. '1,2,3'.")
    else:
        stages = [args.n_pedestrians]
        args.ppo_updates_per_stage = args.ppo_updates  # alias for unified loop

    # Initial env (any of the stages — only obs_dim is needed to size the policy).
    first_env = PointMassNavLite(seed=args.seed, n_pedestrians=max(stages))
    policy = SNCPPolicyLite(
        temporal_input_size=first_env.obs_dim,
        ltc_hidden_size=args.ltc_hidden,
        trunk_hidden_size=args.trunk_hidden,
        action_dim=PointMassNavLite.ACTION_DIM,
        ode_method="euler",
    )
    optimiser = torch.optim.Adam(policy.parameters(), lr=args.lr)

    rollout_history = []
    ppo_history = []
    stage_history = []
    global_update = 0
    for stage_idx, n_ped in enumerate(stages):
        env = PointMassNavLite(seed=args.seed + stage_idx, n_pedestrians=n_ped)
        # obs_dim is constant across all stages (zero-padded ped slots);
        # policy is pre-sized to env.obs_dim and reused as-is.
        assert env.obs_dim == policy.temporal_input_size, (
            f"Stage {stage_idx} obs_dim={env.obs_dim} != policy.temporal_input_size="
            f"{policy.temporal_input_size}; this should be constant."
        )
        stage_start = global_update
        for upd in range(args.ppo_updates_per_stage):
            obs_buf, action_buf, log_prob_buf, advantages_buf, returns_buf, stats = collect_rollout(
                policy, env, args.episodes_per_update, seed=args.seed + global_update
            )
            rollout_history.append({
                "update": global_update,
                "stage": stage_idx,
                "n_pedestrians": n_ped,
                "mean_return": stats["mean_return"],
                "reach_rate": stats["reach_rate"],
                "collision_rate": stats["collision_rate"],
            })
            print(
                f"[stage {stage_idx} n_ped={n_ped} upd {upd:02d}] mean_return={stats['mean_return']:+.3f} "
                f"reach={stats['reach_rate']:.2f} collision={stats['collision_rate']:.2f}"
            )
            ppo_stats = ppo_update(
                policy,
                optimiser,
                obs_buf,
                action_buf,
                log_prob_buf,
                advantages_buf,
                returns_buf,
                clip=args.clip,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                epochs=args.epochs,
            )
            ppo_history.append(ppo_stats)
            global_update += 1
        # Per-stage summary
        stage_rollouts = [r for r in rollout_history if r["stage"] == stage_idx]
        stage_history.append({
            "stage": stage_idx,
            "n_pedestrians": n_ped,
            "updates": args.ppo_updates_per_stage,
            "mean_return_last_5": statistics.fmean([r["mean_return"] for r in stage_rollouts[-5:]]),
            "reach_rate_last_5": statistics.fmean([r["reach_rate"] for r in stage_rollouts[-5:]]),
            "collision_rate_last_5": statistics.fmean([r["collision_rate"] for r in stage_rollouts[-5:]]),
        })
        print(
            f"  >> stage {stage_idx} (n_ped={n_ped}) last-5 mean_return="
            f"{stage_history[-1]['mean_return_last_5']:+.3f} "
            f"reach={stage_history[-1]['reach_rate_last_5']:.2f} "
            f"collision={stage_history[-1]['collision_rate_last_5']:.2f}"
        )

    # Final eval (greedy) on the last (most crowded) stage.
    final_n_ped = stages[-1]
    eval_seed = args.seed + 9999
    env_eval = PointMassNavLite(seed=eval_seed, n_pedestrians=final_n_ped)
    obs = env_eval.reset(seed=eval_seed)
    h = None
    greedy_return = 0.0
    for t in range(PointMassNavLite.HORIZON):
        x = obs.view(1, 1, -1)
        with torch.no_grad():
            action, _, _, _, h = policy.act(x, h0=h)
        step = env_eval.step(action.squeeze(0))
        greedy_return += step.reward
        obs = step.obs
        if step.done:
            break

    payload = {
        "run_id": dt.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        "experiment": "sncp_ppo_lite_point_mass_nav",
        "config": vars(args),
        "stages": stages,
        "stage_history": stage_history,
        "rollout_history": rollout_history,
        "ppo_history": ppo_history,
        "final_greedy_return": greedy_return,
        "final_eval_n_pedestrians": final_n_ped,
        "policy_parameters": sum(p.numel() for p in policy.parameters()),
    }
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{payload['run_id']}_sncp_ppo_lite.json"
    md_path = output_dir / f"{payload['run_id']}_sncp_ppo_lite.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Markdown summary
    title_suffix = (
        f" curriculum (1→{max(stages)} peds)" if args.curriculum
        else f" n_pedestrians={args.n_pedestrians}"
    )
    lines = [
        f"# SNCP-PPO Lite (2D Point-Mass Nav) — {payload['run_id']}",
        "",
        f"- policy params: {payload['policy_parameters']:,}",
        f"- episodes/update × total PPO updates: {args.episodes_per_update} × {global_update} = {args.episodes_per_update * global_update} total episodes",
        f"- ltc_hidden / trunk_hidden: {args.ltc_hidden} / {args.trunk_hidden}",
        f"- seed: {args.seed}",
        f"- mode: {'curriculum' if args.curriculum else 'single'}",
        f"- final greedy return (n_pedestrians={final_n_ped}): {greedy_return:+.3f}",
        "",
    ]
    if stage_history:
        lines += [
            "## Per-stage summary (last 5 updates)",
            "",
            "| stage | n_ped | mean_return | reach_rate | collision_rate |",
            "|---:|---:|---:|---:|---:|",
        ]
        for s in stage_history:
            lines.append(
                f"| {s['stage']} | {s['n_pedestrians']} | {s['mean_return_last_5']:+.3f} | "
                f"{s['reach_rate_last_5']:.2f} | {s['collision_rate_last_5']:.2f} |"
            )
        lines.append("")
    lines += [
        f"## 学习曲线 (per update, stage suffix in 'n_ped' column){title_suffix}",
        "",
        "| update | stage | n_ped | mean_return | reach_rate | collision_rate |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rollout_history:
        lines.append(
            f"| {r['update']:02d} | {r['stage']} | {r['n_pedestrians']} | "
            f"{r['mean_return']:+.3f} | {r['reach_rate']:.2f} | {r['collision_rate']:.2f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nJSON: {json_path}")
    print(f"MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
