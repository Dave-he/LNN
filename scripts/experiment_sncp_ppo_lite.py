"""PPO + SNCPPolicyLite smoke experiment on a 2D point-mass nav task.

This is a *minimal* reproducible smoke (not a real crowdnav). The goal is to
demonstrate that ``lnn.core.sncp_policy_lite.SNCPPolicyLite`` (LTC encoder +
actor-critic) can be trained end-to-end with PPO on a simple task, not to
match heimdilon/sncp-ppo-crowdnav's 86% hard scenario success rate.

Task (2D point-mass nav):
- Start at origin (0, 0). Goal at (1, 1).
- 2 static obstacles at (-0.2, 0.3) and (0.5, -0.3) (radius 0.2).
- Action: [v ∈ [-0.1, 0.1], w ∈ [-π/2, π/2]] (linear/angular velocity).
- Dynamics: x_{t+1} = x_t + v·[cos θ_t, sin θ_t]·dt; θ_{t+1} = θ_t + w·dt
  where dt=1.0, horizon T=20.
- Reward: +10 on goal; -10 on collision; -0.1·|p_t - goal| dense.

PPO:
- Roll out N=8 episodes per update, K=4 epochs of minibatch updates.
- clip=0.2, vf_coef=0.5, ent_coef=0.01.
- 30 PPO updates ≈ 240 episodes.

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
    """2D point-mass navigation with 2 static obstacles."""

    OBS_DIM: int = 4
    ACTION_DIM: int = 2
    HORIZON: int = 20
    DT: float = 1.0
    GOAL_REWARD: float = 10.0
    COLLISION_REWARD: float = -10.0
    DENSE_COEF: float = 0.1
    GOAL_THRESHOLD: float = 0.15
    OBSTACLE_RADIUS: float = 0.2

    def __init__(self, seed: int = 0) -> None:
        self.goal = torch.tensor([1.0, 1.0])
        self.obstacles = [(-0.2, 0.3), (0.5, -0.3)]
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        self.pos = torch.zeros(2)
        self.theta = 0.0
        return torch.tensor(
            [self.pos[0].item(), self.pos[1].item(), (self.goal - self.pos)[0].item(), (self.goal - self.pos)[1].item()]
        )

    def step(self, action: torch.Tensor) -> Step:
        v = float(action[0].clamp(-0.1, 0.1))
        w = float(action[1].clamp(-math.pi / 2, math.pi / 2))
        self.pos[0] += v * math.cos(self.theta) * self.DT
        self.pos[1] += v * math.sin(self.theta) * self.DT
        self.theta += w * self.DT
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))  # wrap

        # Collision check.
        collision = False
        for ox, oy in self.obstacles:
            if (self.pos[0].item() - ox) ** 2 + (self.pos[1].item() - oy) ** 2 < self.OBSTACLE_RADIUS ** 2:
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
        return Step(
            obs=torch.tensor([self.pos[0].item(), self.pos[1].item(), (self.goal - self.pos)[0].item(), (self.goal - self.pos)[1].item()]),
            reward=reward,
            done=done,
            info=info,
        )


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
    obs_dim = PointMassNavLite.OBS_DIM
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
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    env = PointMassNavLite(seed=args.seed)
    policy = SNCPPolicyLite(
        temporal_input_size=PointMassNavLite.OBS_DIM,
        ltc_hidden_size=args.ltc_hidden,
        trunk_hidden_size=args.trunk_hidden,
        action_dim=PointMassNavLite.ACTION_DIM,
        ode_method="euler",
    )
    optimiser = torch.optim.Adam(policy.parameters(), lr=args.lr)

    rollout_history = []
    ppo_history = []
    for upd in range(args.ppo_updates):
        obs_buf, action_buf, log_prob_buf, advantages_buf, returns_buf, stats = collect_rollout(
            policy, env, args.episodes_per_update, seed=args.seed + upd
        )
        rollout_history.append({
            "update": upd,
            "mean_return": stats["mean_return"],
            "reach_rate": stats["reach_rate"],
            "collision_rate": stats["collision_rate"],
        })
        print(
            f"[upd {upd:02d}] mean_return={stats['mean_return']:+.3f} "
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
        print(
            f"           policy_loss={ppo_stats['policy_loss']:+.4f} "
            f"value_loss={ppo_stats['value_loss']:+.4f} entropy={ppo_stats['entropy']:+.4f}"
        )

    # Final eval (greedy): take argmax of the mean.
    eval_seed = args.seed + 999
    env_eval = PointMassNavLite(seed=eval_seed)
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
        "rollout_history": rollout_history,
        "ppo_history": ppo_history,
        "final_greedy_return": greedy_return,
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
    lines = [
        f"# SNCP-PPO Lite (2D Point-Mass Nav) — {payload['run_id']}",
        "",
        f"- policy params: {payload['policy_parameters']:,}",
        f"- episodes/update × PPO updates: {args.episodes_per_update} × {args.ppo_updates} = {args.episodes_per_update * args.ppo_updates} total episodes",
        f"- ltc_hidden / trunk_hidden: {args.ltc_hidden} / {args.trunk_hidden}",
        f"- seed: {args.seed}",
        f"- final greedy return (1 episode): {greedy_return:+.3f}",
        "",
        "## 学习曲线 (mean return per PPO update)",
        "",
        "| update | mean_return | reach_rate | collision_rate |",
        "|---:|---:|---:|---:|",
    ]
    for r in rollout_history:
        lines.append(f"| {r['update']:02d} | {r['mean_return']:+.3f} | {r['reach_rate']:.2f} | {r['collision_rate']:.2f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nJSON: {json_path}")
    print(f"MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
