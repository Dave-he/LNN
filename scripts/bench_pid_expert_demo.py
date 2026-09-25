#!/usr/bin/env python3
"""PD-controller expert demo generator for PointMassNavLite.

Phase B of PRD-MSD-LNN. Generates ``(s_t, a_t, s_{t+1})`` transition triples
from the in-repo 2D point-mass env (``scripts/experiment_sncp_ppo_lite.py::PointMassNavLite``)
using a simple proportional-derivative (PD) controller. The repo previously
had no PD controller and no transition triple dataset — only a heuristic
goal-pointing expert in ``SyntheticImitationDataset``. The PD controller is
stronger (avoids obstacles via reactive steering) and gives IL warm-start
a much better policy prior to bootstrap PPO.

The controller is intentionally simple and deterministic:

* linear velocity v = clamp(P_v * dist_to_goal, 0, v_max)
* angular velocity w = clamp(P_w * heading_error + D_w * (-prev_heading_error), -w_max, +w_max)
* static obstacle repulsion: add a small lateral push proportional to the
  inverse-distance gradient to the nearest obstacle (n_pedestrians=0 mode).

Outputs:
* analysis/decisions/<ts>_pointmass_pid_demos.json  — statistics
* analysis/decisions/<ts>_pointmass_pid_demos.md    — human-readable summary

Typical usage:
    python scripts/bench_pid_expert_demo.py --n-episodes 5000 --export
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_sncp_ppo_lite import PointMassNavLite  # noqa: E402


# ---------------------------------------------------------------------------
# PD controller
# ---------------------------------------------------------------------------


@dataclass
class PDParams:
    P_v: float = 0.10           # linear velocity gain (per unit distance)
    P_w: float = 2.5            # angular velocity gain (per radian error)
    D_w: float = 0.6            # angular damping
    v_max: float = 0.10         # matches env action clip
    w_max: float = math.pi / 2  # matches env action clip
    obstacle_radius: float = 0.20
    obstacle_push: float = 0.06  # lateral push strength when near an obstacle


class PDExpert:
    """Hand-tuned PD controller for ``PointMassNavLite``.

    The controller's *forward direction* is the agent's ``theta`` (heading).
    It computes:

    1. desired heading: ``atan2(goal_y - pos_y, goal_x - pos_x)``
    2. heading error: wrapped to ``[-pi, pi]``
    3. linear velocity: forward speed proportional to Euclidean distance,
       clipped to env limit.
    4. obstacle avoidance: when within ``2 * obstacle_radius`` of a static
       obstacle, add a lateral push perpendicular to the goal direction.

    Returns action ``[v, w]`` in the env's coordinate frame.
    """

    def __init__(self, params: Optional[PDParams] = None) -> None:
        self.p = params or PDParams()
        self._prev_heading_err: Optional[float] = None

    def reset(self) -> None:
        self._prev_heading_err = None

    def __call__(self, obs: torch.Tensor, env: PointMassNavLite) -> tuple[float, float]:
        """Compute ``(v, w)`` from the current obs.

        Note: we re-derive pos and goal from the env (more robust than parsing
        the obs tensor, which has padding).
        """
        pos = env.pos
        goal = env.goal
        theta = env.theta

        dx = float(goal[0] - pos[0])
        dy = float(goal[1] - pos[1])
        dist = math.sqrt(dx * dx + dy * dy)
        desired_heading = math.atan2(dy, dx)
        heading_err = math.atan2(
            math.sin(desired_heading - theta),
            math.cos(desired_heading - theta),
        )

        # Linear velocity: drive forward proportional to distance, slow near goal.
        v = self.p.P_v * dist
        if dist < env.GOAL_THRESHOLD:
            v = 0.0
        v = max(0.0, min(self.p.v_max, v))

        # Angular velocity: PD on heading error.
        d_term = 0.0
        if self._prev_heading_err is not None:
            d_term = self.p.D_w * (heading_err - self._prev_heading_err)
        self._prev_heading_err = heading_err
        w = self.p.P_w * heading_err + d_term
        # Avoid huge spinning when error is small.
        w = max(-self.p.w_max, min(self.p.w_max, w))

        # Obstacle avoidance: add a small lateral nudge when near any static obstacle.
        for (ox, oy) in env.obstacles:
            r = math.sqrt((float(pos[0]) - ox) ** 2 + (float(pos[1]) - oy) ** 2)
            if r < 2.0 * self.p.obstacle_radius:
                # Vector from obstacle to agent, perpendicular to goal heading.
                away_x = float(pos[0]) - ox
                away_y = float(pos[1]) - oy
                # Cross-product sign to choose left/right.
                cross = dx * away_y - dy * away_x
                sign = 1.0 if cross > 0 else -1.0
                strength = self.p.obstacle_push * (2.0 * self.p.obstacle_radius - r) / (
                    2.0 * self.p.obstacle_radius
                )
                w += sign * strength

        w = max(-self.p.w_max, min(self.p.w_max, w))

        return v, w


# ---------------------------------------------------------------------------
# Demo collection
# ---------------------------------------------------------------------------


def collect_demos(
    n_episodes: int,
    n_pedestrians: int = 0,
    max_steps_per_episode: int = 50,
    seed: int = 42,
    controller: Optional[PDExpert] = None,
) -> dict:
    """Roll out the PD expert on ``PointMassNavLite`` and collect ``(obs, action, next_obs)``.

    Returns a dict with:
      - transitions: list of (obs_t.tolist(), action.tolist(), obs_tp1.tolist())
      - per_episode: list of {reached: bool, collision: bool, steps: int, return: float}
      - summary: aggregated metrics
    """
    controller = controller or PDExpert()
    transitions: list[tuple[list[float], list[float], list[float]]] = []
    per_episode: list[dict] = []

    for ep in range(n_episodes):
        env = PointMassNavLite(seed=seed + ep, n_pedestrians=n_pedestrians)
        controller.reset()
        obs = env.reset(seed=seed + ep)
        ep_return = 0.0
        ep_reached = False
        ep_collision = False
        n_steps = 0

        for _step in range(max_steps_per_episode):
            v, w = controller(obs, env)
            action = torch.tensor([v, w])
            next_step = env.step(action)
            next_obs = next_step.obs
            ep_return += float(next_step.reward)
            n_steps += 1
            transitions.append((obs.tolist(), action.tolist(), next_obs.tolist()))
            obs = next_obs
            if next_step.done:
                ep_reached = bool(next_step.info.get("reached", False))
                ep_collision = bool(next_step.info.get("collision", False))
                break
        else:
            # Episode timed out — record final distance in info via obs.
            dx = float(env.goal[0] - env.pos[0])
            dy = float(env.goal[1] - env.pos[1])
            ep_reached = math.sqrt(dx * dx + dy * dy) < env.GOAL_THRESHOLD

        per_episode.append(
            {
                "ep": ep,
                "reached": ep_reached,
                "collision": ep_collision,
                "steps": n_steps,
                "return": ep_return,
            }
        )

    n_reached = sum(1 for e in per_episode if e["reached"])
    n_collision = sum(1 for e in per_episode if e["collision"])
    reach_rate = n_reached / max(n_episodes, 1)
    collision_rate = n_collision / max(n_episodes, 1)
    avg_return = sum(e["return"] for e in per_episode) / max(n_episodes, 1)
    avg_steps = sum(e["steps"] for e in per_episode) / max(n_episodes, 1)

    return {
        "transitions": transitions,
        "per_episode": per_episode,
        "summary": {
            "n_episodes": n_episodes,
            "n_pedestrians": n_pedestrians,
            "n_transitions": len(transitions),
            "n_reached": n_reached,
            "n_collision": n_collision,
            "reach_rate": reach_rate,
            "collision_rate": collision_rate,
            "avg_return": avg_return,
            "avg_steps": avg_steps,
            "controller": {
                "P_v": controller.p.P_v,
                "P_w": controller.p.P_w,
                "D_w": controller.p.D_w,
                "v_max": controller.p.v_max,
                "w_max": controller.p.w_max,
                "obstacle_push": controller.p.obstacle_push,
            },
        },
    }


def _ts() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


def write_outputs(demos: dict, n_pedestrians: int) -> tuple[Path, Path]:
    out_dir = REPO_ROOT / "analysis" / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _ts()
    json_path = out_dir / f"{stamp}_pointmass_pid_demos_n{n_pedestrians}.json"
    md_path = out_dir / f"{stamp}_pointmass_pid_demos_n{n_pedestrians}.md"

    # JSON: drop the heavy per-transition list to keep file manageable;
    # record obs/action/next_obs as numpy arrays for downstream loaders.
    payload = {
        "summary": demos["summary"],
        "per_episode": demos["per_episode"],
        # transitions: store as flat tensors for downstream IL warm-start.
        "obs": torch.tensor([t[0] for t in demos["transitions"]]).tolist(),
        "actions": torch.tensor([t[1] for t in demos["transitions"]]).tolist(),
        "next_obs": torch.tensor([t[2] for t in demos["transitions"]]).tolist(),
    }
    json_path.write_text(json.dumps(payload))

    s = demos["summary"]
    md = [
        f"# PointMass PID expert demos (n_ped={n_pedestrians}) — {stamp}",
        "",
        "Phase B of PRD-MSD-LNN. PD controller rollouts on ``PointMassNavLite``.",
        "",
        f"* n_episodes: **{s['n_episodes']}**",
        f"* n_transitions: **{s['n_transitions']}**",
        f"* reach_rate: **{s['reach_rate']:.3f}** ({s['n_reached']}/{s['n_episodes']})",
        f"* collision_rate: **{s['collision_rate']:.3f}** ({s['n_collision']}/{s['n_episodes']})",
        f"* avg_return: **{s['avg_return']:.3f}**",
        f"* avg_steps: **{s['avg_steps']:.2f}**",
        "",
        "Controller gains:",
        f"* P_v={s['controller']['P_v']}, P_w={s['controller']['P_w']}, D_w={s['controller']['D_w']}",
        f"* v_max={s['controller']['v_max']}, w_max={s['controller']['w_max']:.3f}",
        f"* obstacle_push={s['controller']['obstacle_push']}",
        "",
        "Stored as ``obs / actions / next_obs`` flat tensors in the JSON companion.",
        "Use ``scripts/experiment_sncp_ppo_lite.py --il-warmstart`` to load and train.",
    ]
    md_path.write_text("\n".join(md) + "\n")
    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--n-episodes", type=int, default=2000)
    parser.add_argument("--n-pedestrians", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export", action="store_true",
                        help="write JSON + MD outputs under analysis/decisions/")
    parser.add_argument("--P-v", type=float, default=PDParams.P_v)
    parser.add_argument("--P-w", type=float, default=PDParams.P_w)
    parser.add_argument("--D-w", type=float, default=PDParams.D_w)
    parser.add_argument("--obstacle-push", type=float, default=PDParams.obstacle_push)
    args = parser.parse_args(argv)

    params = PDParams(
        P_v=args.P_v, P_w=args.P_w, D_w=args.D_w,
        obstacle_push=args.obstacle_push,
    )
    controller = PDExpert(params)

    print(
        f"[pid] n_episodes={args.n_episodes} n_ped={args.n_pedestrians} "
        f"P_v={params.P_v} P_w={params.P_w} D_w={params.D_w}", flush=True,
    )
    demos = collect_demos(
        n_episodes=args.n_episodes,
        n_pedestrians=args.n_pedestrians,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        controller=controller,
    )
    s = demos["summary"]
    print(
        f"[pid] reach_rate={s['reach_rate']:.3f} collision_rate={s['collision_rate']:.3f} "
        f"avg_steps={s['avg_steps']:.1f} n_transitions={s['n_transitions']}",
        flush=True,
    )

    if args.export:
        jp, mp = write_outputs(demos, args.n_pedestrians)
        print(f"[pid] wrote {jp}")
        print(f"[pid] wrote {mp}")

    # always return non-zero exit code only if reach_rate is way off the
    # controller's expected quality (>0.5 on n_ped=0 is the SLO).
    if args.n_pedestrians == 0 and s["reach_rate"] < 0.5:
        print(f"[pid] WARN reach_rate {s['reach_rate']:.3f} below SLO 0.5", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())