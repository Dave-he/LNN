#!/usr/bin/env python3
"""Distil the JEPAPolicyHead via supervised behaviour cloning.

Phase C.2 of PRD-MSD-LNN. Two distillation modes:

* ``--target pd`` (default): train ``JEPAPolicyHead`` to mimic the
  PD-expert's first-step action on each demo state. Uses the demos
  already collected by ``scripts/bench_pid_expert_demo.py``. This is
  the simplest and most reliable path: PD reaches the goal 100% of
  the time on n_ped=0, so cloning PD actions gets reach_rate >= 0.95
  in a few epochs.

* ``--target planner``: use ``LatentPlanner`` to generate
  world-model-optimal first-step actions and clone those instead.
  This decouples the policy from the PD-controller prior but
  requires a *trained* world model checkpoint (see
  ``scripts/train_jepa_world_model.py --save-checkpoint``).

After distillation, run ``scripts/bench_jepa_e2e.py --checkpoint
<ckpt>`` to confirm reach_rate > 0 with sub-3 ms latency.

Typical usage:
    python scripts/train_jepa_distilled_policy.py --target pd --epochs 30
    python scripts/train_jepa_distilled_policy.py --target planner \
        --world-model-ckpt checkpoints/jepa_<ts>.pt --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lnn.core.jepa_world_model import (  # noqa: E402
    JEPAWorldModel,
    JEPAPolicy,
    LatentPlanner,
)


def _ts() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


def load_latest_demos() -> tuple[torch.Tensor, torch.Tensor]:
    """Find latest PD demos (any n_ped); return ``(obs, first_step_action)``."""
    out_dir = REPO_ROOT / "analysis" / "decisions"
    candidates = sorted(out_dir.glob("*_pointmass_pid_demos_*.json"))
    if not candidates:
        raise FileNotFoundError(f"no *_pointmass_pid_demos_*.json under {out_dir}")
    payload = json.loads(candidates[-1].read_text())
    obs = torch.tensor(payload["obs"], dtype=torch.float32)
    actions = torch.tensor(payload["actions"], dtype=torch.float32)
    return obs, actions


def load_latest_jepa_ckpt() -> Path:
    ckpt_dir = REPO_ROOT / "checkpoints"
    candidates = sorted(ckpt_dir.glob("jepa_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no jepa_*.pt under {ckpt_dir}")
    return candidates[-1]


def distil_via_planner(
    policy: JEPAPolicy,
    obs: torch.Tensor,
    planner: LatentPlanner,
    *,
    batch_size: int = 256,
    max_states: Optional[int] = None,
) -> torch.Tensor:
    """Generate planner-chosen first-step actions for each obs state.

    Returns:
        actions: ``[N, action_dim]`` planner-chosen actions.
    """
    policy.eval()
    N = obs.shape[0]
    if max_states is not None:
        N = min(N, max_states)
        obs = obs[:N]
    actions = []
    for start in range(0, N, batch_size):
        batch = obs[start:start + batch_size]
        with torch.no_grad():
            z = policy.encoder.encode_step(batch)
            a = planner.distill_action(z)
        actions.append(a)
    return torch.cat(actions, dim=0)


def learned_reward_fn_from_obs(obs_full: torch.Tensor) -> callable:
    """Build a reward function that scores lower-distance-to-goal higher.

    The PointMassNavLite obs has ``[pos.x, pos.y, goal_dx, goal_dy, ...]``
    at positions [0, 1, 2, 3]. So distance to goal = sqrt(goal_dx² + goal_dy²).
    Reward = -distance.

    Returns a callable suitable as ``reward_fn`` for
    :meth:`LatentPlanner.rollout`. The callable receives the world
    model's ``z_next`` prediction; we need to convert it back to obs
    space, but since z is just a latent we don't have that
    inverse-mapping here. Instead, the planner will use the *current*
    obs (passed by the caller) as the reward reference.
    """
    # Note: in our setup, the planner rollout uses ``z0 = encoder(obs)``
    # then rolls out ``z_next``. We don't have z -> obs mapping. So we
    # use a different reward: constant per candidate (no per-step
    # refinement). This makes the planner effectively just sample
    # uniform actions and pick the one closest to zero — which is
    # *not* the same as the goal-distance reward.
    #
    # For real goal-distance planning we need an env simulator.
    # For now, leave the planner with its default reward.
    raise NotImplementedError(
        "learned reward from obs not yet wired — the planner's "
        "default latent-norm reward is the only path; for goal-"
        "directed planning you need a learned reward model."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--target", choices=["pd", "planner"], default="pd",
                        help="imitation source: PD expert or latent planner")
    parser.add_argument("--world-model-ckpt", type=Path, default=None,
                        help="(planner mode) path to trained JEPA state_dict")
    parser.add_argument("--demos", type=Path, default=None,
                        help="explicit demos file (default: latest PD demos)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obs-dim", type=int, default=17,
                        help="17 = 14 base obs + [theta, sin(theta), cos(theta)] "
                             "from PDExpert.augmented_obs (required for BC).")
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--n-tau", type=int, default=4)
    parser.add_argument("--head-type", choices=["mse", "gauss"], default="mse")
    parser.add_argument("--planner-horizon", type=int, default=4)
    parser.add_argument("--planner-candidates", type=int, default=16)
    parser.add_argument("--max-states", type=int, default=None,
                        help="limit training set size for speed")
    parser.add_argument("--save-checkpoint", action="store_true",
                        help="save distilled JEPAPolicy state_dict")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

    obs, pd_actions = load_latest_demos()
    if args.max_states is not None:
        obs = obs[:args.max_states]
        pd_actions = pd_actions[:args.max_states]
    N = obs.shape[0]
    print(f"[distil] target={args.target} N={N} obs_dim={args.obs_dim} "
          f"latent={args.latent_dim} hidden={args.hidden_size} n_tau={args.n_tau}",
          flush=True)

    # Build the JEPA policy (untrained world model + untrained head).
    policy = JEPAPolicy(
        obs_dim=args.obs_dim, action_dim=args.action_dim,
        latent_dim=args.latent_dim, hidden_size=args.hidden_size,
        head_type=args.head_type, n_tau=args.n_tau,
    )

    # If planner mode: load world model + train it briefly, OR use existing ckpt.
    target_actions = pd_actions
    if args.target == "planner":
        if args.world_model_ckpt is None:
            ckpt_path = load_latest_jepa_ckpt()
        else:
            ckpt_path = args.world_model_ckpt
        print(f"[distil] loading world-model from {ckpt_path}", flush=True)
        policy.load_state_dict(torch.load(ckpt_path))
        # Build planner with the freshly-loaded world model.
        wm = policy.world_model
        wm.eval()
        planner = LatentPlanner(
            wm, horizon=args.planner_horizon,
            n_candidates=args.planner_candidates, action_dim=args.action_dim,
        )
        target_actions = distil_via_planner(policy, obs, planner,
                                            batch_size=args.batch_size,
                                            max_states=args.max_states)

    # Distillation: train JEPAPolicyHead via BC on (encoder(obs) -> target_action).
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        perm = torch.randperm(N)
        running = 0.0
        n_batches = 0
        for start in range(0, N, args.batch_size):
            idx = perm[start:start + args.batch_size]
            x = obs[idx]
            y = target_actions[idx]
            z = policy.encoder.encode_step(x)
            if policy.head_obs_skip:
                pred = policy.policy_head(z, x)
            else:
                pred = policy.policy_head(z)
            loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
            n_batches += 1
        avg = running / max(n_batches, 1)
        history.append({"epoch": epoch + 1, "mse": avg})
        if args.log_every and (epoch + 1) % args.log_every == 0:
            print(f"[distil] epoch {epoch+1}/{args.epochs} mse={avg:.5f}", flush=True)

    initial_mse = history[0]["mse"]
    final_mse = history[-1]["mse"]
    print(f"[distil] final_mse={final_mse:.5f} initial_mse={initial_mse:.5f} "
          f"improvement={initial_mse / max(final_mse, 1e-9):.2f}x", flush=True)

    # Persist outputs.
    stamp = _ts()
    out_dir = REPO_ROOT / "analysis" / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stamp}_jepa_distilled_{args.target}.json"
    md_path = out_dir / f"{stamp}_jepa_distilled_{args.target}.md"

    payload = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_states": N,
        "history": history,
        "initial_mse": initial_mse,
        "final_mse": final_mse,
        "improvement_ratio": initial_mse / max(final_mse, 1e-9),
    }
    json_path.write_text(json.dumps(payload, indent=2))

    md = [
        f"# JEPA distillation ({args.target}) — {stamp}",
        "",
        f"Target source: **{args.target}** ({N} states)",
        f"Epochs: {args.epochs}, batch: {args.batch_size}, lr: {args.lr}",
        "",
        f"Initial MSE: **{initial_mse:.5f}**",
        f"Final MSE:   **{final_mse:.5f}**",
        f"Improvement: **{initial_mse / max(final_mse, 1e-9):.2f}x**",
        "",
        "Next step: ``python scripts/bench_jepa_e2e.py --checkpoint "
        "<ckpt>`` to confirm reach_rate > 0.",
    ]
    md_path.write_text("\n".join(md) + "\n")
    print(f"[distil] wrote {json_path}")
    print(f"[distil] wrote {md_path}")

    if args.save_checkpoint:
        ckpt_dir = REPO_ROOT / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = ckpt_dir / f"jepa_distilled_{args.target}_{stamp}.pt"
        torch.save(policy.state_dict(), ckpt)
        print(f"[distil] checkpoint {ckpt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())