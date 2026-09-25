#!/usr/bin/env python3
"""End-to-end benchmark for a trained JEPA decision policy.

Phase C of PRD-MSD-LNN. Loads a JEPAPolicy checkpoint (or builds a
fresh one if none exists), rolls it out on ``PointMassNavLite``, and
reports:

  * reach_rate across N episodes
  * p50 / p99 latency per env step (forward + state-carry)
  * parameter count and model size

The benchmark intentionally uses the *untrained* PPO policy head —
the JEPA forward_inference() path does encoder -> policy head only,
and the policy head has not been PPO-finetuned yet. Reach_rate is
expected to be lower than PD-expert's 1.000 until distillation
(``scripts/train_jepa_distilled_policy.py``) lands.

Typical usage:
    python scripts/bench_jepa_e2e.py --episodes 100 --seed 7
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lnn.core.jepa_world_model import JEPAPolicy  # noqa: E402
from scripts.bench_pid_expert_demo import PDExpert  # noqa: E402
from scripts.experiment_sncp_ppo_lite import PointMassNavLite  # noqa: E402


def _ts() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="path to jepa_*.pt state_dict (default: build fresh)")
    parser.add_argument("--obs-dim", type=int, default=17,
                        help="17 = 14 base obs + [theta, sin(theta), cos(theta)]; "
                             "matches PD-augmented demos used for distillation.")
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--n-tau", type=int, default=1)
    parser.add_argument("--head-type", choices=["mse", "gauss"], default="mse")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--n-pedestrians", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--quantize-int8", action="store_true",
                        help="apply weight-only INT8 quant after build")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    policy = JEPAPolicy(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        latent_dim=args.latent_dim,
        hidden_size=args.hidden_size,
        head_type=args.head_type,
        n_tau=args.n_tau,
    )
    if args.checkpoint is not None:
        policy.load_state_dict(torch.load(args.checkpoint))
    policy.eval()

    if args.quantize_int8:
        try:
            from lnn.core.quantization import quantize_model_inplace
            quantize_model_inplace(policy, per_channel=True)
        except Exception as e:  # pragma: no cover
            print(f"[warn] quantization skipped: {e}", file=sys.stderr)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[bench-jepa] n_params={n_params}", flush=True)

    # Per-step latency histogram
    latencies_ms: list[float] = []
    n_reached = 0
    n_collision = 0
    n_timeout = 0
    returns: list[float] = []

    for ep in range(args.episodes):
        env = PointMassNavLite(seed=args.seed + ep, n_pedestrians=args.n_pedestrians)
        obs = env.reset(seed=args.seed + ep)
        # If the model expects augmented obs (17 dims), wrap env obs with theta.
        if args.obs_dim == 17:
            obs = PDExpert.augmented_obs(env)
        ep_return = 0.0
        reached = False
        collision = False
        for _ in range(args.max_steps):
            t0 = time.perf_counter()
            with torch.no_grad():
                action = policy.forward_inference(obs.unsqueeze(0), deterministic=True).squeeze(0)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            result = env.step(action)
            ep_return += float(result.reward)
            obs = result.obs
            if args.obs_dim == 17:
                obs = PDExpert.augmented_obs(env)
            if result.done:
                reached = bool(result.info.get("reached", False))
                collision = bool(result.info.get("collision", False))
                break
        else:
            n_timeout += 1
        if reached:
            n_reached += 1
        if collision:
            n_collision += 1
        returns.append(ep_return)

    latencies_ms.sort()
    p50 = _percentile(latencies_ms, 50.0)
    p99 = _percentile(latencies_ms, 99.0)
    p999 = _percentile(latencies_ms, 99.9)
    mean_ms = statistics.fmean(latencies_ms)
    std_ms = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    sps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")

    reach_rate = n_reached / max(args.episodes, 1)
    collision_rate = n_collision / max(args.episodes, 1)
    timeout_rate = n_timeout / max(args.episodes, 1)
    avg_return = statistics.fmean(returns) if returns else 0.0

    print(
        f"[bench-jepa] reach_rate={reach_rate:.3f} collision_rate={collision_rate:.3f} "
        f"timeout_rate={timeout_rate:.3f} avg_return={avg_return:.3f}",
        flush=True,
    )
    print(
        f"[bench-jepa] latency p50={p50:.3f}ms p99={p99:.3f}ms p999={p999:.3f}ms "
        f"mean={mean_ms:.3f}ms std={std_ms:.3f}ms sps={sps:.0f}",
        flush=True,
    )

    out_dir = REPO_ROOT / "analysis" / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _ts()
    json_path = out_dir / f"{stamp}_jepa_e2e.json"
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    payload = {
        "config": cfg,
        "n_params": n_params,
        "n_episodes": args.episodes,
        "n_reached": n_reached,
        "n_collision": n_collision,
        "n_timeout": n_timeout,
        "reach_rate": reach_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "avg_return": avg_return,
        "latency_ms": {
            "p50": p50, "p99": p99, "p999": p999, "mean": mean_ms, "std": std_ms,
            "steps_per_sec": sps, "n_steps": len(latencies_ms),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2))
    md_path = json_path.with_suffix(".md")
    md_path.write_text(
        f"# JEPA e2e benchmark ({stamp})\n\n"
        f"* n_params: **{n_params}**\n"
        f"* reach_rate: **{reach_rate:.3f}**\n"
        f"* collision_rate: **{collision_rate:.3f}**\n"
        f"* timeout_rate: **{timeout_rate:.3f}**\n"
        f"* avg_return: **{avg_return:.3f}**\n"
        f"* latency p50: **{p50:.3f} ms**\n"
        f"* latency p99: **{p99:.3f} ms**\n"
        f"* latency p999: **{p999:.3f} ms**\n"
        f"* steps_per_sec: **{sps:.0f}**\n"
        f"* checkpoint: {args.checkpoint}\n"
        f"* quantize_int8: {args.quantize_int8}\n"
    )
    print(f"[bench-jepa] wrote {json_path}")
    print(f"[bench-jepa] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())