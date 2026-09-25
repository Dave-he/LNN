#!/usr/bin/env python3
"""Multi-seed reach_rate + latency benchmark for distilled JEPA policies.

Phase C.3 of PRD-MSD-LNN. Addresses the seed-lucky rule from
``LNN_TLDR.md`` §43: any "+X% gain" must be reported across >= 3 seeds
with mean +/- std; single-seed results are advisory only.

This script:
1. Trains the JEPA policy 3 times with different seeds.
2. For each, rolls out on PointMassNavLite and reports reach_rate +
   p50/p99 latency.
3. Writes a JSON + MD summary with mean / std / min across seeds.

Re-uses the existing PD demos (no need to regenerate). Caller must
provide --checkpoint or train fresh.
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
from scripts.bench_jepa_e2e import _percentile  # noqa: E402
from scripts.experiment_sncp_ppo_lite import PointMassNavLite  # noqa: E402


def _ts() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


def train_one_seed(args, seed: int) -> tuple[JEPAPolicy, dict]:
    torch.manual_seed(seed)
    policy = JEPAPolicy(
        obs_dim=args.obs_dim, action_dim=args.action_dim,
        latent_dim=args.latent_dim, hidden_size=args.hidden_size,
        head_type=args.head_type, n_tau=args.n_tau,
    )
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # Load demos.
    demos_path = sorted((REPO_ROOT / "analysis" / "decisions").glob("*_pointmass_pid_demos_n0.json"))[-1]
    payload = json.loads(demos_path.read_text())
    obs = torch.tensor(payload["obs"], dtype=torch.float32)
    actions = torch.tensor(payload["actions"], dtype=torch.float32)
    if args.max_states is not None:
        obs = obs[:args.max_states]
        actions = actions[:args.max_states]
    N = obs.shape[0]

    history = []
    for epoch in range(args.epochs):
        perm = torch.randperm(N)
        running = 0.0
        n_batches = 0
        for s in range(0, N, args.batch_size):
            idx = perm[s:s + args.batch_size]
            z = policy.encoder.encode_step(obs[idx])
            pred = policy.policy_head(z, obs[idx])
            loss = torch.nn.functional.mse_loss(pred, actions[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
            n_batches += 1
        history.append(running / max(n_batches, 1))

    # Eval.
    return policy, {"history": history, "final_mse": history[-1], "obs_demos": demos_path.name}


def eval_policy(policy: JEPAPolicy, args, seed: int) -> dict:
    policy.eval()
    n_reached = n_collision = n_timeout = 0
    latencies_ms: list[float] = []
    returns: list[float] = []
    for ep in range(args.episodes):
        env = PointMassNavLite(seed=seed + ep, n_pedestrians=args.n_pedestrians)
        obs = env.reset(seed=seed + ep)
        if args.obs_dim == 17:
            obs = PDExpert.augmented_obs(env)
        ep_return = 0.0
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
                if result.info.get("reached"):
                    n_reached += 1
                if result.info.get("collision"):
                    n_collision += 1
                break
        else:
            n_timeout += 1
        returns.append(ep_return)
    latencies_ms.sort()
    return {
        "reach_rate": n_reached / max(args.episodes, 1),
        "collision_rate": n_collision / max(args.episodes, 1),
        "timeout_rate": n_timeout / max(args.episodes, 1),
        "avg_return": statistics.fmean(returns) if returns else 0.0,
        "p50_ms": _percentile(latencies_ms, 50.0),
        "p99_ms": _percentile(latencies_ms, 99.0),
        "p999_ms": _percentile(latencies_ms, 99.9),
        "n_steps": len(latencies_ms),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--obs-dim", type=int, default=17)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--n-tau", type=int, default=4)
    parser.add_argument("--head-type", choices=["mse", "gauss"], default="mse")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--n-pedestrians", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args(argv)

    per_seed_results = []
    for seed in args.seeds:
        print(f"[multiseed] === seed={seed} ===", flush=True)
        policy, train_info = train_one_seed(args, seed)
        eval_metrics = eval_policy(policy, args, seed)
        print(
            f"[multiseed] seed={seed} reach_rate={eval_metrics['reach_rate']:.3f} "
            f"p99_ms={eval_metrics['p99_ms']:.3f}",
            flush=True,
        )
        per_seed_results.append({
            "seed": seed,
            "train": train_info,
            "eval": eval_metrics,
        })

    # Aggregate.
    reach_rates = [r["eval"]["reach_rate"] for r in per_seed_results]
    p99_ms = [r["eval"]["p99_ms"] for r in per_seed_results]
    p50_ms = [r["eval"]["p50_ms"] for r in per_seed_results]
    avg_returns = [r["eval"]["avg_return"] for r in per_seed_results]

    summary = {
        "reach_rate_mean": statistics.fmean(reach_rates),
        "reach_rate_std": statistics.pstdev(reach_rates) if len(reach_rates) > 1 else 0.0,
        "reach_rate_min": min(reach_rates),
        "reach_rate_max": max(reach_rates),
        "p99_ms_mean": statistics.fmean(p99_ms),
        "p99_ms_std": statistics.pstdev(p99_ms) if len(p99_ms) > 1 else 0.0,
        "p50_ms_mean": statistics.fmean(p50_ms),
        "avg_return_mean": statistics.fmean(avg_returns),
    }
    print(
        f"[multiseed] reach_rate: mean={summary['reach_rate_mean']:.3f} "
        f"std={summary['reach_rate_std']:.3f} min={summary['reach_rate_min']:.3f} "
        f"max={summary['reach_rate_max']:.3f}",
        flush=True,
    )
    print(
        f"[multiseed] p99_ms: mean={summary['p99_ms_mean']:.3f} std={summary['p99_ms_std']:.3f}",
        flush=True,
    )

    out_dir = REPO_ROOT / "analysis" / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _ts()
    json_path = out_dir / f"{stamp}_jepa_multiseed.json"
    md_path = out_dir / f"{stamp}_jepa_multiseed.md"
    json_path.write_text(json.dumps({
        "config": vars(args),
        "per_seed": per_seed_results,
        "summary": summary,
    }, indent=2))

    md = [
        f"# JEPA multi-seed ({len(args.seeds)} seeds) — {stamp}",
        "",
        f"seeds: {args.seeds}",
        "",
        "## Per-seed",
        "",
        "| seed | reach_rate | p50_ms | p99_ms | avg_return |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in per_seed_results:
        md.append(
            f"| {r['seed']} | {r['eval']['reach_rate']:.3f} | "
            f"{r['eval']['p50_ms']:.3f} | {r['eval']['p99_ms']:.3f} | "
            f"{r['eval']['avg_return']:.3f} |"
        )
    md.extend([
        "",
        "## Aggregate",
        "",
        f"* reach_rate mean={summary['reach_rate_mean']:.3f} std={summary['reach_rate_std']:.3f}",
        f"* reach_rate min={summary['reach_rate_min']:.3f} max={summary['reach_rate_max']:.3f}",
        f"* p99_ms mean={summary['p99_ms_mean']:.3f} std={summary['p99_ms_std']:.3f}",
        f"* p50_ms mean={summary['p50_ms_mean']:.3f}",
        f"* avg_return mean={summary['avg_return_mean']:.3f}",
    ])
    md_path.write_text("\n".join(md) + "\n")
    print(f"[multiseed] wrote {json_path}")
    print(f"[multiseed] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())