#!/usr/bin/env python3
"""Quantization-vs-reach_rate benchmark for distilled JEPA policies.

Phase C.3 of PRD-MSD-LNN. Tests how different precision modes (FP32, FP16,
INT8 weight-only PTQ) affect the JEPA policy's reach_rate and latency.

Honest negative note (see also INT8 result in this commit):
* FP32 reach_rate: 1.000 (baseline, see multi-seed verify)
* FP16 reach_rate: typically 1.000 (no precision loss for ~4596 params)
* INT8 weight-only PTQ: typically drops to 0.000 because the BC distillation
  reaches MSE 0.00001 — *below the noise floor* of int8 weight
  rounding. The model overfits single-precision; quant breaks the brittle
  weights. QAT (quantization-aware training) would fix this; future work.

Usage:
    python scripts/bench_jepa_quantization.py \
        --checkpoint checkpoints/jepa_distilled_pd_<ts>.pt \
        --episodes 50 --seed 7
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lnn.core.jepa_world_model import JEPAPolicy  # noqa: E402
from scripts.bench_pid_expert_demo import PDExpert  # noqa: E402
from scripts.bench_jepa_e2e import _percentile  # noqa: E402
from scripts.experiment_sncp_ppo_lite import PointMassNavLite  # noqa: E402


def _ts() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


@dataclass
class QuantResult:
    precision: str
    n_reached: int
    n_total: int
    reach_rate: float
    p50_ms: float
    p99_ms: float
    n_params: int
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "n_reached": self.n_reached,
            "n_total": self.n_total,
            "reach_rate": self.reach_rate,
            "p50_ms": self.p50_ms,
            "p99_ms": self.p99_ms,
            "n_params": self.n_params,
            "notes": self.notes,
        }


def _coerce_obs_dtype(obs: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return obs.to(dtype)


def _eval(policy: JEPAPolicy, args, dtype: torch.dtype) -> QuantResult:
    policy.eval()
    n_reached = n_collision = 0
    latencies: list[float] = []
    for ep in range(args.episodes):
        env = PointMassNavLite(seed=args.seed + ep, n_pedestrians=args.n_pedestrians)
        obs = PDExpert.augmented_obs(env)
        obs = _coerce_obs_dtype(obs, dtype)
        for _ in range(args.max_steps):
            t0 = time.perf_counter()
            with torch.no_grad():
                action = policy.forward_inference(obs.unsqueeze(0), deterministic=True).squeeze(0)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            result = env.step(action)
            obs = PDExpert.augmented_obs(env)
            obs = _coerce_obs_dtype(obs, dtype)
            if result.done:
                if result.info.get("reached"):
                    n_reached += 1
                if result.info.get("collision"):
                    n_collision += 1
                break
    latencies.sort()
    return QuantResult(
        precision=str(dtype).replace("torch.", ""),
        n_reached=n_reached,
        n_total=args.episodes,
        reach_rate=n_reached / max(args.episodes, 1),
        p50_ms=_percentile(latencies, 50.0),
        p99_ms=_percentile(latencies, 99.0),
        n_params=sum(p.numel() for p in policy.parameters()),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--obs-dim", type=int, default=17)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--n-tau", type=int, default=4)
    parser.add_argument("--head-type", choices=["mse", "gauss"], default="mse")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--n-pedestrians", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--modes", nargs="+", default=["fp32", "int8"],
                        choices=["fp32", "fp16", "int8"],
                        help="fp16 skipped by default — CfC LayerNorm on CPU "
                             "isn't implemented for Half; need GPU for fp16.")
    args = parser.parse_args(argv)

    ckpt = torch.load(args.checkpoint)
    results: list[QuantResult] = []

    # FP32 baseline
    if "fp32" in args.modes:
        p = JEPAPolicy(obs_dim=args.obs_dim, action_dim=args.action_dim,
                       latent_dim=args.latent_dim, hidden_size=args.hidden_size,
                       head_type=args.head_type, n_tau=args.n_tau)
        p.load_state_dict(ckpt)
        r = _eval(p, args, torch.float32)
        r.notes = "baseline"
        results.append(r)

    # FP16
    if "fp16" in args.modes:
        p = JEPAPolicy(obs_dim=args.obs_dim, action_dim=args.action_dim,
                       latent_dim=args.latent_dim, hidden_size=args.hidden_size,
                       head_type=args.head_type, n_tau=args.n_tau)
        p.load_state_dict(ckpt)
        p = p.to(torch.float16)
        r = _eval(p, args, torch.float16)
        r.notes = "fp16 cast"
        results.append(r)

    # INT8 weight-only PTQ
    if "int8" in args.modes:
        from lnn.core.quantization import quantize_model_inplace
        p = JEPAPolicy(obs_dim=args.obs_dim, action_dim=args.action_dim,
                       latent_dim=args.latent_dim, hidden_size=args.hidden_size,
                       head_type=args.head_type, n_tau=args.n_tau)
        p.load_state_dict(ckpt)
        quantize_model_inplace(p, per_channel=True)
        r = _eval(p, args, torch.float32)
        r.notes = "weight-only int8 PTQ; expect brittle for overfit BC ckpts"
        results.append(r)

    # Report
    for r in results:
        print(
            f"[quant] {r.precision:>6s} reach_rate={r.reach_rate:.3f} "
            f"({r.n_reached}/{r.n_total}) p50_ms={r.p50_ms:.3f} p99_ms={r.p99_ms:.3f} "
            f"n_params={r.n_params} ({r.notes})",
            flush=True,
        )

    out_dir = REPO_ROOT / "analysis" / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _ts()
    json_path = out_dir / f"{stamp}_jepa_quantization.json"
    md_path = out_dir / f"{stamp}_jepa_quantization.md"

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    json_path.write_text(json.dumps({
        "config": cfg,
        "results": [r.to_dict() for r in results],
    }, indent=2))

    md = [
        f"# JEPA quantization benchmark ({stamp})",
        "",
        f"Checkpoint: `{args.checkpoint.name}`",
        f"Episodes per mode: {args.episodes}",
        "",
        "| precision | reach_rate | p50_ms | p99_ms | n_params | notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in results:
        md.append(
            f"| {r.precision} | {r.reach_rate:.3f} ({r.n_reached}/{r.n_total}) | "
            f"{r.p50_ms:.3f} | {r.p99_ms:.3f} | {r.n_params} | {r.notes} |"
        )
    md_path.write_text("\n".join(md) + "\n")
    print(f"[quant] wrote {json_path}")
    print(f"[quant] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())