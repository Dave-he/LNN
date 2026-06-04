#!/usr/bin/env python3
"""SVAF τ-modulated peer-blending — 2-agent toy mesh (PRD §10 #9 stage A).

Replicates the minimum unit of the SVAF §7.1 collective-intelligence coupling
mechanism on a 2-agent toy mesh. Verifies the paper's core claim:

    "Fast neurons (τ < 5s) couple readily; slow neurons (τ > 30s) resist
     coupling — collective awareness through fast coupling, individual
     expertise through slow sovereignty."
    — arXiv 2604.03955v1 §7.1, Table 14

Setup:
- 2 agents A and B with hidden state dimension d = 6 (2 Fast + 2 Medium + 2 Slow)
- Agent A is initialised to `h_A = [1, 1, 1, 1, 1, 1]`
- Agent B (the peer) holds `h_B = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]` for N steps
- After N steps, we measure how close A's hidden state is to B's per τ group
  (smaller distance = more coupling; larger distance = more sovereignty)

Outputs:
- `analysis/svaf/<date>_tau_toy.json` (per-step state + per-group distances)
- `analysis/svaf/<date>_tau_toy.md`   (human summary + verdict)

This is the **stage A** mini-task of PRD §10 #9. Stage B (cfcs/blend with
real CfC backbone) is left for a future iter.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.cfc import (
    default_three_group_tau,
    tau_modulated_blend_update,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis" / "svaf"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def _format_markdown(payload: dict) -> str:
    n_fast = payload["n_fast"]
    n_med = payload["n_medium"]
    n_slow = payload["n_slow"]
    d_fast = payload["final_dist_fast"]
    d_med = payload["final_dist_medium"]
    d_slow = payload["final_dist_slow"]
    n_steps = payload["n_steps"]
    peer = payload["peer_value"]
    md = [
        "# SVAF τ-modulated peer-blending — 2-agent toy mesh",
        "",
        f"_Generated: {payload['generated_at']}_",
        f"_Source: scripts/experiment_svaf_tau_toy.py_",
        f"_Paper: arXiv 2604.03955v1 (Hongwei Xu 2026), §7.1 Eq. 20_",
        "",
        "## Setup",
        f"- hidden dim: {payload['d']}  (Fast × {n_fast} / Medium × {n_med} / Slow × {n_slow})",
        f"- peer value: {peer}  (constant across all N steps)",
        f"- N steps: {n_steps}",
        f"- α_eff = {payload['alpha_eff']}, K = {payload['K']}",
        f"- τ layout: Fast={payload['tau_fast']}, Medium={payload['tau_medium']}, Slow={payload['tau_slow']}",
        "",
        "## Final per-group distance to peer",
        "| Group | n_dims | τ | final distance to peer |",
        "|---|---:|---:|---:|",
        f"| Fast | {n_fast} | {payload['tau_fast']} | {d_fast:.4f} |",
        f"| Medium | {n_med} | {payload['tau_medium']} | {d_med:.4f} |",
        f"| Slow | {n_slow} | {payload['tau_slow']} | {d_slow:.4f} |",
        "",
        "## Verdict",
    ]
    if d_fast < d_med < d_slow:
        md.append("- ✅ Strict ordering Fast < Medium < Slow — the paper's claim holds:")
        md.append("  fast τ couples strongly, slow τ preserves sovereignty.")
    elif d_fast < d_slow and d_slow - d_fast > 0.05:
        md.append(f"- ✅ Loose ordering holds (Fast distance {d_fast:.4f} < Slow distance {d_slow:.4f}),")
        md.append("  but Medium does not strictly order between them.")
    else:
        md.append(f"- ❌ Ordering failed: Fast={d_fast:.4f}, Medium={d_med:.4f}, Slow={d_slow:.4f}")
        md.append("  — fast τ did not couple more than slow τ.")
    md.extend([
        "",
        "## Per-step trajectory (fast dim 0)",
        "",
        "| step | h_A[0] (fast) | h_B[0] |",
        "|---:|---:|---:|",
    ])
    for s in payload["trajectory"]:
        md.append(f"| {s['step']} | {s['h_A_0']:.4f} | {s['h_B_0']:.4f} |")
    md.append("")
    md.append(f"JSON 原数据: `{payload['json_path']}`")
    return "\n".join(md) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=6, help="hidden dim (default 6 = 2+2+2)")
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--peer-value", type=float, default=0.5)
    parser.add_argument("--alpha-eff", type=float, default=0.40,
                        help="peer-level blending strength (paper §3.4: 0.40 aligned)")
    parser.add_argument("--K", type=float, default=30.0,
                        help="scaling constant in β = αK × sim / τ (Eq. 20)")
    parser.add_argument("--out-prefix", type=str, default=dt.date.today().isoformat())
    args = parser.parse_args()

    # Build a 6-dim vector with 2 fast / 2 medium / 2 slow via default helper
    if args.d % 3 != 0:
        print(f"[tau-toy] warning: --d {args.d} not divisible by 3, using default helper")
    tau = default_three_group_tau(args.d)  # 1/3 each
    n_per_group = args.d // 3

    # Agent A: init to 1.0 (h_A), peer B: constant peer_value
    h_a = torch.ones(1, args.d)
    h_b = torch.full((1, args.d), float(args.peer_value))

    # Per-step record (only fast dim 0 for trajectory table)
    trajectory = []
    # Track per-group mean distance
    distances_fast = []
    distances_medium = []
    distances_slow = []
    for step in range(args.n_steps + 1):  # include step 0
        diff = (h_a - h_b).abs().mean(dim=-1).item()  # scalar mean abs diff
        distances_fast.append((h_a[0, :n_per_group] - h_b[0, :n_per_group]).abs().mean().item())
        distances_medium.append((h_a[0, n_per_group:2 * n_per_group] - h_b[0, n_per_group:2 * n_per_group]).abs().mean().item())
        distances_slow.append((h_a[0, 2 * n_per_group:] - h_b[0, 2 * n_per_group:]).abs().mean().item())
        trajectory.append({
            "step": step,
            "h_A_0": h_a[0, 0].item(),
            "h_B_0": h_b[0, 0].item(),
            "mean_abs_diff": diff,
        })
        if step < args.n_steps:
            h_a = tau_modulated_blend_update(h_a, h_b, tau, args.alpha_eff, args.K)

    final_dist_fast = distances_fast[-1]
    final_dist_medium = distances_medium[-1]
    final_dist_slow = distances_slow[-1]

    now = dt.datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S")
    payload = {
        "run_id": run_id,
        "generated_at": now.isoformat(timespec="seconds"),
        "d": args.d,
        "n_fast": n_per_group,
        "n_medium": n_per_group,
        "n_slow": args.d - 2 * n_per_group,
        "tau_fast": 1.0,
        "tau_medium": 10.0,
        "tau_slow": 60.0,
        "alpha_eff": args.alpha_eff,
        "K": args.K,
        "n_steps": args.n_steps,
        "peer_value": args.peer_value,
        "final_dist_fast": final_dist_fast,
        "final_dist_medium": final_dist_medium,
        "final_dist_slow": final_dist_slow,
        "trajectory": trajectory,
        "per_group_distances": {
            "fast": distances_fast,
            "medium": distances_medium,
            "slow": distances_slow,
        },
    }

    out_prefix = args.out_prefix
    json_path = ANALYSIS_DIR / f"{out_prefix}_tau_toy.json"
    md_path = ANALYSIS_DIR / f"{out_prefix}_tau_toy.md"
    payload["json_path"] = str(json_path.relative_to(ROOT))
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    md = _format_markdown(payload)
    with open(md_path, "w") as f:
        f.write(md)

    print(f"=== SVAF τ-blend 2-agent toy mesh (iter#22, PRD §10 #9 stage A) ===")
    print(f"  d={args.d}, n_steps={args.n_steps}, peer={args.peer_value}")
    print(f"  final distance: Fast={final_dist_fast:.4f}, Medium={final_dist_medium:.4f}, Slow={final_dist_slow:.4f}")
    if final_dist_fast < final_dist_slow:
        print("  ✅ Fast < Slow (paper §7.1 'fast couples readily, slow resists')")
    else:
        print("  ❌ Fast ≥ Slow — paper claim not observed")
    print(f"  wrote JSON: {json_path}")
    print(f"  wrote MD:   {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
