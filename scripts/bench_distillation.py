"""N1: Pareto sweep for DLNet-style LNN dual-stage distillation.

Reproduces the DLNet (arXiv 2601.06227) Pareto-sweep experiment on a
synthetic non-stationary AR(2) task. Sweeps student hidden sizes and
records (params, test MSE, train_seconds) for each.

CfC backbone (per N12+N16 finding: structural-generic dt-robustness).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lnn.core.distillation import (
    DistillConfig,
    DualStageDistiller,
)


def make_ar2_regime(n_samples, seq_len, n_feat=4, seed=0, dt_log_sigma=0.0):
    torch.manual_seed(seed)
    x = torch.zeros(n_samples, seq_len, n_feat)
    y = torch.zeros(n_samples, seq_len, 1)
    dt = torch.ones(n_samples, seq_len)
    if dt_log_sigma > 0:
        dt = torch.exp(torch.randn(n_samples, seq_len) * dt_log_sigma - 0.5 * dt_log_sigma**2)
        dt[:, 0] = 1.0
    for s in range(n_samples):
        regime = torch.randint(0, 3, (1,)).item()
        ar1, ar2 = ((0.6, 0.2), (-0.3, 0.5), (0.4, -0.4))[regime]
        noise = torch.randn(seq_len, n_feat) * 0.1
        for t in range(1, seq_len):
            x[s, t] = ar1 * x[s, t - 1] + ar2 * x[s, max(t - 2, 0)] + noise[t]
        y[s, :-1, 0] = x[s, 1:, :].sum(-1)
    return x, y, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=192)
    parser.add_argument("--teacher-hidden", type=int, default=32)
    parser.add_argument("--student-hiddens", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--teacher-retention", choices=["cfc", "hybrid_gate"], default="cfc")
    parser.add_argument("--student-retention", choices=["cfc", "hybrid_gate"], default="cfc")
    parser.add_argument("--date", default="2026-08-05")
    args = parser.parse_args()

    torch.manual_seed(0)
    x, y, dt = make_ar2_regime(args.n_samples, seq_len=24, dt_log_sigma=0.5)
    n_tr = int(0.8 * x.shape[0])
    x_tr, y_tr = x[:n_tr], y[:n_tr]
    x_te, y_te = x[n_tr:], y[n_tr:]

    cfg = DistillConfig(
        input_size=4, output_size=1,
        teacher_hidden=args.teacher_hidden,
        student_hiddens=tuple(args.student_hiddens),
        epochs=args.epochs, batch=8, lr=1e-2,
        teacher_retention_kind=args.teacher_retention,
        student_retention_kind=getattr(args, "student_retention", "cfc"),
    )

    # Run repeats and aggregate
    agg = {}  # {hidden: {'params': [], 'mse': [], 'train_s': []}}
    for r in range(args.repeats):
        torch.manual_seed(42 + r)
        d = DualStageDistiller(cfg)
        results = d.run_pareto_sweep(x_tr, y_tr, x_te, y_te)
        for p in results:
            agg.setdefault(p.student_hidden, {"params": [], "mse": [], "train_s": []})
            agg[p.student_hidden]["params"].append(p.params)
            agg[p.student_hidden]["mse"].append(p.test_mse)
            agg[p.student_hidden]["train_s"].append(p.train_seconds)

    # Build output rows
    rows = []
    for h in sorted(agg.keys()):
        v = agg[h]
        rows.append({
            "hidden": h,
            "params_mean": statistics.mean(v["params"]),
            "test_mse_mean": statistics.mean(v["mse"]),
            "test_mse_std": statistics.stdev(v["mse"]) if len(v["mse"]) > 1 else 0.0,
            "train_seconds_mean": statistics.mean(v["train_s"]),
        })
        print(f"  hidden={h}: params={rows[-1]['params_mean']}, "
              f"MSE={rows[-1]['test_mse_mean']:.4f}±{rows[-1]['test_mse_std']:.4f}, "
              f"train_s={rows[-1]['train_seconds_mean']:.2f}")

    # Find Pareto frontier: points not dominated by any other
    # (smaller params AND lower MSE are both better; here we focus on params-MSE)
    print("\nPareto frontier (params vs test MSE):")
    pareto = []
    for i, r1 in enumerate(rows):
        dominated = False
        for j, r2 in enumerate(rows):
            if i == j:
                continue
            if (r2["params_mean"] <= r1["params_mean"]
                    and r2["test_mse_mean"] <= r1["test_mse_mean"]
                    and (r2["params_mean"] < r1["params_mean"]
                         or r2["test_mse_mean"] < r1["test_mse_mean"])):
                dominated = True
                break
        if not dominated:
            pareto.append(r1)
            print(f"  hidden={r1['hidden']}: {r1['params_mean']} params, "
                  f"MSE={r1['test_mse_mean']:.4f}")

    # Compression ratio vs teacher
    teacher_row = next((r for r in rows if r["hidden"] == args.teacher_hidden), None)
    if teacher_row:
        print(f"\nCompression vs teacher (h={args.teacher_hidden}):")
        for r in rows:
            if r["hidden"] != args.teacher_hidden:
                ratio = teacher_row["params_mean"] / r["params_mean"]
                mse_delta = r["test_mse_mean"] - teacher_row["test_mse_mean"]
                print(f"  h={r['hidden']}: {ratio:.2f}× smaller params, "
                      f"MSE delta={mse_delta:+.4f}")

    # Save
    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Filename includes both teacher and student retention kinds
    base_name = f"{args.date}_distillation_{args.teacher_retention}_to_{args.student_retention}"
    json_path = out_dir / f"{base_name}.json"
    md_path = out_dir / f"{base_name}.md"
    json_path.write_text(json.dumps({
        "date": args.date,
        "task": "ar2_regime_distillation",
        "config": vars(args),
        "rows": rows,
        "pareto_frontier": pareto,
    }, indent=2))

    md = [f"""---
title: DLNet-style LNN Distillation Pareto Sweep — {args.date} (teacher={args.teacher_retention} → student={args.student_retention})
date: {args.date}
tags: [LNN, distillation, pareto, edge-ai, DLNet, knowledge-distillation, dual-stage, N1]
arxiv_refs: [2601.06227, 2106.13898]
---

# DLNet-style LNN Distillation Pareto Sweep — {args.date}

## Setup
- Teacher hidden: {args.teacher_hidden}
- Student hiddens: {args.student_hiddens}
- Task: non-stationary AR(2) + 3-regime + irregular dt (sigma=0.5)
- {args.repeats} repeats × {args.epochs} epochs

## Results

| student hidden | params | test MSE | train seconds |
|---:|---:|---:|---:|
"""]
    for r in rows:
        md.append(f"| {r['hidden']} | {r['params_mean']} | "
                  f"{r['test_mse_mean']:.4f} ± {r['test_mse_std']:.4f} | "
                  f"{r['train_seconds_mean']:.2f} |\n")

    md.append("\n## Pareto frontier\n\n")
    if pareto:
        for r in pareto:
            md.append(f"- hidden={r['hidden']}: {r['params_mean']} params, "
                      f"MSE={r['test_mse_mean']:.4f}\n")
    md.append("\n## Compression ratio vs teacher\n\n")
    if teacher_row:
        for r in rows:
            if r["hidden"] != args.teacher_hidden:
                ratio = teacher_row["params_mean"] / r["params_mean"]
                md.append(f"- h={r['hidden']}: {ratio:.2f}× smaller, "
                          f"MSE delta={r['test_mse_mean'] - teacher_row['test_mse_mean']:+.4f}\n")

    md_path.write_text("".join(md))
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
