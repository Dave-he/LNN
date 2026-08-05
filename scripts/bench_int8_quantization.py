"""N20: int8 quantization on distillation students (DLNet Stage 3).

Pipeline (from N1 + N19):
  1. Distill teacher (CfC or hybrid_gate, h=32) to student (CfC, h in {4,8,12,16})
  2. Apply int8 per-channel quantization to student weights
  3. Measure (a) test MSE delta vs float32, (b) int8 size in bytes

Compares:
  - float32 student (N1/N19 baseline)
  - int8 per-channel student (Stage 3)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lnn.core.distillation import (
    DistillConfig,
    DualStageDistiller,
)
from lnn.core.quantization import (
    quantize_model_inplace,
    total_compressed_size_bytes,
    total_fp32_size_bytes,
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
    )

    # Aggregate over repeats
    agg = {}  # {hidden: {'fp32_mse': [], 'int8_mse': [], 'int8_bytes': []}}
    for r in range(args.repeats):
        torch.manual_seed(42 + r)
        d = DualStageDistiller(cfg)
        results = d.run_pareto_sweep(x_tr, y_tr, x_te, y_te)
        for p in results[1:]:  # skip teacher
            hidden = p.student_hidden
            agg.setdefault(hidden, {"fp32_mse": [], "int8_mse": [], "int8_bytes": [], "fp32_bytes": [], "params": []})
            # Use the saved trained student from the distiller
            student, proj = d.students[hidden]
            # Eval float32
            with torch.no_grad():
                fp32_mse = d.evaluate(student, x_te, y_te)
            # Apply int8
            meta = quantize_model_inplace(student, per_channel=True)
            int8_mse = d.evaluate(student, x_te, y_te)
            agg[hidden]["fp32_mse"].append(fp32_mse)
            agg[hidden]["int8_mse"].append(int8_mse)
            agg[hidden]["int8_bytes"].append(total_compressed_size_bytes(meta))
            agg[hidden]["fp32_bytes"].append(total_fp32_size_bytes(meta))
            agg[hidden]["params"].append(p.params)

    # Build output rows
    rows = []
    for h in sorted(agg.keys()):
        v = agg[h]
        rows.append({
            "hidden": h,
            "params": int(statistics.mean(v["params"])),
            "fp32_mse_mean": statistics.mean(v["fp32_mse"]),
            "fp32_mse_std": statistics.stdev(v["fp32_mse"]) if len(v["fp32_mse"]) > 1 else 0.0,
            "int8_mse_mean": statistics.mean(v["int8_mse"]),
            "int8_mse_std": statistics.stdev(v["int8_mse"]) if len(v["int8_mse"]) > 1 else 0.0,
            "int8_bytes": int(statistics.mean(v["int8_bytes"])),
            "fp32_bytes": int(statistics.mean(v["fp32_bytes"])),
        })
        r = rows[-1]
        delta = r["int8_mse_mean"] - r["fp32_mse_mean"]
        compression = r["fp32_bytes"] / r["int8_bytes"]
        print(f"  h={h}: params={r['params']}, "
              f"fp32 MSE={r['fp32_mse_mean']:.4f}, int8 MSE={r['int8_mse_mean']:.4f}, "
              f"delta={delta:+.4f}, "
              f"int8={r['int8_bytes']}B vs fp32={r['fp32_bytes']}B ({compression:.1f}×)")

    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.date}_int8_quantization_{args.teacher_retention}.json"
    md_path = out_dir / f"{args.date}_int8_quantization_{args.teacher_retention}.md"
    json_path.write_text(json.dumps({
        "date": args.date,
        "task": "int8_quantization_post_distillation",
        "config": vars(args),
        "rows": rows,
    }, indent=2))

    md = [f"""---
title: Int8 Quantization on Distillation Students (N20) — {args.teacher_retention} teacher
date: {args.date}
tags: [LNN, int8, quantization, distillation, edge-ai, DLNet, N20, Stage-3]
---

# Int8 Quantization on Distillation Students (N20) — {args.teacher_retention} teacher

## Setup
- Teacher ({args.teacher_retention}, h={args.teacher_hidden}) → student distillation
- Apply per-channel int8 quantization to student
- Compare float32 vs int8 MSE

## Results

| student h | params | fp32 MSE | int8 MSE | delta | int8 bytes | fp32 bytes | compression |
|---:|---:|---:|---:|---:|---:|---:|---:|
"""]
    for r in rows:
        delta = r["int8_mse_mean"] - r["fp32_mse_mean"]
        compression = r["fp32_bytes"] / r["int8_bytes"]
        md.append(f"| {r['hidden']} | {r['params']} | "
                  f"{r['fp32_mse_mean']:.4f} ± {r['fp32_mse_std']:.4f} | "
                  f"{r['int8_mse_mean']:.4f} ± {r['int8_mse_std']:.4f} | "
                  f"**{delta:+.4f}** | "
                  f"{r['int8_bytes']} | {r['fp32_bytes']} | "
                  f"**{compression:.1f}×** |\n")

    md_path.write_text("".join(md))
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
