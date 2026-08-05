"""N23: Does int8 quantization's 'free-lunch 4x compression' hold under OOD dt?

N20 found that on in-dist dt, int8 quantization gives 4x compression with
zero accuracy loss. N23 tests whether this holds when the test dt
distribution SHIFTS (out-of-distribution).

If int8 quantization breaks under OOD dt (quantization error compounds
with retention's OOD sensitivity), this would be a significant finding
for edge deployment under variable sensor sampling rates.
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

from lnn.core.cfc import CfCCell
from lnn.core.distillation import (
    ActivationAlignedCfCNetwork,
    DistillConfig,
    DualStageDistiller,
)
from lnn.core.quantization import quantize_model_inplace


def make_ar2_with_timestamps(n_samples, seq_len, n_feat=4, seed=0, dt_log_sigma=0.0):
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
    parser.add_argument("--student-hidden", type=int, default=8)
    parser.add_argument("--teacher-retention", choices=["cfc", "hybrid_gate"], default="cfc")
    parser.add_argument("--date", default="2026-08-05")
    args = parser.parse_args()

    n_feat = 4
    out = 1
    hidden = args.student_hidden

    # Train on regular dt (sigma=0)
    torch.manual_seed(0)
    x_train, y_train, _ = make_ar2_with_timestamps(args.n_samples, seq_len=24, dt_log_sigma=0.0)
    n_tr = int(0.8 * x_train.shape[0])
    x_tr, y_tr = x_train[:n_tr], y_train[:n_tr]
    x_te_reg, y_te_reg = x_train[n_tr:], y_train[n_tr:]

    # Test on 3 dt distributions: regular, in-dist irregular, OOD irregular
    test_dts = {}
    for sigma in (0.0, 0.5, 1.0):
        torch.manual_seed(1)
        dt_test = torch.exp(torch.randn(*x_te_reg.shape[:2]) * sigma - 0.5 * sigma**2)
        dt_test[:, 0] = 1.0
        test_dts[sigma] = dt_test

    print(f"Teacher: {args.teacher_retention} (h={args.teacher_hidden}), student h={args.student_hidden}")
    print(f"Train dt: regular (sigma=0)")
    print(f"Test dt: regular (sigma=0), in-dist irregular (sigma=0.5), OOD irregular (sigma=1.0)\n")

    cfg = DistillConfig(
        input_size=n_feat, output_size=out,
        teacher_hidden=args.teacher_hidden,
        student_hiddens=(args.student_hidden,),
        epochs=args.epochs, batch=8, lr=1e-2,
        teacher_retention_kind=args.teacher_retention,
    )

    # Aggregate over repeats
    agg = {sigma: {"fp32": [], "int8": []} for sigma in (0.0, 0.5, 1.0)}
    for r in range(args.repeats):
        torch.manual_seed(42 + r)
        d = DualStageDistiller(cfg)
        d.run_pareto_sweep(x_tr, y_tr, x_te_reg, y_te_reg)
        student, _ = d.students[args.student_hidden]
        for sigma, dt_test in test_dts.items():
            # Custom forward with per-step dt (ActivationAlignedCfCNetwork doesn't accept dt kwarg)
            def student_forward(model, x, dt):
                b, t, _ = x.shape
                h = x.new_zeros(b, model.hidden_size)
                outs = []
                for ti in range(t):
                    dti = dt[:, ti] if isinstance(dt, torch.Tensor) else dt
                    if isinstance(dti, torch.Tensor) and dti.dim() == 1:
                        dti = dti.unsqueeze(-1)
                    h = model.cell(x[:, ti, :], h, dt=dti)
                    outs.append(model.readout(h))
                return torch.stack(outs, dim=1)
            with torch.no_grad():
                fp32_mse = nn.functional.mse_loss(student_forward(student, x_te_reg, dt_test), y_te_reg).item()
            quantize_model_inplace(student, per_channel=True)
            with torch.no_grad():
                int8_mse = nn.functional.mse_loss(student_forward(student, x_te_reg, dt_test), y_te_reg).item()
            agg[sigma]["fp32"].append(fp32_mse)
            agg[sigma]["int8"].append(int8_mse)

    # Build output
    rows = []
    for sigma in (0.0, 0.5, 1.0):
        v = agg[sigma]
        fp32_mean = statistics.mean(v["fp32"])
        int8_mean = statistics.mean(v["int8"])
        delta = int8_mean - fp32_mean
        rows.append({
            "test_dt_sigma": sigma,
            "fp32_mse_mean": fp32_mean,
            "int8_mse_mean": int8_mean,
            "delta": delta,
        })
        print(f"  σ_test={sigma}: fp32 MSE={fp32_mean:.4f}, int8 MSE={int8_mean:.4f}, "
              f"delta={delta:+.4f}")

    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.date}_int8_ood_dt_{args.teacher_retention}.json"
    md_path = out_dir / f"{args.date}_int8_ood_dt_{args.teacher_retention}.md"
    json_path.write_text(json.dumps({
        "date": args.date,
        "task": "int8_quantization_ood_dt_n23",
        "config": vars(args),
        "rows": rows,
    }, indent=2))

    md = [f"""---
title: Int8 quantization on OOD dt (N23) — {args.teacher_retention} teacher
date: {args.date}
tags: [LNN, int8, quantization, distillation, OOD-dt, N23]
---

# Int8 quantization on OOD dt (N23) — {args.teacher_retention} teacher

## Setup
- Teacher ({args.teacher_retention}, h={args.teacher_hidden}) → student (CfC, h={args.student_hidden})
- Train dt: regular (sigma=0)
- Test dt: regular (0), in-dist (0.5), OOD (1.0)

## Results

| test dt σ | fp32 MSE | int8 MSE | delta |
|---:|---:|---:|---:|
"""]
    for r in rows:
        md.append(f"| {r['test_dt_sigma']} | {r['fp32_mse_mean']:.4f} | "
                  f"{r['int8_mse_mean']:.4f} | **{r['delta']:+.4f}** |\n")
    md_path.write_text("".join(md))
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
