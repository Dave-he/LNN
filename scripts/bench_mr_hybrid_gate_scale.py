"""N14: MR-hybrid_gate-CfC at h=64 — verify N13 honest finding.

N13 found that MR-hybrid_gate-CfC at h=24 (each expert gets 6 dim) is
11% worse than single-expert hybrid_gate because the multi-rate structure
doesn't help when per-expert hidden is too small.

N14 tests whether this gap closes at h=64 (each expert gets 16 dim, which
is the threshold from N3's Pareto sweep).
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
from lnn.core.memory_fusion_cfc import MemoryFusionCfCCell
from lnn.core.multirate_tfp_cfc import MultiRateTfpCfCNetwork


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


def train_eval(model, x_tr, y_tr, dt_tr, x_te_reg, y_te_reg, x_te_irreg, y_te_irreg, dt_te_irreg,
               epochs=4, batch=8, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x_tr.shape[0]
    t0 = time.perf_counter()
    for _ in range(epochs):
        for b in range(0, n, batch):
            xb = x_tr[b:b + batch]
            yb = y_tr[b:b + batch]
            dtb = dt_tr[b:b + batch]
            opt.zero_grad()
            pred = model(xb, dt=dtb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            opt.step()
    train_s = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        test_mse_reg = nn.functional.mse_loss(model(x_te_reg, dt=1.0), y_te_reg).item()
        test_mse_irreg = nn.functional.mse_loss(model(x_te_irreg, dt=dt_te_irreg), y_te_irreg).item()
    return {"test_mse_regular": test_mse_reg, "test_mse_irregular": test_mse_irreg, "train_seconds": train_s}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--hidden-list", type=int, nargs="+", default=[24, 32, 48, 64])
    parser.add_argument("--n-tau", type=int, default=4)
    parser.add_argument("--date", default="2026-08-05")
    args = parser.parse_args()

    sigma_train = 0.5
    n_feat = 4
    out = 1

    torch.manual_seed(0)
    x_irreg, y_irreg, dt_irreg = make_ar2_with_timestamps(
        args.n_samples, args.seq_len, seed=0, dt_log_sigma=sigma_train,
    )
    n_tr = int(0.8 * x_irreg.shape[0])
    x_tr, y_tr, dt_tr = x_irreg[:n_tr], y_irreg[:n_tr], dt_irreg[:n_tr]
    x_te_reg, y_te_reg = x_irreg[n_tr:], y_irreg[n_tr:]
    torch.manual_seed(1)
    dt_te_irreg = torch.exp(torch.randn(*x_te_reg.shape[:2]) * sigma_train - 0.5 * sigma_train**2)
    dt_te_irreg[:, 0] = 1.0

    print(f"Task: AR(2) + 3-regime + irregular dt (sigma={sigma_train})")
    print(f"Models: CfC (single), mfc-hybrid_gate (single), MR-hybrid_gate-CfC (n_tau={args.n_tau})")
    print(f"Sweep h ∈ {args.hidden_list}\n")

    results = {}  # {model: {h: {'mse': [...], 'train_s': [...]}}}
    for name in ("cfc", "mfc-hybrid_gate", "mr-hybrid-gate-cfc"):
        results[name] = {}
        for h in args.hidden_list:
            msess_reg, msess_irreg, trains = [], [], []
            for r in range(args.repeats):
                torch.manual_seed(42 + r)
                # n_tau=2, top_k_active=1 effectively gives single-expert (only 1 expert active)
                if name == "cfc":
                    model = MultiRateTfpCfCNetwork(
                        n_feat, h, out, n_tau=2, top_k_active=1,
                        expert_retention_kind="cfc",
                    )
                elif name == "mfc-hybrid_gate":
                    model = MultiRateTfpCfCNetwork(
                        n_feat, h, out, n_tau=2, top_k_active=1,
                        expert_retention_kind="hybrid_gate",
                    )
                else:
                    model = MultiRateTfpCfCNetwork(
                        n_feat, h, out, n_tau=args.n_tau,
                        expert_retention_kind="hybrid_gate",
                    )
                res = train_eval(model, x_tr, y_tr, dt_tr,
                                 x_te_reg, y_te_reg,
                                 x_te_reg, y_te_reg, dt_te_irreg,
                                 epochs=args.epochs)
                msess_reg.append(res["test_mse_regular"])
                msess_irreg.append(res["test_mse_irregular"])
                trains.append(res["train_seconds"])
            n_params = sum(p.numel() for p in model.parameters())
            per_expert = h // args.n_tau if name == "mr-hybrid-gate-cfc" else h
            results[name][h] = {
                "params": n_params,
                "per_expert_hidden": per_expert,
                "mse_reg_mean": statistics.mean(msess_reg),
                "mse_reg_std": statistics.stdev(msess_reg) if len(msess_reg) > 1 else 0.0,
                "mse_irreg_mean": statistics.mean(msess_irreg),
                "mse_irreg_std": statistics.stdev(msess_irreg) if len(msess_irreg) > 1 else 0.0,
                "train_s_mean": statistics.mean(trains),
            }
            r = results[name][h]
            print(f"  {name:25s} h={h:3d} (per_exp={per_expert:2d}, params={n_params:5d}): "
                  f"reg={r['mse_reg_mean']:.4f}  irr={r['mse_irreg_mean']:.4f}  "
                  f"train={r['train_s_mean']:.1f}s")

    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.date}_mr_hybrid_gate_scale.json"
    md_path = out_dir / f"{args.date}_mr_hybrid_gate_scale.md"
    json_path.write_text(json.dumps({
        "date": args.date,
        "task": "mr_hybrid_gate_scale_n14",
        "config": vars(args),
        "results": results,
    }, indent=2))

    md = [f"""---
title: MR-hybrid_gate-CfC at h≥64 (N14) — verify N13 honest finding — {args.date}
date: {args.date}
tags: [LNN, MR-MoE, hybrid_gate, multi-rate, scale-up, N14, n_tau=4]
---

# MR-hybrid_gate-CfC at h≥64 (N14) — verify N13 honest finding

## Setup
- Task: AR(2) + 3-regime + irregular dt (sigma=0.5)
- Models: CfC, mfc-hybrid_gate (single expert), MR-hybrid_gate-CfC (n_tau={args.n_tau})
- Sweep h ∈ {args.hidden_list}

## Results

| model | h | per_expert | params | reg MSE | irr MSE | train s |
|---|---:|---:|---:|---:|---:|---:|
"""]
    for name in ("cfc", "mfc-hybrid_gate", "mr-hybrid-gate-cfc"):
        for h in args.hidden_list:
            r = results[name][h]
            md.append(f"| {name} | {h} | {r['per_expert_hidden']} | {r['params']} | "
                      f"{r['mse_reg_mean']:.4f} ± {r['mse_reg_std']:.4f} | "
                      f"{r['mse_irreg_mean']:.4f} ± {r['mse_irreg_std']:.4f} | "
                      f"{r['train_s_mean']:.1f} |\n")
    md_path.write_text("".join(md))
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
