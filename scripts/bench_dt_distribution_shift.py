"""N12: hybrid_gate transferability under dt-distribution shift.

Hypothesis:
    Hybrid_gate's input-dependent alpha may either:
    (A) Learn the *training* dt-distribution and underperform at OOD dt, OR
    (B) Learn a *general* dt-robustness mechanism that transfers.

Setup:
    Train on dt ~ LogNormal(0, sigma_train).
    Test on dt ~ LogNormal(0, sigma_test) for sigma_test ∈ {0.3, 0.5, 1.0}.
    Compare against cfc (always 1.00x) and mfc-tfp (worse on irregular).

If hybrid_gate's degradation ratio at (sigma_train=0.5, sigma_test=1.0) ≈ 1.00x,
then α learned a general mechanism (positive transferability).
If degradation > 1.05x, then α overfits to training dt-distribution (negative).
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


def make_ar2_with_timestamps(n_samples, seq_len, n_feat=4, seed=0, dt_log_sigma=0.0):
    torch.manual_seed(seed)
    x = torch.zeros(n_samples, seq_len, n_feat)
    y = torch.zeros(n_samples, seq_len, 1)
    dt = torch.ones(n_samples, seq_len)
    if dt_log_sigma > 0:
        dt = torch.exp(torch.randn(n_samples, seq_len) * dt_log_sigma - 0.5 * dt_log_sigma ** 2)
        dt[:, 0] = 1.0
    for s in range(n_samples):
        regime = torch.randint(0, 3, (1,)).item()
        ar1, ar2 = ((0.6, 0.2), (-0.3, 0.5), (0.4, -0.4))[regime]
        noise = torch.randn(seq_len, n_feat) * 0.1
        for t in range(1, seq_len):
            x[s, t] = ar1 * x[s, t - 1] + ar2 * x[s, max(t - 2, 0)] + noise[t]
        y[s, :-1, 0] = x[s, 1:, :].sum(-1)
    return x, y, dt


class _SeqWrap(nn.Module):
    def __init__(self, cell, out_dim):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(cell.hidden_size, out_dim)

    def forward(self, x, dt=1.0):
        b, t, _ = x.shape
        h = torch.zeros(b, self.cell.hidden_size)
        outs = []
        for i in range(t):
            dt_i = dt[:, i] if isinstance(dt, torch.Tensor) else dt
            if isinstance(dt_i, torch.Tensor) and dt_i.dim() == 1:
                dt_i = dt_i.unsqueeze(-1)
            h = self.cell(x[:, i, :], h, dt=dt_i)
            outs.append(self.head(h))
        return torch.stack(outs, dim=1)


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
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=192)
    parser.add_argument("--date", default="2026-08-05")
    args = parser.parse_args()

    sigma_train = 0.5
    sigma_tests = [0.3, 0.5, 1.0]  # in-dist, in-dist, OOD
    n_feat = 4
    out = 1
    hidden = args.hidden

    # Build training set (irregular dt)
    torch.manual_seed(0)
    x_irreg, y_irreg, dt_irreg = make_ar2_with_timestamps(
        args.n_samples, args.seq_len, seed=0, dt_log_sigma=sigma_train,
    )
    n_tr = int(0.8 * x_irreg.shape[0])
    x_tr, y_tr, dt_tr = x_irreg[:n_tr], y_irreg[:n_tr], dt_irreg[:n_tr]
    x_te_reg, y_te_reg = x_irreg[n_tr:], y_irreg[n_tr:]

    # Build test sets for each sigma_test
    test_sets = {}
    for sigma in sigma_tests:
        torch.manual_seed(1)
        dt_test = torch.exp(torch.randn(*x_te_reg.shape[:2]) * sigma - 0.5 * sigma**2)
        dt_test[:, 0] = 1.0
        test_sets[sigma] = dt_test

    print(f"Train dt: LogNormal(0, {sigma_train})")
    print(f"Test regular dt: 1.0")
    for sigma in sigma_tests:
        ts = test_sets[sigma]
        print(f"Test LogNormal(0, {sigma}): range=[{ts.min().item():.3f}, {ts.max().item():.3f}], mean={ts.mean().item():.3f}")

    models = {
        "cfc-baseline": lambda: _SeqWrap(CfCCell(n_feat, hidden), out),
        "mfc-tfp":      lambda: _SeqWrap(MemoryFusionCfCCell(n_feat, hidden, retention_kind="tfp"), out),
        "mfc-hybrid":   lambda: _SeqWrap(MemoryFusionCfCCell(n_feat, hidden, retention_kind="hybrid"), out),
        "mfc-hybrid_gate": lambda: _SeqWrap(MemoryFusionCfCCell(n_feat, hidden, retention_kind="hybrid_gate"), out),
    }
    results = {}  # {model_name: {sigma: {test_mse, degradation}}}
    for name, factory in models.items():
        results[name] = {}
        for sigma in sigma_tests:
            msess_reg, msess_irreg, trains = [], [], []
            for r in range(args.repeats):
                torch.manual_seed(42 + r)
                model = factory()
                res = train_eval(model, x_tr, y_tr, dt_tr,
                                 x_te_reg, y_te_reg,
                                 x_te_reg, y_te_reg, test_sets[sigma],
                                 epochs=args.epochs)
                msess_reg.append(res["test_mse_regular"])
                msess_irreg.append(res["test_mse_irregular"])
                trains.append(res["train_seconds"])
            results[name][sigma] = {
                "test_mse_regular_mean": statistics.mean(msess_reg),
                "test_mse_regular_std": statistics.stdev(msess_reg) if len(msess_reg) > 1 else 0.0,
                "test_mse_irregular_mean": statistics.mean(msess_irreg),
                "test_mse_irregular_std": statistics.stdev(msess_irreg) if len(msess_irreg) > 1 else 0.0,
                "degradation_ratio": statistics.mean(msess_irreg) / max(statistics.mean(msess_reg), 1e-9),
            }
        row = results[name]
        print(f"  {name}: " + " | ".join(
            f"σ_test={σ}: reg={row[σ]['test_mse_regular_mean']:.4f} "
            f"irr={row[σ]['test_mse_irregular_mean']:.4f} "
            f"ratio={row[σ]['degradation_ratio']:.2f}x"
            for σ in sigma_tests
        ))

    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.date}_dt_distribution_shift.json"
    md_path = out_dir / f"{args.date}_dt_distribution_shift.md"
    json_path.write_text(json.dumps({
        "date": args.date,
        "task": "dt_distribution_shift_transferability",
        "sigma_train": sigma_train,
        "sigma_tests": sigma_tests,
        "results": results,
    }, indent=2))

    md = [f"""---
title: dt distribution shift transferability (N12): hybrid_gate vs CfC vs TFP — {args.date}
date: {args.date}
tags: [LNN, CfC, TFP, hybrid_gate, dt-distribution-shift, transferability, robustness, N12]
---

# dt distribution shift transferability (N12) — {args.date}

## Setup
- Train dt: LogNormal(0, {sigma_train}) — IRREGULAR (in-dist for σ_test=0.5)
- Test regular: dt = 1.0
- Test σ_test ∈ {{0.3, 0.5, 1.0}}: in-dist (0.5), similar (0.3), OOD (1.0)
- 2 repeats × 4 epochs

## Results

| model | σ_test=0.3 reg | irr | ratio | σ_test=0.5 reg | irr | ratio | σ_test=1.0 reg | irr | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""]
    for name in models:
        row = results[name]
        md.append(f"| {name} | "
                  f"{row[0.3]['test_mse_regular_mean']:.4f} | {row[0.3]['test_mse_irregular_mean']:.4f} | **{row[0.3]['degradation_ratio']:.2f}x** | "
                  f"{row[0.5]['test_mse_regular_mean']:.4f} | {row[0.5]['test_mse_irregular_mean']:.4f} | **{row[0.5]['degradation_ratio']:.2f}x** | "
                  f"{row[1.0]['test_mse_regular_mean']:.4f} | {row[1.0]['test_mse_irregular_mean']:.4f} | **{row[1.0]['degradation_ratio']:.2f}x** |\n")
    md.append("""
## Interpretation

A model that *learns general dt-robustness* should have degradation ratio
roughly constant across σ_test values. A model that *overfits to training
dt-distribution* will see degradation increase with σ_test distance from
σ_train=0.5.

Key questions:
1. Does mfc-hybrid_gate transfer (ratio ≈ 1.00x across σ_test)?
2. Or does it overfit (ratio grows with |σ_test - 0.5|)?
3. Compare against mfc-tfp (which should overfit most strongly).
""")
    md_path.write_text("".join(md))
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
