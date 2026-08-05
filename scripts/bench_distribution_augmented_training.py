"""N15: Can distribution-augmented training fix hybrid_gate's OOD transferability?

Hypothesis (from N12):
    hybrid_gate's input-dep alpha overfits training dt distribution.
    Mixing multiple dt distributions during training may force alpha to
    learn a more general dt-robustness mechanism.

Setup:
    - Single-dist training (N12 baseline): dt ~ LogNormal(0, 0.5)
    - Mixed training (N15): per-batch dt randomly drawn from
        LogNormal(0, sigma) for sigma in {0.3, 0.5, 1.0}
    - Test on dt ~ LogNormal(0, sigma) for sigma in {0.3, 0.5, 1.0}

If mixed training gives degradation ratio ≈ 1.00x across all sigma_test,
then distribution-augmented training enables OOD transferability.
If not, then N12's finding stands and hybrid_gate fundamentally overfits.
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


def train_eval_mixed(model, x_base, y_base, sigmas_mix, x_te_reg, y_te_reg, x_te_irreg, y_te_irreg,
                     dt_te_irreg, epochs=4, batch=8, lr=1e-2):
    """Train with per-batch random dt sigma from sigmas_mix list."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x_base.shape[0]
    t0 = time.perf_counter()
    for _ in range(epochs):
        for b in range(0, n, batch):
            xb = x_base[b:b + batch]
            yb = y_base[b:b + batch]
            # Per-batch sample a dt sigma
            sigma_idx = torch.randint(0, len(sigmas_mix), (1,)).item()
            sigma = sigmas_mix[sigma_idx]
            torch.manual_seed(sigma_idx * 1000 + b)
            dtb = torch.exp(torch.randn(*xb.shape[:2]) * sigma - 0.5 * sigma**2)
            dtb[:, 0] = 1.0
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

    sigmas_train_mix = [0.3, 0.5, 1.0]
    sigma_tests = [0.3, 0.5, 1.0]
    n_feat = 4
    out = 1
    hidden = args.hidden

    # Build the same x/y data for all dt distributions (signal unchanged)
    torch.manual_seed(0)
    x_base, y_base, _ = make_ar2_with_timestamps(args.n_samples, args.seq_len, seed=0, dt_log_sigma=0.5)
    n_tr = int(0.8 * x_base.shape[0])
    x_tr, y_tr = x_base[:n_tr], y_base[:n_tr]
    x_te_reg, y_te_reg = x_base[n_tr:], y_base[n_tr:]

    # Pre-compute test dt tensors for each sigma
    test_dts = {}
    for sigma in sigma_tests:
        torch.manual_seed(1)
        dt_test = torch.exp(torch.randn(*x_te_reg.shape[:2]) * sigma - 0.5 * sigma**2)
        dt_test[:, 0] = 1.0
        test_dts[sigma] = dt_test

    print(f"Train dt: MIXED per-batch LogNormal(0, sigma) for sigma in {sigmas_train_mix}")
    print(f"Test dt: per-sigma LogNormal(0, sigma_test) for sigma_test in {sigma_tests}")
    for sigma in sigma_tests:
        ts = test_dts[sigma]
        print(f"  sigma_test={sigma}: range=[{ts.min().item():.3f}, {ts.max().item():.3f}], mean={ts.mean().item():.3f}")

    models = {
        "cfc-baseline (regular train only)": lambda: _SeqWrap(CfCCell(n_feat, hidden), out),
        "mfc-hybrid_gate (mixed dt train)":
            lambda: _SeqWrap(MemoryFusionCfCCell(n_feat, hidden, retention_kind="hybrid_gate"), out),
    }
    results = {}
    for name, factory in models.items():
        results[name] = {}
        for sigma in sigma_tests:
            msess_reg, msess_irreg = [], []
            for r in range(args.repeats):
                torch.manual_seed(42 + r)
                model = factory()
                res = train_eval_mixed(model, x_tr, y_tr, sigmas_train_mix,
                                       x_te_reg, y_te_reg,
                                       x_te_reg, y_te_reg, test_dts[sigma],
                                       epochs=args.epochs)
                msess_reg.append(res["test_mse_regular"])
                msess_irreg.append(res["test_mse_irregular"])
            results[name][sigma] = {
                "test_mse_regular_mean": statistics.mean(msess_reg),
                "test_mse_regular_std": statistics.stdev(msess_reg) if len(msess_reg) > 1 else 0.0,
                "test_mse_irregular_mean": statistics.mean(msess_irreg),
                "test_mse_irregular_std": statistics.stdev(msess_irreg) if len(msess_irreg) > 1 else 0.0,
                "degradation_ratio": statistics.mean(msess_irreg) / max(statistics.mean(msess_reg), 1e-9),
            }
        row = results[name]
        print(f"  {name}: " + " | ".join(
            f"σ={σ}: ratio={row[σ]['degradation_ratio']:.2f}x "
            f"irr={row[σ]['test_mse_irregular_mean']:.4f}"
            for σ in sigma_tests
        ))

    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.date}_distribution_augmented_training.json"
    md_path = out_dir / f"{args.date}_distribution_augmented_training.md"
    json_path.write_text(json.dumps({
        "date": args.date,
        "task": "distribution_augmented_training_for_OOD_transferability",
        "sigmas_train_mix": sigmas_train_mix,
        "sigma_tests": sigma_tests,
        "results": results,
    }, indent=2))

    md = [f"""---
title: Distribution-augmented training (N15): does it fix hybrid_gate OOD transferability? — {args.date}
date: {args.date}
tags: [LNN, CfC, hybrid_gate, distribution-augmented-training, OOD, transferability, N15]
---

# Distribution-augmented training (N15) — {args.date}

## Setup
- **Train dt**: per-batch random sample from LogNormal(0, sigma) for sigma in {sigmas_train_mix}
  (mixed distributions, force the model to see all 3 distributions during training)
- **Test dt**: per-sigma LogNormal(0, sigma_test) for sigma_test in {sigma_tests}
- 2 models x 3 sigma_test x 2 repeats x 4 epochs

## Results

| model | σ=0.3 | σ=0.5 | σ=1.0 |
|---|---:|---:|---:|
"""]
    for name in models:
        row = results[name]
        md.append(f"| {name} | "
                  f"reg={row[0.3]['test_mse_regular_mean']:.4f} irr={row[0.3]['test_mse_irregular_mean']:.4f} **{row[0.3]['degradation_ratio']:.2f}x** | "
                  f"reg={row[0.5]['test_mse_regular_mean']:.4f} irr={row[0.5]['test_mse_irregular_mean']:.4f} **{row[0.5]['degradation_ratio']:.2f}x** | "
                  f"reg={row[1.0]['test_mse_regular_mean']:.4f} irr={row[1.0]['test_mse_irregular_mean']:.4f} **{row[1.0]['degradation_ratio']:.2f}x** |\n")

    # N12 baseline for comparison
    md.append("""
## N12 baseline (single-distribution training, for comparison)

| model | σ=0.3 | σ=0.5 | σ=1.0 |
|---|---:|---:|---:|
""")
    n12_baseline = {
        "cfc-baseline":           {"0.3": 1.00, "0.5": 1.00, "1.0": 1.00},
        "mfc-hybrid_gate (N11)":  {"0.3": 1.01, "0.5": 1.04, "1.0": 1.10},
    }
    for name, ratios in n12_baseline.items():
        md.append(f"| {name} | **{ratios['0.3']:.2f}x** | **{ratios['0.5']:.2f}x** | **{ratios['1.0']:.2f}x** |\n")
    md.append("""
## Verdict (TBD — see results above)

If hybrid_gate mixed-train row stays around **1.00x across all sigma_test**,
distribution-augmented training fixes N12's finding (POSITIVE).
If it stays at 1.10x for OOD like N12, the finding stands (NEGATIVE).

The CfC row should be 1.00x across all (no learning, structural generic).
""")
    md_path.write_text("".join(md))
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
