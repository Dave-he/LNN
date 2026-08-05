"""N22: alpha capacity hypothesis — does deeper/wider α MLP enable generic
dt-robustness?

N15 found that distribution-augmented training partially fixed α OOD
overfitting (1.10x -> 1.07x). The hypothesis was that α MLP is too
small to learn generic dt-robustness (only interpolation).

N22 tests whether a LARGER α MLP (more depth, more width) can break
the interpolation ceiling and achieve CfC-level OOD performance (1.00x
across all sigma_test).
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


def train_eval_mixed(model, x_tr, y_tr, sigmas_mix, x_te_reg, y_te_reg,
                     x_te_irreg, y_te_irreg, dt_te_irreg, epochs=4, batch=8, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x_tr.shape[0]
    for _ in range(epochs):
        for b in range(0, n, batch):
            xb = x_tr[b:b + batch]
            yb = y_tr[b:b + batch]
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
    model.eval()
    with torch.no_grad():
        test_mse_reg = nn.functional.mse_loss(model(x_te_reg, dt=1.0), y_te_reg).item()
        test_mse_irreg = nn.functional.mse_loss(model(x_te_irreg, dt=dt_te_irreg), y_te_irreg).item()
    return {"test_mse_regular": test_mse_reg, "test_mse_irregular": test_mse_irreg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--date", default="2026-08-05")
    args = parser.parse_args()

    sigmas_train_mix = [0.3, 0.5, 1.0]
    sigma_tests = [0.3, 0.5, 1.0]
    n_feat = 4
    out = 1
    hidden = args.hidden

    torch.manual_seed(0)
    x, y, _ = make_ar2_with_timestamps(args.n_samples, seq_len=32, dt_log_sigma=0.5)
    n_tr = int(0.8 * x.shape[0])
    x_tr, y_tr = x[:n_tr], y[:n_tr]
    x_te_reg, y_te_reg = x[n_tr:], y[n_tr:]
    test_dts = {}
    for sigma in sigma_tests:
        torch.manual_seed(1)
        dt_test = torch.exp(torch.randn(*x_te_reg.shape[:2]) * sigma - 0.5 * sigma**2)
        dt_test[:, 0] = 1.0
        test_dts[sigma] = dt_test

    print(f"Task: AR(2) + 3-regime, mixed-dt training ({sigmas_train_mix}), test on {sigma_tests}")
    print(f"Models: α MLP capacity variants\n")

    # 5 alpha capacity variants + CfC baseline
    variants = [
        ("cfc-baseline",                       1, 0,  2137),
        ("mfc-hybrid_gate (depth=1, w=branch_dim, N11/N15 baseline)", 1, 0, 3577),
        ("mfc-hybrid_gate (depth=2, w=2*branch_dim)", 2, 0, 0),  # params computed later
        ("mfc-hybrid_gate (depth=3, w=2*branch_dim)", 3, 0, 0),
        ("mfc-hybrid_gate (depth=3, w=4*branch_dim)", 3, 0, 0),
    ]
    results = {}
    for name, depth, width, expected_params in variants:
        results[name] = {}
        for sigma in sigma_tests:
            msess_reg, msess_irreg = [], []
            for r in range(args.repeats):
                torch.manual_seed(42 + r)
                if "cfc-baseline" in name:
                    model = _SeqWrap(CfCCell(n_feat, hidden), out)
                else:
                    model = _SeqWrap(
                        MemoryFusionCfCCell(
                            n_feat, hidden, retention_kind="hybrid_gate",
                            alpha_mlp_depth=depth, alpha_mlp_width=width,
                        ), out,
                    )
                res = train_eval_mixed(model, x_tr, y_tr, sigmas_train_mix,
                                      x_te_reg, y_te_reg,
                                      x_te_reg, y_te_reg, test_dts[sigma],
                                      epochs=args.epochs)
                msess_reg.append(res["test_mse_regular"])
                msess_irreg.append(res["test_mse_irregular"])
            n_params = sum(p.numel() for p in model.parameters())
            results[name][sigma] = {
                "test_mse_regular_mean": statistics.mean(msess_reg),
                "test_mse_irregular_mean": statistics.mean(msess_irreg),
                "degradation_ratio": statistics.mean(msess_irreg) / max(statistics.mean(msess_reg), 1e-9),
                "params": n_params,
            }
        row = results[name]
        print(f"  {name[:50]:50s} (params={row[sigma_tests[0]]['params']:5d}): " + " | ".join(
            f"σ={σ}: {row[σ]['degradation_ratio']:.2f}x" for σ in sigma_tests
        ))

    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.date}_alpha_capacity.json"
    md_path = out_dir / f"{args.date}_alpha_capacity.md"
    json_path.write_text(json.dumps({
        "date": args.date,
        "task": "alpha_capacity_hypothesis_n22",
        "config": vars(args),
        "results": results,
    }, indent=2))

    md = [f"""---
title: α MLP capacity hypothesis (N22) — does deeper/wider α break interpolation ceiling? — {args.date}
date: {args.date}
tags: [LNN, hybrid_gate, alpha-capacity, MLP-depth, OOD, N22]
---

# α MLP capacity hypothesis (N22) — {args.date}

## Setup
- Task: AR(2) + 3-regime, mixed-dt training (sigma in {{0.3, 0.5, 1.0}})
- Test on 3 sigma values (N15 setup for direct comparison)
- 5 α capacity variants + CfC baseline

## Results (degradation ratio)

| model | params | σ=0.3 | σ=0.5 | σ=1.0 (OOD) |
|---|---:|---:|---:|---:|
"""]
    for name, _, _, _ in variants:
        row = results[name]
        r0 = row[0.3]
        r1 = row[0.5]
        r2 = row[1.0]
        md.append(f"| {name} | {r0['params']} | "
                  f"**{r0['degradation_ratio']:.2f}x** | "
                  f"**{r1['degradation_ratio']:.2f}x** | "
                  f"**{r2['degradation_ratio']:.2f}x** |\n")
    md_path.write_text("".join(md))
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
