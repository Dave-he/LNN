"""N2: L-RFM (frozen LTC features + linear readout) benchmark.

Closes the 50% done N2 gap by implementing arXiv 2606.15571's
Liquid Random Feature Methods as a code module, then benchmarking
against trained CfC on AR(2) 3-regime.

Setup:
    L-RFM (frozen LTC, n_features=64) + linear readout
    vs. trained CfC (h=24, 8 / 16 / 24)

Tests whether *frozen* LTC random features can match a *trained* small
CfC on a simple AR(2) prediction task. If yes, this validates the L-RFM
hypothesis (random features + closed-form LTC = competitive frozen baseline).
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
from lnn.core.lrfm import LRFMSequenceRegressor


def make_ar2_regime(n_samples, seq_len, n_feat=4, seed=0):
    torch.manual_seed(seed)
    x = torch.zeros(n_samples, seq_len, n_feat)
    y = torch.zeros(n_samples, seq_len, 1)
    for s in range(n_samples):
        regime = torch.randint(0, 3, (1,)).item()
        ar1, ar2 = ((0.6, 0.2), (-0.3, 0.5), (0.4, -0.4))[regime]
        noise = torch.randn(seq_len, n_feat) * 0.1
        for t in range(1, seq_len):
            x[s, t] = ar1 * x[s, t - 1] + ar2 * x[s, max(t - 2, 0)] + noise[t]
        y[s, :-1, 0] = x[s, 1:, :].sum(-1)
    return x, y


class _CfCWrap(nn.Module):
    def __init__(self, cell, out_dim):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(cell.hidden_size, out_dim)

    def forward(self, x, dt=1.0):
        b, t, _ = x.shape
        h = torch.zeros(b, self.cell.hidden_size)
        outs = []
        for i in range(t):
            h = self.cell(x[:, i, :], h, dt=1.0)
            outs.append(self.head(h))
        return torch.stack(outs, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=192)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--date", default="2026-08-05")
    args = parser.parse_args()

    n_feat = 4
    out = 1

    torch.manual_seed(0)
    x, y = make_ar2_regime(args.n_samples, args.seq_len, seed=0)
    n_tr = int(0.8 * x.shape[0])
    x_tr, y_tr = x[:n_tr], y[:n_tr]
    x_te, y_te = x[n_tr:], y[n_tr:]

    models = {}
    models["L-RFM n_features=32"] = lambda: LRFMSequenceRegressor(
        input_size=n_feat, output_size=out, n_features=32
    )
    models["L-RFM n_features=64"] = lambda: LRFMSequenceRegressor(
        input_size=n_feat, output_size=out, n_features=64
    )
    models["L-RFM n_features=128"] = lambda: LRFMSequenceRegressor(
        input_size=n_feat, output_size=out, n_features=128
    )
    for h in (8, 16, 24):
        models[f"CfC h={h}"] = lambda h=h: _CfCWrap(CfCCell(n_feat, h), out)

    rows = []
    for name, factory in models.items():
        msess, params, trains = [], [], []
        for r in range(args.repeats):
            torch.manual_seed(42 + r)
            model = factory()
            n_params = sum(p.numel() for p in model.parameters())
            params.append(n_params)
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
            t0 = time.perf_counter()
            for _ in range(args.epochs):
                for b in range(0, n_tr, 8):
                    xb = x_tr[b:b + 8]
                    yb = y_tr[b:b + 8]
                    opt.zero_grad()
                    pred = model(xb, dt=1.0)
                    loss = nn.functional.mse_loss(pred, yb)
                    loss.backward()
                    opt.step()
            train_s = time.perf_counter() - t0
            model.eval()
            with torch.no_grad():
                mse = nn.functional.mse_loss(model(x_te, dt=1.0), y_te).item()
            msess.append(mse)
            trains.append(train_s)
        rows.append({
            "model": name,
            "params": int(statistics.mean(params)),
            "test_mse_mean": statistics.mean(msess),
            "test_mse_std": statistics.stdev(msess) if len(msess) > 1 else 0.0,
            "train_s_mean": statistics.mean(trains),
        })
        r = rows[-1]
        print(f"  {name:30s}: params={r['params']:5d}, "
              f"MSE={r['test_mse_mean']:.4f}±{r['test_mse_std']:.4f}, "
              f"train_s={r['train_s_mean']:.2f}")

    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.date}_lrfm.json"
    md_path = out_dir / f"{args.date}_lrfm.md"
    json_path.write_text(json.dumps({
        "date": args.date,
        "task": "lrfm_frozen_ltc_features_vs_trained_cfc",
        "config": vars(args),
        "rows": rows,
    }, indent=2))

    md = [f"""---
title: L-RFM (frozen LTC features) vs trained CfC — N2 closure
date: {args.date}
tags: [LNN, L-RFM, random-features, frozen-LTC, frozen-feature, N2, foundational]
arxiv_refs: [2606.15571, 2106.13898]
---

# L-RFM (frozen LTC features) vs trained CfC — N2 closure

## Setup
- Task: AR(2) 3-regime, next-step prediction
- Models:
  - **L-RFM n_features=32/64/128** — frozen LTC random features + linear readout
  - **CfC h=8/16/24** — trained CfC cell + linear head
- 192 samples, sl=24, 2 repeats × 4 epochs

## Results

| model | params | test MSE | train s |
|---|---:|---:|---:|
"""]
    for r in rows:
        md.append(f"| {r['model']} | {r['params']} | "
                  f"{r['test_mse_mean']:.4f} ± {r['test_mse_std']:.4f} | "
                  f"{r['train_s_mean']:.2f} |\n")
    md_path.write_text("".join(md))
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
