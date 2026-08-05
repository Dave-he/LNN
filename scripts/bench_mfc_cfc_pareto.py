"""Pareto sweep for MemoryFusionCfCCell across hidden sizes × sequence lengths.

Validates the MFC-TFP advantage seen on the single-config benchmark by
running a hidden × seq_len grid and checking whether the (param, MSE)
Pareto frontier consistently favors MFC-TFP over CfC.

Usage:
    python3 scripts/bench_mfc_cfc_pareto.py [--repeats 3] [--epochs 3]
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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def make_ar2_regime(n_samples: int, seq_len: int, n_feat: int = 4, seed: int = 0):
    """Synthetic non-stationary AR(2) with 3 regime changes per sequence."""
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


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


class _SeqWrap(nn.Module):
    def __init__(self, cell: nn.Module, out_dim: int):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(cell.hidden_size, out_dim)

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        b, t, _ = x.shape
        h = torch.zeros(b, self.cell.hidden_size, device=x.device, dtype=x.dtype)
        outs = []
        for i in range(t):
            h = self.cell(x[:, i, :], h, dt=dt)
            outs.append(self.head(h))
        return torch.stack(outs, dim=1)


def make_model(name: str, in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    if name == "cfc":
        return _SeqWrap(CfCCell(in_dim, hidden), out_dim)
    if name == "mfc-cfc":
        return _SeqWrap(MemoryFusionCfCCell(in_dim, hidden, retention_kind="cfc"), out_dim)
    if name == "mfc-tfp":
        return _SeqWrap(MemoryFusionCfCCell(in_dim, hidden, retention_kind="tfp"), out_dim)
    if name == "mfc-nsfd":
        return _SeqWrap(MemoryFusionCfCCell(in_dim, hidden, retention_kind="nsfd"), out_dim)
    if name == "gru":
        class _G(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(in_dim, hidden, batch_first=True)
                self.head = nn.Linear(hidden, out_dim)

            def forward(self, x, dt=1.0):
                o, _ = self.gru(x)
                return self.head(o)
        return _G()
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Train/eval
# ---------------------------------------------------------------------------


def train_eval(model: nn.Module, x_tr, y_tr, x_te, y_te,
               epochs: int, batch: int, lr: float):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x_tr.shape[0]
    t0 = time.perf_counter()
    for _ in range(epochs):
        for b in range(0, n, batch):
            xb = x_tr[b:b + batch]
            yb = y_tr[b:b + batch]
            opt.zero_grad()
            pred = model(xb, dt=1.0)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            opt.step()
    train_s = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        n_inf = 0
        for b in range(0, x_te.shape[0], batch):
            _ = model(x_te[b:b + batch], dt=1.0)
            n_inf += x_te[b:b + batch].shape[0] * x_te.shape[1]
        inf_s = time.perf_counter() - t0
    with torch.no_grad():
        test_mse = nn.functional.mse_loss(model(x_te, dt=1.0), y_te).item()
    return {
        "test_mse": test_mse,
        "inference_steps_per_sec": n_inf / inf_s,
        "train_seconds": train_s,
    }


# ---------------------------------------------------------------------------
# Pareto driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-samples", type=int, default=384)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[16, 24, 32])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[32, 48, 96])
    parser.add_argument("--date", default="2026-08-05")
    args = parser.parse_args()

    models = ["cfc", "mfc-cfc", "mfc-tfp", "mfc-nsfd", "gru"]
    grid = []
    for h in args.hidden_sizes:
        for sl in args.seq_lens:
            grid.append((h, sl))
    print(f"Pareto grid: {grid}", flush=True)
    print(f"Models: {models}", flush=True)

    rows = []
    for h, sl in grid:
        x, y = make_ar2_regime(n_samples=args.n_samples, seq_len=sl, n_feat=4, seed=0)
        n_tr = int(0.8 * x.shape[0])
        x_tr, y_tr = x[:n_tr], y[:n_tr]
        x_te, y_te = x[n_tr:], y[n_tr:]
        for name in models:
            torch.manual_seed(42)
            model = make_model(name, in_dim=4, hidden=h, out_dim=1)
            n_params = sum(p.numel() for p in model.parameters())
            msess, infs, trains = [], [], []
            for _ in range(args.repeats):
                r = train_eval(model, x_tr, y_tr, x_te, y_te,
                               epochs=args.epochs, batch=args.batch, lr=args.lr)
                msess.append(r["test_mse"])
                infs.append(r["inference_steps_per_sec"])
                trains.append(r["train_seconds"])
            row = {
                "model": name, "hidden": h, "seq_len": sl,
                "params": n_params,
                "test_mse_mean": statistics.mean(msess),
                "test_mse_std": statistics.stdev(msess) if len(msess) > 1 else 0.0,
                "inference_steps_per_sec_mean": statistics.mean(infs),
                "train_seconds_mean": statistics.mean(trains),
            }
            rows.append(row)
            print(f"  {name} h={h} sl={sl}  mse={row['test_mse_mean']:.4f}±{row['test_mse_std']:.4f}  params={n_params}", flush=True)

    # Write outputs
    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.date}_mfc_cfc_pareto.json"
    md_path = out_dir / f"{args.date}_mfc_cfc_pareto.md"
    json_path.write_text(json.dumps({
        "date": args.date,
        "task": "non_stationary_ar2_regime_change",
        "config": vars(args),
        "rows": rows,
    }, indent=2))

    # Pareto analysis: for each (h, sl), which model is best?
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault((r["hidden"], r["seq_len"]), []).append(r)
    pareto_lines = ["## Pareto-frontier analysis\n",
                    "For each (hidden, seq_len) cell, the model with the lowest **test_mse_mean** is the Pareto winner (within-seed noise band). Tied models are listed together.\n"]
    for cfg, group in sorted(by_cfg.items()):
        min_mse = min(g["test_mse_mean"] for g in group)
        winners = [g["model"] for g in group
                   if abs(g["test_mse_mean"] - min_mse) <= 1e-4]
        pareto_lines.append(f"- h={cfg[0]}, seq_len={cfg[1]}: min MSE = {min_mse:.4f}  winners = {winners}\n")
    pareto_block = "".join(pareto_lines)

    # Markdown
    md = [f"""---
title: MemoryFusionCfC Pareto Sweep — hidden × seq_len grid ({args.date})
date: {args.date}
tags: [LNN, CfC, TFP, NSFD, cross-paper, retention, memory-fusion, pareto, benchmark]
---

# MemoryFusionCfC Pareto Sweep — {args.date}

## Grid
- hidden sizes: {args.hidden_sizes}
- seq lengths: {args.seq_lens}
- repeats: {args.repeats}, epochs: {args.epochs}, batch: {args.batch}, lr: {args.lr}
- n_samples: {args.n_samples}
- task: synthetic non-stationary AR(2) + 3-regime

## Results table

| model | hidden | seq_len | params | test MSE (mean ± std) | inf steps/s | train s |
|---|---|---|---:|---:|---:|---:|
"""]
    for r in rows:
        md.append(f"| {r['model']} | {r['hidden']} | {r['seq_len']} | {r['params']} | "
                  f"{r['test_mse_mean']:.4f} ± {r['test_mse_std']:.4f} | "
                  f"{r['inference_steps_per_sec_mean']:.1f} | "
                  f"{r['train_seconds_mean']:.2f} |\n")
    md.append("\n")
    md.append(pareto_block)
    md.append("\n## Verdict\n\nTBD — see report.\n")
    md_path.write_text("".join(md))
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
