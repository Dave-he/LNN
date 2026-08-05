"""N16: Verify CfC's structural-generic dt-robustness on harder tasks.

N12 found CfC achieves 1.00x degradation across all sigma_test on simple
3-regime AR(2). This benchmark tests whether the claim generalises to:
  - More regimes (5, 8 vs original 3)
  - Longer sequences (96 vs original 32/48)
  - Regimes with overlapping AR coefficients
  - Mixed non-stationarity (regime drift within sequence)

If CfC maintains 1.00x degradation across ALL these task variants,
N12's "structural-generic" claim is fully validated.
If CfC breaks on harder tasks, then CfC's robustness has limits.
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
# Data: multiple non-stationary task variants
# ---------------------------------------------------------------------------


def make_task_ar2_regime(n_samples, seq_len, n_feat, n_regimes, regime_overlap=False,
                         intra_drift=False, seed=0, dt_log_sigma=0.0):
    """Build a synthetic non-stationary AR(2) task.

    Args:
        n_regimes: number of distinct AR coefficient regimes (3 in original task)
        regime_overlap: if True, regimes share some coefficients (harder)
        intra_drift: if True, regime changes mid-sequence (harder)
    """
    torch.manual_seed(seed)
    x = torch.zeros(n_samples, seq_len, n_feat)
    y = torch.zeros(n_samples, seq_len, 1)
    dt = torch.ones(n_samples, seq_len)
    if dt_log_sigma > 0:
        dt = torch.exp(torch.randn(n_samples, seq_len) * dt_log_sigma - 0.5 * dt_log_sigma**2)
        dt[:, 0] = 1.0
    # Define regimes (expand from original 3)
    base_regimes = [(0.6, 0.2), (-0.3, 0.5), (0.4, -0.4),
                    (0.2, 0.6), (0.5, -0.3), (-0.4, 0.4),
                    (0.7, -0.1), (-0.2, 0.5)]
    regimes = base_regimes[:n_regimes]
    if regime_overlap:
        # Make regimes similar — use only 2 unique pairs and replicate
        regimes = [(0.6, 0.2), (-0.3, 0.5)] * (n_regimes // 2 + 1)
        regimes = regimes[:n_regimes]
    for s in range(n_samples):
        if intra_drift:
            # Regime can change mid-sequence (harder task)
            n_switches = max(1, n_regimes // 2)
            switch_points = sorted(torch.randperm(seq_len - 1)[:n_switches].tolist())
            regime_seq = [torch.randint(0, n_regimes, (1,)).item() for _ in range(n_switches + 1)]
            regimes_for_seq = iter(regime_seq)
        else:
            regime = torch.randint(0, n_regimes, (1,)).item()
        ar1, ar2 = regimes[regime if not intra_drift else next(regimes_for_seq)]
        noise = torch.randn(seq_len, n_feat) * 0.1
        current_regime = regime if not intra_drift else 0
        for t in range(1, seq_len):
            if intra_drift and t in switch_points:
                current_regime = next(regimes_for_seq)
                ar1, ar2 = regimes[current_regime]
            x[s, t] = ar1 * x[s, t - 1] + ar2 * x[s, max(t - 2, 0)] + noise[t]
        y[s, :-1, 0] = x[s, 1:, :].sum(-1)
    return x, y, dt


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


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
    parser.add_argument("--n-samples", type=int, default=192)
    parser.add_argument("--date", default="2026-08-05")
    args = parser.parse_args()

    # Define 4 task variants of increasing difficulty
    task_variants = [
        ("3-regime (N12 baseline)", {"n_regimes": 3, "regime_overlap": False, "intra_drift": False, "seq_len": 32}),
        ("5-regime",                {"n_regimes": 5, "regime_overlap": False, "intra_drift": False, "seq_len": 32}),
        ("8-regime",                {"n_regimes": 8, "regime_overlap": False, "intra_drift": False, "seq_len": 32}),
        ("3-regime + intra-drift",  {"n_regimes": 3, "regime_overlap": False, "intra_drift": True,  "seq_len": 32}),
        ("3-regime + overlap",      {"n_regimes": 4, "regime_overlap": True,  "intra_drift": False, "seq_len": 32}),
        ("3-regime long (sl=96)",   {"n_regimes": 3, "regime_overlap": False, "intra_drift": False, "seq_len": 96}),
    ]

    n_feat = 4
    out = 1
    hidden = 24
    sigma_train = 0.5
    sigma_test = 0.5  # in-dist test to check task-difficulty effect separately from dt-shift

    results = {}
    for task_name, task_kw in task_variants:
        torch.manual_seed(0)
        x_irreg, y_irreg, dt_irreg = make_task_ar2_regime(
            n_samples=args.n_samples, n_feat=n_feat, dt_log_sigma=sigma_train,
            **task_kw, seed=0,
        )
        n_tr = int(0.8 * x_irreg.shape[0])
        x_tr, y_tr, dt_tr = x_irreg[:n_tr], y_irreg[:n_tr], dt_irreg[:n_tr]
        x_te_reg, y_te_reg = x_irreg[n_tr:], y_irreg[n_tr:]
        torch.manual_seed(1)
        dt_te_irreg = torch.exp(torch.randn(*x_te_reg.shape[:2]) * sigma_test - 0.5 * sigma_test**2)
        dt_te_irreg[:, 0] = 1.0

        results[task_name] = {}
        for model_name in ("cfc-baseline", "mfc-tfp", "mfc-hybrid_gate"):
            msess_reg, msess_irreg = [], []
            for r in range(args.repeats):
                torch.manual_seed(42 + r)
                if model_name == "cfc-baseline":
                    model = _SeqWrap(CfCCell(n_feat, hidden), out)
                elif model_name == "mfc-tfp":
                    model = _SeqWrap(MemoryFusionCfCCell(n_feat, hidden, retention_kind="tfp"), out)
                else:
                    model = _SeqWrap(MemoryFusionCfCCell(n_feat, hidden, retention_kind="hybrid_gate"), out)
                res = train_eval(model, x_tr, y_tr, dt_tr,
                                 x_te_reg, y_te_reg,
                                 x_te_reg, y_te_reg, dt_te_irreg,
                                 epochs=args.epochs)
                msess_reg.append(res["test_mse_regular"])
                msess_irreg.append(res["test_mse_irregular"])
            results[task_name][model_name] = {
                "test_mse_regular_mean": statistics.mean(msess_reg),
                "test_mse_regular_std": statistics.stdev(msess_reg) if len(msess_reg) > 1 else 0.0,
                "test_mse_irregular_mean": statistics.mean(msess_irreg),
                "test_mse_irregular_std": statistics.stdev(msess_irreg) if len(msess_irreg) > 1 else 0.0,
                "degradation_ratio": statistics.mean(msess_irreg) / max(statistics.mean(msess_reg), 1e-9),
            }
        cfc_row = results[task_name]["cfc-baseline"]
        tfp_row = results[task_name]["mfc-tfp"]
        hg_row = results[task_name]["mfc-hybrid_gate"]
        print(f"  {task_name}: "
              f"cfc={cfc_row['degradation_ratio']:.2f}x "
              f"tfp={tfp_row['degradation_ratio']:.2f}x "
              f"hybrid_gate={hg_row['degradation_ratio']:.2f}x")

    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.date}_cfc_transferability.json"
    md_path = out_dir / f"{args.date}_cfc_transferability.md"
    json_path.write_text(json.dumps({
        "date": args.date,
        "task": "cfc_transferability_on_harder_tasks",
        "config": vars(args),
        "results": results,
    }, indent=2))

    md = [f"""---
title: CfC transferability on harder tasks (N16) — {args.date}
date: {args.date}
tags: [LNN, CfC, TFP, hybrid_gate, transferability, multi-regime, structural-generic, N16]
---

# CfC transferability on harder tasks (N16) — {args.date}

## Setup
- 6 task variants of increasing difficulty (all AR(2) family)
- Same dt distribution train (sigma=0.5) and test (sigma=0.5) — in-dist for dt
- 3 models x 6 tasks x 2 repeats x 4 epochs

## Tasks
"""]
    for name, kw in task_variants:
        md.append(f"- **{name}**: {kw}\n")
    md.append("\n## Results (degradation ratio)\n\n")
    md.append("| task | cfc-baseline | mfc-tfp | mfc-hybrid_gate |\n")
    md.append("|---|---:|---:|---:|\n")
    for name, _ in task_variants:
        cfc_row = results[name]["cfc-baseline"]
        tfp_row = results[name]["mfc-tfp"]
        hg_row = results[name]["mfc-hybrid_gate"]
        md.append(f"| {name} | "
                  f"**{cfc_row['degradation_ratio']:.2f}x** | "
                  f"{tfp_row['degradation_ratio']:.2f}x | "
                  f"{hg_row['degradation_ratio']:.2f}x |\n")
    md.append("""
## Verdict (TBD)

A "structural-generic" mechanism (N12 finding) should maintain
≈ 1.00x degradation across ALL task variants. If cfc-baseline
breaks on harder tasks, the finding is limited to 3-regime AR(2).
""")
    md_path.write_text("".join(md))
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
