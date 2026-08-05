"""N9 verification: Train MFC-Hybrid under *irregular* dt to see if alpha learns
conditional gating between CfC (dt-robust) and TFP (explicit-dt) paths.

Setup:
    - Train dt ~ LogNormal(0, 0.5)  (jittered, same as last benchmark's test)
    - Test A: dt = 1.0 (regular)
    - Test B: dt ~ LogNormal(0, 0.5) (irregular, same distribution as train)

Hypothesis:
    If hybrid's α truly learns conditional gating, we expect:
        - On irregular dt (matching train): α → 1 (CfC path dominant)
        - On regular dt (different from train): α → 0 (TFP path dominant)
    But because training is single-distribution, alpha may converge to a
    single value rather than per-input conditional gating. Either result is
    informative.

Comparisons:
    cfc-baseline  : upper bound on dt-robustness
    mfc-cfc       : identical numerical behaviour, sanity gate
    mfc-tfp       : TFP-only, expected to underperform on irregular dt
    mfc-hybrid    : the subject of N9

Output:
    analysis/jetson/<date>_hybrid_irregular_train.{md,json}
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


# ---------------------------------------------------------------------------
# Sequence wrapper with α tracking
# ---------------------------------------------------------------------------


class _HybridSeqWrap(nn.Module):
    """Wraps MemoryFusionCfCCell(retention_kind='hybrid') with α-trace."""

    def __init__(self, cell: MemoryFusionCfCCell, out_dim: int):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(cell.hidden_size, out_dim)
        self._alpha_history: list[float] = []

    def forward(self, x: torch.Tensor, dt=1.0):
        b, t, _ = x.shape
        h = torch.zeros(b, self.cell.hidden_size, device=x.device, dtype=x.dtype)
        outs = []
        for i in range(t):
            dt_i = dt[:, i] if isinstance(dt, torch.Tensor) else dt
            if isinstance(dt_i, torch.Tensor) and dt_i.dim() == 1:
                dt_i = dt_i.unsqueeze(-1)
            h = self.cell(x[:, i, :], h, dt=dt_i)
            outs.append(self.head(h))
        return torch.stack(outs, dim=1)

    def snapshot_alpha(self):
        with torch.no_grad():
            alphas = torch.cat([torch.sigmoid(p).flatten() for p in self.cell.alpha])
            self._alpha_history.append({
                "mean": alphas.mean().item(),
                "min": alphas.min().item(),
                "max": alphas.max().item(),
                "std": alphas.std().item(),
            })


class _SeqWrap(nn.Module):
    def __init__(self, cell: nn.Module, out_dim: int):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(cell.hidden_size, out_dim)

    def forward(self, x: torch.Tensor, dt=1.0):
        b, t, _ = x.shape
        h = torch.zeros(b, self.cell.hidden_size, device=x.device, dtype=x.dtype)
        outs = []
        for i in range(t):
            dt_i = dt[:, i] if isinstance(dt, torch.Tensor) else dt
            if isinstance(dt_i, torch.Tensor) and dt_i.dim() == 1:
                dt_i = dt_i.unsqueeze(-1)
            h = self.cell(x[:, i, :], h, dt=dt_i)
            outs.append(self.head(h))
        return torch.stack(outs, dim=1)


def make_model(name: str, in_dim: int, hidden: int, out_dim: int):
    if name == "cfc":
        return _SeqWrap(CfCCell(in_dim, hidden), out_dim)
    if name == "mfc-cfc":
        return _SeqWrap(MemoryFusionCfCCell(in_dim, hidden, retention_kind="cfc"), out_dim)
    if name == "mfc-tfp":
        return _SeqWrap(MemoryFusionCfCCell(in_dim, hidden, retention_kind="tfp"), out_dim)
    if name == "mfc-hybrid":
        cell = MemoryFusionCfCCell(in_dim, hidden, retention_kind="hybrid")
        return _HybridSeqWrap(cell, out_dim)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------


def train_eval(model, x_tr, y_tr, dt_tr, x_te_reg, y_te_reg, x_te_irreg, y_te_irreg, dt_te_irreg,
               epochs=3, batch=8, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x_tr.shape[0]
    t0 = time.perf_counter()
    for ep in range(epochs):
        for b in range(0, n, batch):
            xb = x_tr[b:b + batch]
            yb = y_tr[b:b + batch]
            dtb = dt_tr[b:b + batch]
            opt.zero_grad()
            pred = model(xb, dt=dtb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            opt.step()
        # Snapshot α at end of each epoch (only for hybrid)
        if isinstance(model, _HybridSeqWrap):
            model.snapshot_alpha()
    train_s = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        test_mse_reg = nn.functional.mse_loss(model(x_te_reg, dt=1.0), y_te_reg).item()
        test_mse_irreg = nn.functional.mse_loss(model(x_te_irreg, dt=dt_te_irreg), y_te_irreg).item()
    return {
        "test_mse_regular": test_mse_reg,
        "test_mse_irregular": test_mse_irreg,
        "train_seconds": train_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=384)
    parser.add_argument("--dt-sigma", type=float, default=0.5)
    parser.add_argument("--date", default="2026-08-05")
    args = parser.parse_args()

    # Build irregular-dt training set
    torch.manual_seed(0)
    x_irreg, y_irreg, dt_irreg = make_ar2_with_timestamps(
        args.n_samples, args.seq_len, seed=0, dt_log_sigma=args.dt_sigma,
    )
    n_tr = int(0.8 * x_irreg.shape[0])
    x_tr, y_tr, dt_tr = x_irreg[:n_tr], y_irreg[:n_tr], dt_irreg[:n_tr]

    # Regular-dt test (deterministic) + irregular-dt test (sampled)
    x_te_reg, y_te_reg = x_irreg[n_tr:], y_irreg[n_tr:]
    torch.manual_seed(1)
    dt_te_irreg = torch.exp(torch.randn(*x_te_reg.shape[:2]) * args.dt_sigma - 0.5 * args.dt_sigma ** 2)
    dt_te_irreg[:, 0] = 1.0

    print(f"Train dt: LogNormal(0, {args.dt_sigma}) — IRREGULAR")
    print(f"Test regular dt: all 1.0")
    print(f"Test irregular dt: LogNormal(0, {args.dt_sigma}) — same distribution as train")
    print(f"  effective dt range: [{dt_tr.min().item():.3f}, {dt_tr.max().item():.3f}]")
    print(f"  mean train dt: {dt_tr.mean().item():.3f}")

    rows = []
    for name in ("cfc", "mfc-cfc", "mfc-tfp", "mfc-hybrid"):
        msess_reg, msess_irreg, trains = [], [], []
        alpha_history_runs = []
        for r in range(args.repeats):
            torch.manual_seed(42 + r)
            model = make_model(name, in_dim=4, hidden=args.hidden, out_dim=1)
            res = train_eval(model, x_tr, y_tr, dt_tr,
                             x_te_reg, y_te_reg,
                             x_te_reg, y_te_reg, dt_te_irreg,
                             epochs=args.epochs)
            msess_reg.append(res["test_mse_regular"])
            msess_irreg.append(res["test_mse_irregular"])
            trains.append(res["train_seconds"])
            if isinstance(model, _HybridSeqWrap):
                alpha_history_runs.append(model._alpha_history)
        row = {
            "model": name,
            "test_mse_regular_mean": statistics.mean(msess_reg),
            "test_mse_regular_std": statistics.stdev(msess_reg) if len(msess_reg) > 1 else 0.0,
            "test_mse_irregular_mean": statistics.mean(msess_irreg),
            "test_mse_irregular_std": statistics.stdev(msess_irreg) if len(msess_irreg) > 1 else 0.0,
            "train_seconds_mean": statistics.mean(trains),
        }
        row["degradation_ratio"] = (
            row["test_mse_irregular_mean"] / max(row["test_mse_regular_mean"], 1e-9)
        )
        # Average α trajectory across runs
        if alpha_history_runs:
            alpha_mean_per_epoch = []
            for ep in range(args.epochs):
                ep_means = [h[ep]["mean"] for h in alpha_history_runs if len(h) > ep]
                if ep_means:
                    alpha_mean_per_epoch.append(sum(ep_means) / len(ep_means))
            row["alpha_trajectory_mean"] = alpha_mean_per_epoch
        rows.append(row)
        print(f"  {name}: regular={row['test_mse_regular_mean']:.4f}, "
              f"irregular={row['test_mse_irregular_mean']:.4f}, "
              f"ratio={row['degradation_ratio']:.2f}x")
        if "alpha_trajectory_mean" in row:
            print(f"    α trajectory (epoch-end mean): {[round(a,3) for a in row['alpha_trajectory_mean']]}")

    # Write outputs
    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.date}_hybrid_irregular_train.json"
    md_path = out_dir / f"{args.date}_hybrid_irregular_train.md"
    json_path.write_text(json.dumps({
        "date": args.date,
        "task": "ar2_regime_irregular_dt_train_then_test_both",
        "config": vars(args),
        "rows": rows,
    }, indent=2))

    md = [f"""---
title: MFC-Hybrid retention — trained on irregular Δt to verify α conditional gating (N9) — {args.date}
date: {args.date}
tags: [LNN, CfC, TFP, hybrid, retention, irregular-dt, conditional-gating, alpha-learning, N9]
---

# MFC-Hybrid retention — trained on irregular Δt (N9 verification)

## Setup
- **Training dt**: LogNormal(0, {args.dt_sigma}) — IRREGULAR, range [{dt_tr.min().item():.3f}, {dt_tr.max().item():.3f}]
- **Test A (regular)**: dt=1.0 constant
- **Test B (irregular)**: LogNormal(0, {args.dt_sigma}) — same distribution as training

## Hypothesis
If hybrid's α learns **conditional gating** (input-dependent α switching), then:
- α should approach 1 (CfC path) under dt-jitter input that the model trained on
- α might approach 0 (TFP path) under regular dt input that the model didn't train on
Alternatively, α may converge to a single value (no per-input conditioning).

## Results ({args.repeats} repeats × {args.epochs} epochs, mean±std)

| model | test_mse_regular | test_mse_irregular | degradation ratio |
|---|---:|---:|---:|
"""]
    for r in rows:
        md.append(f"| {r['model']} | {r['test_mse_regular_mean']:.4f} ± {r['test_mse_regular_std']:.4f} | "
                  f"{r['test_mse_irregular_mean']:.4f} ± {r['test_mse_irregular_std']:.4f} | "
                  f"**{r['degradation_ratio']:.2f}×** |\n")

    # α trajectory
    for r in rows:
        if "alpha_trajectory_mean" in r:
            traj = [round(a, 3) for a in r["alpha_trajectory_mean"]]
            md.append(f"\n**{r['model']} α trajectory (epoch-end mean over {args.repeats} runs)**: {traj}\n")

    md.append("\n## Verdict\nTBD — see report.\n")
    md_path.write_text("\n".join(md))
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
