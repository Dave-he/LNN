#!/usr/bin/env python3
"""Multi-series real gate transfer (round 283).

r282 found the VELOCITY gate wins on the real Henry Hub natural-gas
series (not acceleration, contra synthetic r281). This bench tests
whether that generalises across FIVE real financial series cached in the
same CSV (gas spot, WTI oil, Treasury 10Y, S&P Energy, coal), which span
a range of volatility profiles (vol_ratio 3-33×).

Question: is "velocity gate best" a general real-return-series property,
or specific to natural gas?

Each series: standardised one-step return prediction, T=64, chronological
split, train-only normalisation (no look-ahead). Reports per-series
overall + high-vol MSE for the five gate modes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lnn.core.ste_entropy_neuron_wise_cfc import STEWithEntropy  # noqa: E402
from lnn.core.liquid_tau_ste_cfc import LiquidTauSTECfCCell  # noqa: E402
from lnn.core.pred_gated_liquid_tau_cfc import (  # noqa: E402
    PredictabilityGatedLiquidTauCfCCell,
)
from lnn.core.accel_gated_liquid_tau_cfc import (  # noqa: E402
    AccelGatedLiquidTauCfCCell,
)
from lnn.core.blend_gated_liquid_tau_cfc import (  # noqa: E402
    BlendGatedLiquidTauCfCCell,
)

CSV = ROOT / "analysis/paper_replication/simulated_henry_hub.csv"

# real series (column -> short label); chosen for clean, non-degenerate
# return behaviour and a spread of volatility profiles.
SERIES = {
    "gas": "Spot Price",       # vol_ratio 33× (highly nonstationary)
    "oil": "WTI Price",        # vol_ratio 3.2×
    "rates": "Treasury_10Y",   # vol_ratio 3.6×
    "equity": "SP_Energy",     # vol_ratio 3.0×
    "coal": "Coal_Index",      # vol_ratio 4.1×
}


def load_series(column, T=64, train_frac=0.7, vol_window=30):
    """Load one CSV column as standardised one-step returns; train-only
    normalisation; high-vol test mask. No look-ahead."""
    import pandas as pd
    df = pd.read_csv(CSV)
    price = df[column].values.astype(np.float64)
    price = price[~np.isnan(price)]
    ret = np.diff(price) / (np.abs(price[:-1]) + 1e-8)
    n = len(ret)
    split = int(train_frac * n)
    mu, sd = ret[:split].mean(), ret[:split].std() + 1e-8
    z = (ret - mu) / sd
    rv = np.zeros(n)
    for i in range(n):
        lo = max(0, i - vol_window)
        rv[i] = ret[lo:i].std() if i > lo + 1 else 0.0
    hi_thresh = np.quantile(rv[:split], 0.75)

    def windows(zs, rvs):
        xs, ys, hv = [], [], []
        for i in range(len(zs) - T):
            xs.append(zs[i:i + T])
            ys.append(zs[i + 1:i + T + 1])
            hv.append(rvs[i + T] > hi_thresh)
        x = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1)
        return x, y, torch.tensor(np.array(hv), dtype=torch.bool)

    x_tr, y_tr, _ = windows(z[:split], rv[:split])
    x_te, y_te, hv_te = windows(z[split:], rv[split:])
    return {"x_tr": x_tr, "y_tr": y_tr, "x_te": x_te, "y_te": y_te,
            "hi_vol_mask": hv_te}


class ReadoutHead(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        return self.head(x)


class SeqModel(nn.Module):
    def __init__(self, cell, h, lam):
        super().__init__()
        self.cell = cell
        self.head = ReadoutHead(h)
        self.entropy_lambda = float(lam)

    def forward(self, x):
        out, _ = self.cell(x)
        return self.head(out)

    def extra_loss(self):
        if self.entropy_lambda <= 0:
            return torch.tensor(0.0)
        return self.cell.extra_loss()


MODES = {
    "static_tau": dict(kind="static"),
    "liquid_tau": dict(kind="liquid"),
    "gated_vel": dict(kind="gated_vel", pred_gate_beta=4.0, ema_gamma=0.5),
    "gated_accel": dict(kind="gated_accel", pred_gate_beta=4.0, ema_gamma=0.5),
    "gated_blend": dict(kind="gated_blend", pred_gate_beta=4.0, ema_gamma=0.5),
}
_COMMON = dict(input_size=1, hidden_size=192, density=0.3,
               ste_temperature=1.0, entropy_lambda=0.1)


def make_model(cfg):
    k = cfg["kind"]
    if k == "static":
        cell = STEWithEntropy(**_COMMON)
    elif k == "liquid":
        cell = LiquidTauSTECfCCell(liquid_tau_strength=1.0, **_COMMON)
    elif k == "gated_vel":
        cell = PredictabilityGatedLiquidTauCfCCell(
            liquid_tau_strength=1.0, pred_gate_beta=cfg["pred_gate_beta"],
            ema_gamma=cfg["ema_gamma"], **_COMMON)
    elif k == "gated_accel":
        cell = AccelGatedLiquidTauCfCCell(
            liquid_tau_strength=1.0, pred_gate_beta=cfg["pred_gate_beta"],
            ema_gamma=cfg["ema_gamma"], diff_order=2, **_COMMON)
    else:
        cell = BlendGatedLiquidTauCfCCell(
            liquid_tau_strength=1.0, pred_gate_beta=cfg["pred_gate_beta"],
            ema_gamma=cfg["ema_gamma"], gate_mode="blend", **_COMMON)
    return SeqModel(cell, _COMMON["hidden_size"], _COMMON["entropy_lambda"])


def train_eval(model, data, epochs, lr, bs, device):
    model.to(device)
    x_tr, y_tr = data["x_tr"].to(device), data["y_tr"].to(device)
    x_te, y_te = data["x_te"].to(device), data["y_te"].to(device)
    hv = data["hi_vol_mask"].to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = x_tr.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        xs, ys = x_tr[perm], y_tr[perm]
        for i in range(0, N, bs):
            xb, yb = xs[i:i + bs], ys[i:i + bs]
            if xb.shape[0] == 0:
                continue
            loss = (model(xb) - yb).pow(2).mean() + model.extra_loss()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(x_te)
        err = (pred[:, -1, 0] - y_te[:, -1, 0]).pow(2)
        overall = float(err.mean().item())
        hi = float(err[hv].mean().item()) if hv.any() else float("nan")
        calm = float(err[~hv].mean().item()) if (~hv).any() else float("nan")
    return {"test_mse": overall, "hi_vol_mse": hi, "calm_mse": calm}


def gate_of(model, x):
    cell = model.cell
    with torch.no_grad():
        if isinstance(cell, PredictabilityGatedLiquidTauCfCCell):
            _, _, aux = cell(x, return_aux=True)
            return aux["gate_mean"]
        if isinstance(cell, LiquidTauSTECfCCell):
            return 1.0
    return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--series", nargs="+", default=["gas", "oil", "equity"])
    p.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    p.add_argument("--out", type=str,
                   default="analysis/multiseries_gates_bench.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bench] device={device} series={args.series} modes={args.modes}")

    results = {"config": {"epochs": args.epochs, "lr": args.lr,
                          "T": args.T, "seeds": args.seeds,
                          "series": args.series, "modes": args.modes,
                          "series_columns": {s: SERIES[s] for s in args.series}},
               "cells": []}

    for sname in args.series:
        data = load_series(SERIES[sname], T=args.T)
        print(f"[bench] --- {sname} ({SERIES[sname]}): train={tuple(data['x_tr'].shape)} "
              f"test={tuple(data['x_te'].shape)} hi_vol="
              f"{int(data['hi_vol_mask'].sum())}/{len(data['hi_vol_mask'])} ---")
        for mode in args.modes:
            for seed in args.seeds:
                torch.manual_seed(seed)
                model = make_model(MODES[mode])
                t0 = time.time()
                out = train_eval(model, data, args.epochs, args.lr,
                                 args.batch_size, device)
                el = time.time() - t0
                gm = gate_of(model, data["x_te"][:16].to(device))
                results["cells"].append({
                    "series": sname, "mode": mode, "seed": seed,
                    "test_mse": out["test_mse"], "hi_vol_mse": out["hi_vol_mse"],
                    "calm_mse": out["calm_mse"], "gate_mean": gm,
                    "elapsed_sec": round(el, 2)})
                print(f"[bench] {sname:7s} {mode:13s} seed={seed} "
                      f"overall={out['test_mse']:.4f} hi_vol={out['hi_vol_mse']:.4f} "
                      f"gate={gm:.3f} ({el:.1f}s)")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {outp}")

    # per-series overall matrix + best gate
    def mean(xs):
        xs = [v for v in xs if v == v]
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n[bench] Overall test MSE by series × mode (best gate marked *):")
    header = f"{'series':8s} | " + " | ".join(f"{m:>11s}" for m in args.modes)
    print(header)
    for sname in args.series:
        row = {}
        for m in args.modes:
            row[m] = mean([c["test_mse"] for c in results["cells"]
                           if c["series"] == sname and c["mode"] == m])
        gated = {m: row[m] for m in args.modes if m.startswith("gated")}
        best = min(gated, key=gated.get) if gated else None
        cells = " | ".join(
            (f"{row[m]:>10.4f}" + ("*" if m == best else " ")) for m in args.modes)
        print(f"{sname:8s} | {cells}")

    print("\n[bench] Δ%% vs static per series (best gated mode):")
    for sname in args.series:
        st = mean([c["test_mse"] for c in results["cells"]
                   if c["series"] == sname and c["mode"] == "static_tau"])
        line = f"  {sname:8s} static={st:.4f}"
        for m in args.modes:
            if m == "static_tau":
                continue
            v = mean([c["test_mse"] for c in results["cells"]
                      if c["series"] == sname and c["mode"] == m])
            pct = 100 * (v - st) / st if st > 0 else float("nan")
            line += f"  {m.replace('gated_', 'g_')}={pct:+.1f}%"
        print(line)


if __name__ == "__main__":
    main()
