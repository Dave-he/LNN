#!/usr/bin/env python3
"""Real Henry Hub natural-gas gate evaluation (round 282).

Tests whether r281's synthetic finding — the ACCELERATION gate is the
best liquid-τ gate on nonstationary data — transfers to the REAL Henry
Hub natural-gas spot-price series, the literal motivating domain of the
gate line (natural-gas LNN arXiv:2604.24788, "limit responsiveness when
regimes shift rapidly").

Data: analysis/paper_replication/simulated_henry_hub.csv
  2645 daily obs (2015-2025). Spot Price 1.20-34.91. Genuinely
  nonstationary: rolling-30 return vol ranges 33× (0.009-0.284).

Task: one-step-ahead prediction of the STANDARDISED spot return, from a
sliding window of T past returns. Chronological split (no shuffle).
Normalisation stats computed on TRAIN ONLY (no look-ahead leakage).

We report overall test MSE AND a HIGH-VOL subset MSE (test windows whose
target day falls in the top-quartile rolling-30 volatility) — the
regime-shift stress subset where the gates should matter most.

Five modes: static / liquid / gated_vel / gated_accel / gated_blend.
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


# ---------------------------------------------------------------------------
# Data loading (chronological, train-only normalisation, no look-ahead)
# ---------------------------------------------------------------------------
def load_henry_hub(T=64, train_frac=0.7, vol_window=30):
    """Return train/test windows of standardised returns + a high-vol
    test mask. All normalisation stats are computed on the TRAIN split
    only to avoid look-ahead leakage.

    Returns dict with x_tr, y_tr, x_te, y_te (torch, (n, T, 1)/(n, T, 1))
    and hi_vol_mask (bool over test windows).
    """
    import pandas as pd
    df = pd.read_csv(CSV)
    price = df["Spot Price"].values.astype(np.float64)
    ret = np.diff(price) / (price[:-1] + 1e-8)  # daily returns, len N-1

    # chronological split index on the RETURN series
    n = len(ret)
    split = int(train_frac * n)

    # train-only standardisation
    mu, sd = ret[:split].mean(), ret[:split].std() + 1e-8
    z = (ret - mu) / sd  # standardised returns (full series)

    # rolling volatility (causal) for the high-vol subset labelling
    rv = np.zeros(n)
    for i in range(n):
        lo = max(0, i - vol_window)
        rv[i] = ret[lo:i].std() if i > lo + 1 else 0.0
    hi_thresh = np.quantile(rv[:split], 0.75)  # train-quartile threshold

    def windows(z_slice, rv_slice, base):
        xs, ys, hivol = [], [], []
        for i in range(len(z_slice) - T):
            xs.append(z_slice[i:i + T])
            ys.append(z_slice[i + 1:i + T + 1])
            hivol.append(rv_slice[i + T] > hi_thresh)  # target-day vol
        x = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1)
        return x, y, torch.tensor(np.array(hivol), dtype=torch.bool)

    x_tr, y_tr, _ = windows(z[:split], rv[:split], 0)
    x_te, y_te, hivol_te = windows(z[split:], rv[split:], split)
    return {"x_tr": x_tr, "y_tr": y_tr, "x_te": x_te, "y_te": y_te,
            "hi_vol_mask": hivol_te, "mu": float(mu), "sd": float(sd),
            "hi_thresh": float(hi_thresh)}


class ReadoutHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, h):
        return self.head(h)


class SeqModel(nn.Module):
    def __init__(self, cell, hidden_size, entropy_lambda):
        super().__init__()
        self.cell = cell
        self.head = ReadoutHead(hidden_size)
        self.entropy_lambda = float(entropy_lambda)

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
    if cfg["kind"] == "static":
        cell = STEWithEntropy(**_COMMON)
    elif cfg["kind"] == "liquid":
        cell = LiquidTauSTECfCCell(liquid_tau_strength=1.0, **_COMMON)
    elif cfg["kind"] == "gated_vel":
        cell = PredictabilityGatedLiquidTauCfCCell(
            liquid_tau_strength=1.0, pred_gate_beta=cfg["pred_gate_beta"],
            ema_gamma=cfg["ema_gamma"], **_COMMON)
    elif cfg["kind"] == "gated_accel":
        cell = AccelGatedLiquidTauCfCCell(
            liquid_tau_strength=1.0, pred_gate_beta=cfg["pred_gate_beta"],
            ema_gamma=cfg["ema_gamma"], diff_order=2, **_COMMON)
    else:  # gated_blend
        cell = BlendGatedLiquidTauCfCCell(
            liquid_tau_strength=1.0, pred_gate_beta=cfg["pred_gate_beta"],
            ema_gamma=cfg["ema_gamma"], gate_mode="blend", **_COMMON)
    return SeqModel(cell, _COMMON["hidden_size"], _COMMON["entropy_lambda"])


def train_eval(model, data, epochs, lr, batch_size, device):
    model.to(device)
    x_tr, y_tr = data["x_tr"].to(device), data["y_tr"].to(device)
    x_te, y_te = data["x_te"].to(device), data["y_te"].to(device)
    hivol = data["hi_vol_mask"].to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = x_tr.shape[0]
    model.train()
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        xs, ys = x_tr[perm], y_tr[perm]
        ep, nb = 0.0, 0
        for i in range(0, N, batch_size):
            xb, yb = xs[i:i + batch_size], ys[i:i + batch_size]
            if xb.shape[0] == 0:
                continue
            pred = model(xb)
            mse = (pred - yb).pow(2).mean()
            loss = mse + model.extra_loss()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            ep += float(mse.item()); nb += 1
        last = ep / max(nb, 1)
    model.eval()
    with torch.no_grad():
        pred = model(x_te)
        # per-window last-step error (the actual forecast target)
        err = (pred[:, -1, 0] - y_te[:, -1, 0]).pow(2)  # (n_test,)
        overall = float(err.mean().item())
        hi = float(err[hivol].mean().item()) if hivol.any() else float("nan")
        lo = float(err[~hivol].mean().item()) if (~hivol).any() else float("nan")
    return {"test_mse": overall, "hi_vol_mse": hi, "calm_mse": lo,
            "train_loss_last": last}


def collect_gate(model, x_sample):
    cell = model.cell
    with torch.no_grad():
        if isinstance(cell, PredictabilityGatedLiquidTauCfCCell):
            _, _, aux = cell(x_sample, return_aux=True)
            return aux["gate_mean"], aux["tau_temporal_std"]
        if isinstance(cell, LiquidTauSTECfCCell):
            _, _, aux = cell(x_sample, return_aux=True)
            return 1.0, aux["tau_temporal_std"]
    return float("nan"), 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", type=str,
                   default="analysis/henry_hub_gates_bench.json")
    p.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_henry_hub(T=args.T)
    print(f"[bench] device={device} | train={data['x_tr'].shape} "
          f"test={data['x_te'].shape} | hi_vol test windows="
          f"{int(data['hi_vol_mask'].sum())}/{len(data['hi_vol_mask'])}")

    results = {"config": {"epochs": args.epochs, "lr": args.lr,
                          "batch_size": args.batch_size, "T": args.T,
                          "seeds": args.seeds, "modes": args.modes,
                          "dataset": "henry_hub_spot_return",
                          "train_frac": 0.7,
                          "hi_thresh": data["hi_thresh"]}, "cells": []}

    for mode_name in args.modes:
        cfg = MODES[mode_name]
        for seed in args.seeds:
            torch.manual_seed(seed)
            model = make_model(cfg)
            t0 = time.time()
            out = train_eval(model, data, args.epochs, args.lr,
                             args.batch_size, device)
            el = time.time() - t0
            gm, tt = collect_gate(model, data["x_te"][:16].to(device))
            results["cells"].append({
                "mode": mode_name, "seed": seed,
                "test_mse": out["test_mse"], "hi_vol_mse": out["hi_vol_mse"],
                "calm_mse": out["calm_mse"],
                "train_loss_last": out["train_loss_last"],
                "gate_mean": gm, "tau_temporal_std": tt,
                "elapsed_sec": round(el, 2)})
            print(f"[bench] {mode_name:13s} seed={seed} "
                  f"overall={out['test_mse']:.5f} hi_vol={out['hi_vol_mse']:.5f} "
                  f"calm={out['calm_mse']:.5f} gate={gm:.3f} ({el:.1f}s)")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {outp}")

    agg = {}
    for c in results["cells"]:
        a = agg.setdefault(c["mode"], {"o": [], "h": [], "c": []})
        a["o"].append(c["test_mse"]); a["h"].append(c["hi_vol_mse"])
        a["c"].append(c["calm_mse"])

    def m(xs):
        xs = [v for v in xs if v == v]  # drop nan
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n[bench] Henry Hub mean MSE (overall | hi_vol | calm):")
    print(f"{'mode':13s} | {'overall':>9s} | {'hi_vol':>9s} | {'calm':>9s}")
    st = m(agg["static_tau"]["o"]) if "static_tau" in agg else float("nan")
    for mode in args.modes:
        if mode not in agg:
            continue
        a = agg[mode]
        o = m(a["o"])
        dpct = 100.0 * (o - st) / st if st == st and st > 0 else float("nan")
        print(f"{mode:13s} | {o:>9.5f} | {m(a['h']):>9.5f} | "
              f"{m(a['c']):>9.5f}   (overall {dpct:+.1f}% vs static)")


if __name__ == "__main__":
    main()
