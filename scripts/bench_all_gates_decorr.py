#!/usr/bin/env python3
"""Henry Hub validation for r295: does decorrelation default help all 3 gate variants?

Tests whether the r294 in-cell decorrelation default at λ=1e-5 helps
not just blend_gated (already validated r294) but also:
  - pred_gated (r278): velocity gate
  - accel_gated (r279): acceleration gate

If the SP generalizes, r295 adds 2 SP (pred_gated + accel_gated).
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


def load_henry_hub(T=64, train_frac=0.7, vol_window=30):
    """Chronological split, train-only normalisation (r282 loader)."""
    import pandas as pd
    df = pd.read_csv(CSV)
    price = df["Spot Price"].values.astype(np.float64)
    ret = np.diff(price) / (price[:-1] + 1e-8)
    n = len(ret)
    split = int(train_frac * n)
    mu, sd = ret[:split].mean(), ret[:split].std() + 1e-8
    z = (ret - mu) / sd
    rv = np.zeros(n)
    for i in range(n):
        lo = max(0, i - vol_window)
        rv[i] = ret[lo:i].std() if i > lo + 1 else 0.0
    hi_thresh = np.quantile(rv[:split], 0.75)

    def windows(z_slice, rv_slice):
        xs, ys, hivol = [], [], []
        for i in range(len(z_slice) - T):
            xs.append(z_slice[i:i + T])
            ys.append(z_slice[i + 1:i + T + 1])
            hivol.append(rv_slice[i + T] > hi_thresh)
        x = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1)
        return x, y, torch.tensor(np.array(hivol), dtype=torch.bool)

    x_tr, y_tr, _ = windows(z[:split], rv[:split])
    x_te, y_te, hivol_te = windows(z[split:], rv[split:])
    return {"x_tr": x_tr, "y_tr": y_tr, "x_te": x_te, "y_te": y_te,
            "hi_vol_mask": hivol_te}


class SeqModel(nn.Module):
    def __init__(self, cell, hidden_size, entropy_lambda):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(hidden_size, 1)
        self.entropy_lambda = float(entropy_lambda)

    def forward(self, x):
        out, _ = self.cell(x)
        return self.head(out)

    def extra_loss(self):
        return self.cell.extra_loss()


# (mode_name, cell_class, kind)
MODES = {
    "static_tau": ("static", STEWithEntropy, None),
    "pred_gated_off": ("pred", PredictabilityGatedLiquidTauCfCCell, "off"),
    "pred_gated_default": ("pred", PredictabilityGatedLiquidTauCfCCell, "default"),
    "accel_gated_off": ("accel", AccelGatedLiquidTauCfCCell, "off"),
    "accel_gated_default": ("accel", AccelGatedLiquidTauCfCCell, "default"),
    "blend_gated_off": ("blend", BlendGatedLiquidTauCfCCell, "off"),
    "blend_gated_default": ("blend", BlendGatedLiquidTauCfCCell, "default"),
}
_COMMON = dict(input_size=1, hidden_size=128, density=0.3,
               ste_temperature=1.0, entropy_lambda=0.1)


def make_model(cfg):
    kind, cls, decorr_choice = cfg
    if kind == "static":
        return SeqModel(cls(**_COMMON), _COMMON["hidden_size"],
                         _COMMON["entropy_lambda"])
    decorr = 0.0 if decorr_choice == "off" else 1e-5
    if kind == "pred":
        cell = cls(
            liquid_tau_strength=1.0, pred_gate_beta=4.0, ema_gamma=0.5,
            decorr_lambda=decorr, **_COMMON)
    elif kind == "accel":
        cell = cls(
            liquid_tau_strength=1.0, pred_gate_beta=4.0, ema_gamma=0.5,
            diff_order=2, decorr_lambda=decorr, **_COMMON)
    elif kind == "blend":
        cell = cls(
            liquid_tau_strength=1.0, pred_gate_beta=4.0, ema_gamma=0.5,
            gate_mode="blend", decorr_lambda=decorr, **_COMMON)
    return SeqModel(cell, _COMMON["hidden_size"],
                    _COMMON["entropy_lambda"])


def train_one(model, x_tr, y_tr, x_te, y_te, epochs, lr, bs, device):
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = x_tr.shape[0]
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(N)
        xb_all, yb_all = x_tr[perm], y_tr[perm]
        ep, nb = 0.0, 0
        for i in range(0, N, bs):
            xb = xb_all[i:i + bs].to(device)
            yb = yb_all[i:i + bs].to(device)
            if xb.shape[0] == 0:
                continue
            opt.zero_grad()
            pred = model(xb)
            mse = (pred - yb).pow(2).mean()
            extra = model.extra_loss()
            loss = mse + extra
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep += float(mse.item())
            nb += 1
        last = ep / max(nb, 1)
    model.eval()
    with torch.no_grad():
        pred_te = model(x_te.to(device))
        mse_all = float((pred_te - y_te.to(device)).pow(2).mean())
    return {"test_mse": mse_all, "train_loss_last": last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    ap.add_argument("--out", type=str,
                    default="analysis/all_gates_decorr_bench.json")
    args = ap.parse_args()

    device = torch.device("cpu")
    print(f"[bench] device={device} epochs={args.epochs} seeds={args.seeds}")
    data = load_henry_hub(T=args.T)
    print(f"[bench] data: train={data['x_tr'].shape[0]} test={data['x_te'].shape[0]} "
          f"hi_vol={int(data['hi_vol_mask'].sum())}")

    results = {"config": vars(args),
               "data_summary": {
                   "n_train": int(data["x_tr"].shape[0]),
                   "n_test": int(data["x_te"].shape[0]),
                   "n_hi_vol": int(data["hi_vol_mask"].sum()),
               },
               "cells": []}

    hi_vol_mask = data["hi_vol_mask"].to(device)

    for mode in args.modes:
        cfg = MODES[mode]
        for seed in args.seeds:
            torch.manual_seed(seed)
            model = make_model(cfg)
            t0 = time.time()
            out = train_one(model, data["x_tr"], data["y_tr"],
                            data["x_te"], data["y_te"],
                            args.epochs, args.lr, args.batch_size, device)
            el = time.time() - t0
            model.eval()
            with torch.no_grad():
                pred_te = model(data["x_te"].to(device))
                mse_hi = float(((pred_te - data["y_te"].to(device)).pow(2)
                                * hi_vol_mask.unsqueeze(-1).unsqueeze(-1).float()
                                ).sum() / max(hi_vol_mask.sum().item(), 1))
            results["cells"].append({
                "mode": mode, "seed": seed,
                "test_mse": out["test_mse"], "test_mse_hi_vol": mse_hi,
                "train_loss_last": out["train_loss_last"],
                "elapsed_sec": round(el, 2)})
            print(f"[bench] {mode:22s} s{seed} "
                  f"mse={out['test_mse']:.5f} hi_vol={mse_hi:.5f} "
                  f"({el:.1f}s)")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {outp}")

    def agg(key):
        s = {}
        for c in results["cells"]:
            s.setdefault(c["mode"], []).append(c[key])
        return s
    mse_all = agg("test_mse")
    mse_hi = agg("test_mse_hi_vol")

    print("\n[bench] mean test MSE — overall | high-vol:")
    print(f"{'mode':22s} | {'overall':>10s} | {'hi_vol':>10s}")
    for mode in args.modes:
        all_mse = sum(mse_all.get(mode, [])) / max(len(mse_all.get(mode, [])), 1)
        hi = sum(mse_hi.get(mode, [])) / max(len(mse_hi.get(mode, [])), 1)
        print(f"{mode:22s} | {all_mse:>10.5f} | {hi:>10.5f}")

    print("\n[bench] Δ%% vs OFF variant (negative=better):")
    pairs = [("pred_gated", "pred_gated_off", "pred_gated_default"),
             ("accel_gated", "accel_gated_off", "accel_gated_default"),
             ("blend_gated", "blend_gated_off", "blend_gated_default")]
    for label, off_m, def_m in pairs:
        base_all = sum(mse_all.get(off_m, [])) / max(len(mse_all.get(off_m, [])), 1)
        base_hi = sum(mse_hi.get(off_m, [])) / max(len(mse_hi.get(off_m, [])), 1)
        new_all = sum(mse_all.get(def_m, [])) / max(len(mse_all.get(def_m, [])), 1)
        new_hi = sum(mse_hi.get(def_m, [])) / max(len(mse_hi.get(def_m, [])), 1)
        d_all = 100 * (new_all - base_all) / max(abs(base_all), 1e-12)
        d_hi = 100 * (new_hi - base_hi) / max(abs(base_hi), 1e-12)
        ok = d_all <= 5.0 and d_hi <= 5.0
        print(f"  {label}: overall Δ%={d_all:+.1f}% hi_vol Δ%={d_hi:+.1f}%  "
              f"{'OK — decorrelation helps' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()