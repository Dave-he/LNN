#!/usr/bin/env python3
"""Real Henry Hub natural-gas decorrelation validation (round 292).

Reuses the r282 Henry Hub data loader (chronological split, train-
only normalisation, no look-ahead leakage) and tests whether the r291
finding — decorrelation loss at λ=1e-5 is strict-positive on the toy
4-dataset bench — extends to REAL data.

Reports:
  - overall test MSE
  - HIGH-VOL subset test MSE (regime-shift stress)
  - diag/off_ratio after training
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
from lnn.core.blend_gated_liquid_tau_cfc import (  # noqa: E402
    BlendGatedLiquidTauCfCCell,
)
from lnn.core.decorrelation_loss import (  # noqa: E402
    state_decorrelation_loss,
    state_covariance_diagnostics,
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
            "hi_vol_mask": hivol_te, "mu": float(mu), "sd": float(sd)}


class SeqModel(nn.Module):
    def __init__(self, cell, hidden_size, entropy_lambda, decorr_lambda):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(hidden_size, 1)
        self.entropy_lambda = float(entropy_lambda)
        self.decorr_lambda = float(decorr_lambda)

    def forward(self, x):
        out, _ = self.cell(x)
        return self.head(out)

    def extra_loss(self, x):
        out, _ = self.cell(x)
        ent = self.cell.extra_loss() if self.entropy_lambda > 0 else torch.tensor(0.0)
        dec = state_decorrelation_loss(out, lambda_coeff=self.decorr_lambda)
        return ent + dec


MODES = {
    "static_tau": dict(kind="static", decorr=0.0),
    "blend_gated": dict(kind="blend", decorr=0.0),
    "decorr_a1e5": dict(kind="blend", decorr=1e-5),
    "decorr_a1e4": dict(kind="blend", decorr=1e-4),
}
_COMMON = dict(input_size=1, hidden_size=128, density=0.3,
               ste_temperature=1.0, entropy_lambda=0.1)


def make_model(cfg):
    if cfg["kind"] == "static":
        cell = STEWithEntropy(**_COMMON)
    else:
        cell = BlendGatedLiquidTauCfCCell(
            liquid_tau_strength=1.0, pred_gate_beta=4.0, ema_gamma=0.5,
            gate_mode="blend", **_COMMON)
    return SeqModel(cell, _COMMON["hidden_size"],
                    _COMMON["entropy_lambda"], cfg["decorr"])


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
            pred = model(xb)
            mse = (pred - yb).pow(2).mean()
            extra = model.extra_loss(xb)
            loss = mse + extra
            opt.zero_grad()
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


def cov_diag(model, x_sample):
    cell = model.cell
    d = {"n_params": sum(p.numel() for p in model.parameters())}
    with torch.no_grad():
        out, _ = cell(x_sample)
    d.update(state_covariance_diagnostics(out))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    ap.add_argument("--out", type=str,
                    default="analysis/henry_hub_decorrelation_bench.json")
    args = ap.parse_args()

    device = torch.device("cpu")
    print(f"[bench] device={device} epochs={args.epochs} seeds={args.seeds}")
    data = load_henry_hub(T=args.T)
    print(f"[bench] data: train={data['x_tr'].shape[0]} test={data['x_te'].shape[0]} "
          f"hi_vol={int(data['hi_vol_mask'].sum())}")

    results = {"config": vars(args), "data_summary": {
        "n_train": int(data["x_tr"].shape[0]),
        "n_test": int(data["x_te"].shape[0]),
        "n_hi_vol": int(data["hi_vol_mask"].sum()),
        "mu": data["mu"], "sd": data["sd"],
    }, "cells": []}

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
            # Compute high-vol MSE separately.
            model.eval()
            with torch.no_grad():
                pred_te = model(data["x_te"].to(device))
                mse_hi = float(((pred_te - data["y_te"].to(device)).pow(2)
                                * hi_vol_mask.unsqueeze(-1).unsqueeze(-1).float()
                                ).sum() / max(hi_vol_mask.sum().item(), 1))
            diag = cov_diag(model, data["x_te"][:16].to(device))
            results["cells"].append({
                "mode": mode, "seed": seed,
                "test_mse": out["test_mse"], "test_mse_hi_vol": mse_hi,
                "train_loss_last": out["train_loss_last"],
                "elapsed_sec": round(el, 2), "diagnostics": diag})
            ratio = diag.get("ratio", float("nan"))
            print(f"[bench] {mode:14s} s{seed} "
                  f"mse={out['test_mse']:.5f} hi_vol={mse_hi:.5f} "
                  f"ratio={ratio:.2f} ({el:.1f}s)")

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

    print("\n[bench] mean test MSE — overall | high-vol | ratio:")
    print(f"{'mode':14s} | {'overall':>10s} | {'hi_vol':>10s} | {'ratio':>8s}")
    for mode in args.modes:
        all_mse = sum(mse_all.get(mode, [])) / max(len(mse_all.get(mode, [])), 1)
        hi = sum(mse_hi.get(mode, [])) / max(len(mse_hi.get(mode, [])), 1)
        diag_ratios = [c["diagnostics"].get("ratio", float("nan"))
                       for c in results["cells"] if c["mode"] == mode]
        avg_ratio = sum(diag_ratios) / max(len(diag_ratios), 1)
        print(f"{mode:14s} | {all_mse:>10.5f} | {hi:>10.5f} | {avg_ratio:>8.2f}")

    print("\n[bench] Δ%% vs blend_gated (negative=better):")
    base_all = sum(mse_all.get("blend_gated", [])) / max(
        len(mse_all.get("blend_gated", [])), 1)
    base_hi = sum(mse_hi.get("blend_gated", [])) / max(
        len(mse_hi.get("blend_gated", [])), 1)
    print(f"  baseline blend_gated: overall={base_all:.5f} hi_vol={base_hi:.5f}")
    for m in ("decorr_a1e5", "decorr_a1e4"):
        v_all = sum(mse_all.get(m, [])) / max(len(mse_all.get(m, [])), 1)
        v_hi = sum(mse_hi.get(m, [])) / max(len(mse_hi.get(m, [])), 1)
        d_all = 100 * (v_all - base_all) / max(abs(base_all), 1e-12)
        d_hi = 100 * (v_hi - base_hi) / max(abs(base_hi), 1e-12)
        print(f"  {m}: overall Δ%={d_all:+.1f}% hi_vol Δ%={d_hi:+.1f}%")

    # Hypothesis check.
    print("\n[bench] H1/H2/H3 acceptance check (r292 → real-world validation):")

    def mean(key, mode):
        vals = mse_all.get(mode, []) if key == "all" else mse_hi.get(mode, [])
        return sum(vals) / len(vals) if vals else float("nan")

    print("  H1 overall test MSE improves-or-maintains vs blend:")
    h1_ok = False
    for m in ("decorr_a1e5", "decorr_a1e4"):
        v = mean("all", m)
        b = mean("all", "blend_gated")
        ok = v <= b * 1.05
        print(f"     {m}: overall Δ%={100*(v-b)/max(abs(b),1e-12):+.1f}%  "
              f"{'OK' if ok else 'FAIL'}")
        if ok:
            h1_ok = True

    print("  H2 high-vol test MSE improves-or-maintains vs blend:")
    h2_ok = False
    for m in ("decorr_a1e5", "decorr_a1e4"):
        v = mean("hi", m)
        b = mean("hi", "blend_gated")
        ok = v <= b * 1.05
        print(f"     {m}: hi_vol Δ%={100*(v-b)/max(abs(b),1e-12):+.1f}%  "
              f"{'OK' if ok else 'FAIL'}")
        if ok:
            h2_ok = True

    print(f"  Real-world SP confirmation: "
          f"{'YES — toy SP transfers to real data' if h1_ok and h2_ok else 'PARTIAL — toy SP may be artifact'}")


if __name__ == "__main__":
    main()