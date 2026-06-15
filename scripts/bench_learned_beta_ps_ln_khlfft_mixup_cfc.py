"""Round 197 — Mixup bench (PRD #10-159, 2026-06-16)."""
from __future__ import annotations

import json
import math
import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_mixup_cfc import (
    make_lbps_lnkhlfft_mixup_5_3_2,
    mixup_loss,
)


# ---- 3 dataset generators (consistent with r187-r196) ----

def make_sin_irr_dataset(n=400, T=32, D=2, missing_rate=0.3, seed=0):
    """Sin waves with random amplitude/freq + random missing values."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, T)
    amp = rng.uniform(0.5, 1.5, size=(n, 1))
    freq = rng.uniform(0.5, 1.5, size=(n, 1))
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1))
    x = np.zeros((n, T, D), dtype=np.float32)
    x[..., 0] = amp * np.sin(freq * t + phase)
    x[..., 1] = amp * np.cos(freq * t + phase)
    target = x[..., 0:1]
    mask = rng.uniform(size=x.shape) > missing_rate
    x = x * mask
    return torch.from_numpy(x), torch.from_numpy(target), mask.astype(np.float32)


def make_structured_irr_dataset(n=400, T=32, D=2, missing_rate=0.3, seed=0):
    """Sin (t < T/2) + linear (t >= T/2) + random missing."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)
    half = T // 2
    x = np.zeros((n, T, D), dtype=np.float32)
    x[:, :half, 0] = np.sin(t[:half] * 2 * np.pi)
    x[:, :half, 1] = np.cos(t[:half] * 2 * np.pi)
    x[:, half:, 0] = t[half:] * 2 - 1
    x[:, half:, 1] = t[half:] * 3 - 1
    target = x[..., 0:1]
    mask = rng.uniform(size=x.shape) > missing_rate
    x = x * mask
    return torch.from_numpy(x), torch.from_numpy(target), mask.astype(np.float32)


def make_random_irr_dataset(n=400, T=32, D=2, missing_rate=0.3, seed=0):
    """Pure noise with random missing values."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, T, D)).astype(np.float32) * 0.5
    target = x[..., 0:1]
    mask = rng.uniform(size=x.shape) > missing_rate
    x = x * mask
    return torch.from_numpy(x), torch.from_numpy(target), mask.astype(np.float32)


DATASETS = {
    "sin_irr": make_sin_irr_dataset,
    "structured_irr": make_structured_irr_dataset,
    "random_irr": make_random_irr_dataset,
}


def train_eval_split(x, y, train_frac=0.7):
    n = x.shape[0]
    n_tr = int(n * train_frac)
    return x[:n_tr], y[:n_tr], x[n_tr:], y[n_tr:]


def train_one(model, x_train, y_train, epochs=30, lr=1e-2, batch_size=16, use_mixup=True):
    """Train with optional mixup. Returns train_loss trajectory."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x_train.shape[0]
    losses = []
    for ep in range(epochs):
        # Shuffle
        idx = torch.randperm(n)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]
            opt.zero_grad()
            out = model(xb)
            if use_mixup and isinstance(out, tuple):
                y, mix_idx, lam = out
                loss = mixup_loss(y, yb, mix_idx, lam)
            else:
                y = out if not isinstance(out, tuple) else out[0]
                loss = F.mse_loss(y, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        losses.append(ep_loss / max(nb, 1))
    return losses


def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        out = model(x_test)
        if isinstance(out, tuple):
            y = out[0]
        else:
            y = out
        mse = F.mse_loss(y, y_test).item()
    return mse


def run_one_cell(ds_name, cond, alpha, seed, hidden=12, epochs=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds_fn = DATASETS[ds_name]
    x, y, _mask = ds_fn(seed=seed * 17 + 3)
    x_tr, y_tr, x_te, y_te = train_eval_split(x, y)
    # alpha=0 → MSE baseline (no mixup), else mixup with given alpha
    model = make_lbps_lnkhlfft_mixup_5_3_2(
        input_size=2, hidden_size=hidden, output_size=1,
        mixup_alpha=alpha,
    )
    t0 = time.time()
    train_one(model, x_tr, y_tr, epochs=epochs, use_mixup=(alpha > 0))
    elapsed = time.time() - t0
    test_mse = evaluate(model, x_te, y_te)
    return {
        "dataset": ds_name,
        "cond": cond,
        "alpha": alpha,
        "seed": seed,
        "test_mse": test_mse,
        "epochs": epochs,
        "elapsed_s": round(elapsed, 2),
    }


def main():
    """24 cells: 4 conds × 3 datasets × 2 seeds × 30 epochs."""
    conds = [
        ("mse", 0.0),     # pure MSE baseline (no mixup)
        ("mixup_01", 0.1),  # mild mixup
        ("mixup_02", 0.2),  # standard mixup
        ("mixup_04", 0.4),  # strong mixup
    ]
    rows = []
    for ds_name in DATASETS:
        for cond, alpha in conds:
            for seed in [0, 1]:
                row = run_one_cell(ds_name, cond, alpha, seed)
                rows.append(row)
                print(f"  {ds_name:18s} {cond:10s} α={alpha:.2f} seed={seed} "
                      f"test_mse={row['test_mse']:.4f} time={row['elapsed_s']}s")
    # Aggregate
    print("\n=== AGGREGATE (mean over seeds) ===")
    print(f"{'cond':12s} {'sin_irr':10s} {'struct_irr':10s} {'random_irr':10s} {'mean':10s} {'Δ':6s}")
    for cond, alpha in conds:
        per_ds = {}
        for ds_name in DATASETS:
            vals = [r["test_mse"] for r in rows
                    if r["cond"] == cond and r["dataset"] == ds_name]
            per_ds[ds_name] = np.mean(vals) if vals else float("nan")
        m = np.mean(list(per_ds.values()))
        delta = ""
        if cond == "mse":
            base_mean = m
        else:
            base = next(np.mean([r["test_mse"] for r in rows
                                  if r["cond"] == "mse" and r["dataset"] == ds])
                        for ds in DATASETS)
            base_mean = base
            delta = f"{(m / base_mean - 1) * 100:+.1f}%"
        print(f"{cond:12s} {per_ds['sin_irr']:.4f}     {per_ds['structured_irr']:.4f}     "
              f"{per_ds['random_irr']:.4f}     {m:.4f}     {delta}")
    # Save
    out = {
        "round": 197,
        "date": "2026-06-16",
        "config": "lbps_lnkhlfft_mixup_5_3_2",
        "epochs": 30,
        "seeds": [0, 1],
        "n_cells": len(rows),
        "rows": rows,
    }
    out_path = "results/bench_learned_beta_ps_ln_khlfft_mixup_cfc.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
