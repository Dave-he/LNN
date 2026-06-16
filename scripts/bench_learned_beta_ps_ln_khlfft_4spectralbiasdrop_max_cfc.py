"""Round 220 — bench for 4ScaleSpectralBiasDropMaxCfC (PRD #10-182, 2026-06-16)."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_cfc import make_lbps_lnkhlfft_5_3_2
from lnn.core.learned_beta_ps_ln_khlfft_4spectralbiasdrop_cfc import make_lbps_lnkhlfft_4spectralbiasdrop_5_3_2
from lnn.core.learned_beta_ps_ln_khlfft_4spectralbiasdrop_max_cfc import make_lbps_lnkhlfft_4spectralbiasdrop_max_5_3_2


def make_sin_irr_dataset(n=400, T=32, D=2, missing_rate=0.3, seed=0):
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
    return torch.from_numpy(x), torch.from_numpy(target)


def make_structured_irr_dataset(n=400, T=32, D=2, missing_rate=0.3, seed=0):
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
    return torch.from_numpy(x), torch.from_numpy(target)


def make_random_irr_dataset(n=400, T=32, D=2, missing_rate=0.3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, T, D)).astype(np.float32) * 0.5
    target = x[..., 0:1]
    mask = rng.uniform(size=x.shape) > missing_rate
    x = x * mask
    return torch.from_numpy(x), torch.from_numpy(target)


DATASETS = {
    "sin_irr": make_sin_irr_dataset,
    "structured_irr": make_structured_irr_dataset,
    "random_irr": make_random_irr_dataset,
}


def train_one(model, x_train, y_train, epochs=30, lr=1e-2, batch_size=16):
    n = x_train.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            yp = model(xb)
            loss = (yp - yb).pow(2).mean()
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def run_one_cell(ds_name, cond, seed, hidden=16, epochs=30):
    x, y = DATASETS[ds_name](seed=seed)
    n = x.shape[0]
    n_tr = int(0.8 * n)
    x_tr, y_tr = x[:n_tr], y[:n_tr]
    x_te, y_te = x[n_tr:], y[n_tr:]
    D = x.shape[-1]
    if cond == "cf":
        model = make_lbps_lnkhlfft_5_3_2(D, hidden, 1)
    elif cond == "4spectralbiasdrop":
        model = make_lbps_lnkhlfft_4spectralbiasdrop_5_3_2(D, hidden, 1)
    elif cond == "4spectralbiasdrop_max":
        model = make_lbps_lnkhlfft_4spectralbiasdrop_max_5_3_2(D, hidden, 1)
    else:
        raise ValueError(cond)
    t0 = time.time()
    train_one(model, x_tr, y_tr, epochs=epochs)
    model.eval()
    with torch.no_grad():
        yp = model(x_te)
        test_mse = float((yp - y_te).pow(2).mean().item())
    return test_mse, time.time() - t0


def main():
    conds = ["cf", "4spectralbiasdrop", "4spectralbiasdrop_max"]
    seeds = [0, 1]
    rows = []
    for ds_name in DATASETS:
        for cond in conds:
            for seed in seeds:
                mse, elapsed = run_one_cell(ds_name, cond, seed)
                rows.append({
                    "dataset": ds_name, "cond": cond, "seed": seed,
                    "test_mse": mse, "epochs": 30, "elapsed_s": round(elapsed, 2),
                })
                print(f"  {ds_name:18s} {cond:30s} s{seed}  test_mse={mse:.4f}  ({elapsed:.1f}s)")
    out = {
        "round": 220, "date": "2026-06-16",
        "config": "lbps_lnkhlfft_4spectralbiasdrop_max_5_3_2 (max combo) vs r216 (avg) vs baseline",
        "epochs": 30, "seeds": seeds, "n_cells": len(rows), "rows": rows,
    }
    out_path = "results/bench_learned_beta_ps_ln_khlfft_4spectralbiasdrop_max_cfc.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
