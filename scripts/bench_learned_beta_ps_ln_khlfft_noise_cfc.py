"""Round 192 — bench for input Gaussian noise augmentation (PRD #10-154)."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_noise_cfc import make_lbps_lnkhlfft_noise_5_3_2
from lnn.core.learned_beta_ps_ln_khlfft_cfc import make_lbps_lnkhlfft_5_3_2


def make_sin_irr(B=32, T=32, D=2, missing_rate=0.3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, T)
    y = np.zeros((B, T, D), dtype=np.float32)
    y[..., 0] = np.sin(t)[None, :]
    y[..., 1] = np.cos(t)[None, :]
    target = y[..., 0:1].copy()
    mask = rng.random((B, T, D)) < missing_rate
    y[mask] = np.nan
    return y, target


def make_structured_irr(B=32, T=32, D=2, missing_rate=0.3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, T)
    y = np.zeros((B, T, D), dtype=np.float32)
    y[..., 0] = np.sin(t)[None, :]
    y[..., 1] = np.cos(t)[None, :] * 0.5 + 0.5 * t[None, :] / (4 * np.pi)
    target = y[..., 0:1].copy()
    mask = rng.random((B, T, D)) < missing_rate
    y[mask] = np.nan
    return y, target


def make_random_irr(B=32, T=32, D=2, missing_rate=0.3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((B, T, D)).astype(np.float32) * 0.3
    target = y[..., 0:1].copy()
    mask = rng.random((B, T, D)) < missing_rate
    y[mask] = np.nan
    return y, target


def to_tensor(arr, device="cpu"):
    return torch.from_numpy(np.nan_to_num(arr, nan=0.0)).to(device)


def make_nan_mask(arr, device="cpu"):
    return torch.from_numpy(np.isnan(arr).astype(np.float32)).to(device)


def train_and_eval(model, x, target, mask, n_epochs=30, lr=1e-2, device="cpu"):
    x_t = to_tensor(x, device)
    t_t = torch.from_numpy(target).to(device)
    m_t = mask
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    initial_loss = None
    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        y = model(x_t)
        loss = F.mse_loss(y, t_t)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        y_eval = model(x_t)
        eval_mse = F.mse_loss(y_eval, t_t).item()
    return {"initial_loss": initial_loss, "eval_mse": eval_mse}


def run_dataset(name, make_fn, n_seeds=2, n_epochs=30, hidden=12):
    device = "cpu"
    results = []
    conds = [
        ("lbps_lnkhlfft_5_3_2_mse", lambda: make_lbps_lnkhlfft_5_3_2(2, hidden, 1)),
        ("lbps_lnkhlfft_5_3_2_noise05", lambda: make_lbps_lnkhlfft_noise_5_3_2(2, hidden, 1, noise_sigma=0.05)),
        ("lbps_lnkhlfft_5_3_2_noise10", lambda: make_lbps_lnkhlfft_noise_5_3_2(2, hidden, 1, noise_sigma=0.10)),
    ]
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        x, target = make_fn(seed=seed)
        mask = make_nan_mask(x)
        for cname, ctor in conds:
            torch.manual_seed(seed)
            model = ctor()
            n_params = sum(p.numel() for p in model.parameters())
            r = train_and_eval(model, x, target, mask, n_epochs=n_epochs, device=device)
            r["cond"] = cname
            r["seed"] = seed
            r["n_params"] = n_params
            results.append(r)
            print(f"  {name:12s} {cname:32s} seed={seed} eval_mse={r['eval_mse']:.4f}")
    return results


def main():
    n_epochs = 30
    n_seeds = 2
    out = {"n_epochs": n_epochs, "n_seeds": n_seeds, "datasets": {}}
    for name, fn in [
        ("sin", make_sin_irr),
        ("structured", make_structured_irr),
        ("random", make_random_irr),
    ]:
        print(f"\n=== {name} ===")
        out["datasets"][name] = run_dataset(name, fn, n_seeds=n_seeds, n_epochs=n_epochs)
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "bench_learned_beta_ps_ln_khlfft_noise_cfc.json")
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n=== Summary (mean eval_mse ± std) ===")
    for ds, rows in out["datasets"].items():
        print(f"\n{ds}:")
        for cname in ["lbps_lnkhlfft_5_3_2_mse", "lbps_lnkhlfft_5_3_2_noise05", "lbps_lnkhlfft_5_3_2_noise10"]:
            vals = [r["eval_mse"] for r in rows if r["cond"] == cname]
            if vals:
                m = np.mean(vals)
                s = np.std(vals)
                print(f"  {cname:32s} {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
