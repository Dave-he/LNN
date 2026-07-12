#!/usr/bin/env python3
"""r298: irregular TS validation of r295 decorrelation default.

Reuses r102 QuITE data loaders (sin_irr, structured_irr, random_irr)
but uses BlendGatedLiquidTauCfCCell (which has r295 decorrelation
default) instead of CfCCell. Tests whether the r295 SP generalizes
to irregular time series with ~50% missing rate.
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

from lnn.core.blend_gated_liquid_tau_cfc import (  # noqa: E402
    BlendGatedLiquidTauCfCCell,
)


# Data factories (copied from r102 bench_quite_irregular_ts.py).
def make_irregular(target_fn, T_max, D, seed,
                   gap_rate=0.3, nan_rate=0.5):
    rng = np.random.default_rng(seed)
    raw_times = np.cumsum(rng.exponential(1.0, size=T_max + 5))
    indices = np.sort(rng.choice(T_max + 5, size=T_max, replace=False))
    times = raw_times[indices]
    times = (times - times.min()) / (times.max() - times.min() + 1e-9)
    drop = rng.random(T_max) < gap_rate
    times[drop] = -1.0
    times_t = torch.tensor(times, dtype=torch.float32)
    target = target_fn(times_t).to(torch.float32)
    obs = torch.zeros(T_max, D, dtype=torch.float32)
    obs[:, 0] = target
    obs[:, 1:] = torch.randn(T_max, D - 1) * 0.3
    nan_mask = rng.random((T_max, D)) < nan_rate
    obs[nan_mask] = float("nan")
    mask = torch.tensor(times >= 0, dtype=torch.bool)
    times = torch.tensor(times, dtype=torch.float32)
    times[~mask] = 0.0
    return obs, times, mask, target


def make_sin_irr(T, D, seed):
    def fn(t):
        return torch.sin(2 * np.pi * t)
    return make_irregular(fn, T, D, seed)


def make_structured_irr(T, D, seed):
    def fn(t):
        out = torch.zeros_like(t)
        mask = t >= 0
        m1 = mask & (t < 0.5)
        m2 = mask & (t >= 0.5)
        out[m1] = torch.sin(2 * np.pi * t[m1])
        out[m2] = torch.sign(torch.sin(20 * np.pi * t[m2]))
        return out
    return make_irregular(fn, T, D, seed)


def make_random_irr(T, D, seed):
    def fn(t):
        out = torch.zeros_like(t)
        mask = t >= 0
        out[mask] = torch.randn(int(mask.sum()))
        return out
    return make_irregular(fn, T, D, seed)


DATASETS = {
    "sin_irr": make_sin_irr,
    "structured_irr": make_structured_irr,
    "random_irr": make_random_irr,
}


class SeqModel(nn.Module):
    def __init__(self, cell, hidden_size):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, obs, times, mask):
        clean = torch.where(torch.isfinite(obs), obs, torch.zeros_like(obs))
        T = clean.shape[0]
        x = clean.unsqueeze(0)  # (1, T, D)
        out, _ = self.cell(x)
        # Use last valid mask position's hidden state.
        last_t = int(torch.where(mask)[0][-1].item()) if mask.any() else T - 1
        return self.head(out[0, last_t:last_t + 1, :]).squeeze(0)

    def extra_loss(self):
        return self.cell.extra_loss()


# (mode_name, cell_kwargs)
def make_blend_off():
    return BlendGatedLiquidTauCfCCell(
        input_size=2, hidden_size=64, density=0.3,
        liquid_tau_strength=1.0, pred_gate_beta=4.0, ema_gamma=0.5,
        gate_mode="blend", entropy_lambda=0.0, decorr_lambda=0.0)


def make_blend_default():
    return BlendGatedLiquidTauCfCCell(
        input_size=2, hidden_size=64, density=0.3,
        liquid_tau_strength=1.0, pred_gate_beta=4.0, ema_gamma=0.5,
        gate_mode="blend", entropy_lambda=0.0)
    # r295: decorr_lambda=1e-5 default


MODES = {
    "static_tau": ("static", None),
    "blend_off": ("blend_off", None),
    "blend_default": ("blend_default", None),
}


def make_model(cfg):
    kind, _ = cfg
    if kind == "static":
        from lnn.core.ste_entropy_neuron_wise_cfc import STEWithEntropy
        cell = STEWithEntropy(input_size=2, hidden_size=64, density=0.3,
                               ste_temperature=1.0, entropy_lambda=0.1)
    elif kind == "blend_off":
        cell = make_blend_off()
    else:
        cell = make_blend_default()
    return SeqModel(cell, 64)


def train_one(model, obs_tr, target_tr, mask_tr, obs_te, target_te,
              mask_te, epochs, lr, bs, device):
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = obs_tr.shape[0]
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(N)
        ep, nb = 0.0, 0
        for i in range(0, N, bs):
            idx = perm[i:i + bs].numpy()
            opt.zero_grad()
            # Accumulate losses and backward once.
            losses = []
            ep_loss = 0.0
            for j in idx:
                pred = model(obs_tr[j].to(device),
                              torch.zeros(obs_tr[j].shape[0]).to(device),
                              mask_tr[j].to(device))
                # pred is scalar; target_tr[j] is (T,). Use last-step MSE.
                target_last = target_tr[j][-1].to(device)
                mse = (pred - target_last) ** 2
                extra = model.extra_loss()
                losses.append(mse + extra)
                ep_loss += float(mse.item())
                nb += 1
            total = torch.stack(losses).sum()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep += ep_loss
        last = ep / max(nb, 1)
    model.eval()
    test_mse = 0.0
    n_test = obs_te.shape[0]
    with torch.no_grad():
        for j in range(n_test):
            pred = model(obs_te[j].to(device),
                          torch.zeros(obs_te[j].shape[0]).to(device),
                          mask_te[j].to(device))
            target_last = target_te[j][-1].to(device)
            test_mse += float((pred - target_last).pow(2).item())
    return {"test_mse": test_mse / max(n_test, 1), "train_loss_last": last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    ap.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    ap.add_argument("--out", type=str,
                    default="analysis/irregular_decorrelation_bench.json")
    args = ap.parse_args()

    device = torch.device("cpu")
    print(f"[bench] device={device} epochs={args.epochs} seeds={args.seeds}")

    results = {"config": vars(args), "cells": []}

    for mode in args.modes:
        cfg = MODES[mode]
        for ds in args.datasets:
            for seed in args.seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                factory = DATASETS[ds]
                # Generate train and test sequences.
                obs_tr = []
                target_tr = []
                mask_tr = []
                obs_te = []
                target_te = []
                mask_te = []
                for j in range(args.n_train):
                    o, t, m, tgt = factory(args.T, args.D, seed * 1000 + j)
                    obs_tr.append(o)
                    target_tr.append(tgt)
                    mask_tr.append(m)
                for j in range(args.n_test):
                    o, t, m, tgt = factory(args.T, args.D,
                                              seed * 1000 + 10000 + j)
                    obs_te.append(o)
                    target_te.append(tgt)
                    mask_te.append(m)
                obs_tr_s = torch.stack(obs_tr)
                target_tr_s = torch.stack(target_tr)
                mask_tr_s = torch.stack(mask_tr)
                obs_te_s = torch.stack(obs_te)
                target_te_s = torch.stack(target_te)
                mask_te_s = torch.stack(mask_te)

                model = make_model(cfg)
                t0 = time.time()
                out = train_one(model, obs_tr_s, target_tr_s, mask_tr_s,
                                obs_te_s, target_te_s, mask_te_s,
                                args.epochs, args.lr,
                                args.n_train, device)
                el = time.time() - t0
                results["cells"].append({
                    "mode": mode, "dataset": ds, "seed": seed,
                    "test_mse": out["test_mse"],
                    "train_loss_last": out["train_loss_last"],
                    "elapsed_sec": round(el, 2)})
                print(f"[bench] {mode:14s} {ds:14s} s{seed} "
                      f"mse={out['test_mse']:.5f} ({el:.1f}s)")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {outp}")

    def agg(key):
        s = {}
        for c in results["cells"]:
            s.setdefault((c["mode"], c["dataset"]), []).append(c[key])
        return s
    mse = agg("test_mse")

    print("\n[bench] mean test MSE — irregular TS:")
    hdr = " | ".join(f"{d:>14s}" for d in args.datasets)
    print(f"{'mode':14s} | {hdr}")
    for mode in args.modes:
        cells = []
        for d in args.datasets:
            v = mse.get((mode, d), [])
            vm = sum(v) / len(v) if v else float("nan")
            cells.append(f"{vm:.5f}")
        print(f"{mode:14s} | " + " | ".join(f"{c:>14s}" for c in cells))

    print("\n[bench] Δ%% vs blend_off (negative=better):")
    for d in args.datasets:
        base = mse.get(("blend_off", d), [])
        if not base:
            continue
        bm = sum(base) / len(base)
        line = f"  {d:14s}: blend_off={bm:.5f}"
        for m in ("blend_default",):
            v = mse.get((m, d), [])
            if v:
                vm = sum(v) / len(v)
                line += f"  {m}={vm:.5f} ({100*(vm-bm)/max(abs(bm),1e-12):+.1f}%)"
        print(line)


if __name__ == "__main__":
    main()