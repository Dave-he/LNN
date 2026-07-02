#!/usr/bin/env python3
"""Benchmark for AccelGatedLiquidTauCfCCell (round 279).

Tests whether gating the liquid τ on ACCELERATION (2nd difference, |Δ²x|)
instead of VELOCITY (1st difference, |Δ¹x|, r278) recovers r277's toy_sin
win while keeping r278's random fix — i.e. turns the gated liquid τ into
a strict Pareto default.

Four modes (the full story):
  * static_tau    — r267 production (static per-neuron τ)
  * liquid_tau    — r277 (input-dependent τ, ungated)
  * gated_vel     — r278 (velocity gate, |Δ¹x|)  [reproduces r278]
  * gated_accel   — r279 NEW (acceleration gate, |Δ²x|)

Shared params match r267-r278 production:
  input_size=1, hidden=192, T=64, density=0.3, ste_temperature=1.0,
  entropy_lambda=0.1, 100 epochs, lr=1e-2, batch=16.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Toy data generators (identical to r276/r277/r278)
# ---------------------------------------------------------------------------
def make_toy_sin(T=64, n_samples=256, seed=0):
    t = torch.linspace(0, 1, T + 1).unsqueeze(0).repeat(n_samples, 1)
    y = torch.sin(2 * math.pi * t)
    return y[:, :-1].unsqueeze(-1), y[:, 1:].unsqueeze(-1)


def make_structured(T=64, n_samples=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_segments = 4
    seg_len = (T + 1) // n_segments
    levels = torch.tensor([0.0, 1.0, -0.5, 0.7])
    y = torch.zeros(n_samples, T + 1)
    for i in range(n_segments):
        start = i * seg_len
        end = (i + 1) * seg_len if i < n_segments - 1 else T + 1
        y[:, start:end] = levels[i % len(levels)]
    y = y + torch.randn(n_samples, T + 1, generator=g) * 0.01
    return y[:, :-1].unsqueeze(-1), y[:, 1:].unsqueeze(-1)


def make_random(T=64, n_samples=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(n_samples, T + 1, generator=g)
    return y[:, :-1].unsqueeze(-1), y[:, 1:].unsqueeze(-1)


DATA_FACTORIES = {
    "toy_sin": make_toy_sin,
    "structured": make_structured,
    "random": make_random,
}


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
            liquid_tau_strength=1.0,
            pred_gate_beta=cfg["pred_gate_beta"],
            ema_gamma=cfg["ema_gamma"], **_COMMON)
    elif cfg["kind"] == "gated_accel":
        cell = AccelGatedLiquidTauCfCCell(
            liquid_tau_strength=1.0,
            pred_gate_beta=cfg["pred_gate_beta"],
            ema_gamma=cfg["ema_gamma"], diff_order=2, **_COMMON)
    else:  # gated_blend
        cell = BlendGatedLiquidTauCfCCell(
            liquid_tau_strength=1.0,
            pred_gate_beta=cfg["pred_gate_beta"],
            ema_gamma=cfg["ema_gamma"], gate_mode="blend", **_COMMON)
    return SeqModel(cell, _COMMON["hidden_size"], _COMMON["entropy_lambda"])


def train_one(model, x_train, y_train, x_eval, y_eval,
              epochs, lr, batch_size, device):
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = x_train.shape[0]
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        x_tr, y_tr = x_train[perm], y_train[perm]
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, N, batch_size):
            xb = x_tr[i:i + batch_size].to(device)
            yb = y_tr[i:i + batch_size].to(device)
            if xb.shape[0] == 0:
                continue
            pred = model(xb)
            mse = (pred - yb).pow(2).mean()
            loss = mse + model.extra_loss()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            epoch_loss += float(mse.item())
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
    model.eval()
    with torch.no_grad():
        pred = model(x_eval.to(device))
        test_mse = float((pred - y_eval.to(device)).pow(2).mean().item())
    return {"test_mse": test_mse, "train_loss_last": losses[-1]}


def collect_diagnostics(model, x_sample):
    cell = model.cell
    diag = {"n_params": sum(p.numel() for p in model.parameters())}
    with torch.no_grad():
        if isinstance(cell, (PredictabilityGatedLiquidTauCfCCell,)):
            # AccelGated is a subclass of PredictabilityGated.
            _, _, aux = cell(x_sample, return_aux=True)
            diag.update(tau_temporal_std=aux["tau_temporal_std"],
                        gate_mean=aux["gate_mean"], gate_min=aux["gate_min"],
                        gate_max=aux["gate_max"])
        elif isinstance(cell, LiquidTauSTECfCCell):
            _, _, aux = cell(x_sample, return_aux=True)
            diag.update(tau_temporal_std=aux["tau_temporal_std"],
                        gate_mean=1.0, gate_min=1.0, gate_max=1.0)
        else:
            _, _, aux = cell(x_sample, return_aux=True)
            diag.update(tau_temporal_std=0.0, gate_mean=float("nan"),
                        gate_min=float("nan"), gate_max=float("nan"))
    return diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--datasets", nargs="+",
                        default=["toy_sin", "structured", "random"])
    parser.add_argument("--out", type=str,
                        default="analysis/blend_gated_bench.json")
    parser.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bench] device={device}")

    results = {"config": {"epochs": args.epochs, "lr": args.lr,
                          "batch_size": args.batch_size, "T": args.T,
                          "seeds": args.seeds, "datasets": args.datasets,
                          "modes": args.modes}, "cells": []}

    for mode_name in args.modes:
        cfg = MODES[mode_name]
        for ds_name in args.datasets:
            for seed in args.seeds:
                torch.manual_seed(seed)
                x, y = DATA_FACTORIES[ds_name](
                    T=args.T, n_samples=args.n_samples, seed=seed)
                n_train = int(0.8 * x.shape[0])
                model = make_model(cfg)
                t0 = time.time()
                out = train_one(model, x[:n_train], y[:n_train],
                                x[n_train:], y[n_train:],
                                epochs=args.epochs, lr=args.lr,
                                batch_size=args.batch_size, device=device)
                elapsed = time.time() - t0
                diag = collect_diagnostics(model, x[n_train:][:16].to(device))
                results["cells"].append({
                    "mode": mode_name, "dataset": ds_name, "seed": seed,
                    "test_mse": out["test_mse"],
                    "train_loss_last": out["train_loss_last"],
                    "elapsed_sec": round(elapsed, 2), "diagnostics": diag})
                print(f"[bench] mode={mode_name:13s} ds={ds_name:10s} "
                      f"seed={seed} test_mse={out['test_mse']:.6f} "
                      f"gate={diag['gate_mean']:.3f} "
                      f"tau_tstd={diag['tau_temporal_std']:.4f} ({elapsed:.1f}s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {out_path}")

    summary = {}
    for cell in results["cells"]:
        summary.setdefault((cell["mode"], cell["dataset"]), []).append(
            cell["test_mse"])
    print("\n[bench] Mean test_mse by mode × dataset:")
    print(f"{'mode':13s} | " + " | ".join(f"{ds:>12s}" for ds in args.datasets))
    for mode_name in args.modes:
        cols = [sum(v) / len(v) if (v := summary.get((mode_name, ds), []))
                else float("nan") for ds in args.datasets]
        print(f"{mode_name:13s} | " + " | ".join(f"{c:>12.6f}" for c in cols))

    # Δ vs static baseline (negative = better than static).
    print("\n[bench] Δ%% vs static_tau (negative = better):")
    for ds in args.datasets:
        s = summary.get(("static_tau", ds), [])
        if not s:
            continue
        sm = sum(s) / len(s)
        line = f"  {ds:12s}: static={sm:.6f}"
        for m in ("liquid_tau", "gated_vel", "gated_accel", "gated_blend"):
            v = summary.get((m, ds), [])
            if v:
                vm = sum(v) / len(v)
                pct = 100.0 * (vm - sm) / max(abs(sm), 1e-12)
                line += f"  {m}={vm:.6f} ({pct:+.1f}%)"
        print(line)


if __name__ == "__main__":
    main()
