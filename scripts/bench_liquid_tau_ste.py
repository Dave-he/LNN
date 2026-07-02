#!/usr/bin/env python3
"""Benchmark for LiquidTauSTECfCCell — liquid (input-dependent) τ (round 277).

Tests the 2026-literature-grounded hypothesis (PRD #10-114) that
restoring an **input-dependent time constant** on top of the STE
sparsity base helps on nonstationary / structured data, while not
hurting smooth data.

Conditions:
  * static_tau   — r267 production (STEWithEntropy, static per-neuron τ)
  * liquid_tau   — NEW (LiquidTauSTECfCCell, input-dependent τ)

All shared params match r267-r275 production:
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


# ---------------------------------------------------------------------------
# Toy data generators (single-channel) — identical to r276
# ---------------------------------------------------------------------------
def make_toy_sin(T: int = 64, n_samples: int = 256, seed: int = 0):
    t = torch.linspace(0, 1, T + 1).unsqueeze(0).repeat(n_samples, 1)
    y = torch.sin(2 * math.pi * t)
    x = y[:, :-1].unsqueeze(-1)
    y_target = y[:, 1:].unsqueeze(-1)
    return x, y_target


def make_structured(T: int = 64, n_samples: int = 256, seed: int = 0):
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
    x = y[:, :-1].unsqueeze(-1)
    y_target = y[:, 1:].unsqueeze(-1)
    return x, y_target


def make_random(T: int = 64, n_samples: int = 256, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(n_samples, T + 1, generator=g)
    x = y[:, :-1].unsqueeze(-1)
    y_target = y[:, 1:].unsqueeze(-1)
    return x, y_target


DATA_FACTORIES = {
    "toy_sin": make_toy_sin,
    "structured": make_structured,
    "random": make_random,
}


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------
class ReadoutHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, h):
        return self.head(h)


class SeqModel(nn.Module):
    def __init__(self, cell: nn.Module, hidden_size: int, entropy_lambda: float):
        super().__init__()
        self.cell = cell
        self.head = ReadoutHead(hidden_size)
        self.entropy_lambda = float(entropy_lambda)

    def forward(self, x):
        out, _ = self.cell(x)
        return self.head(out)

    def extra_loss(self) -> torch.Tensor:
        if self.entropy_lambda <= 0:
            return torch.tensor(0.0)
        return self.cell.extra_loss()


MODES = {
    "static_tau": dict(kind="static", hidden_size=192, density=0.3,
                       ste_temperature=1.0, entropy_lambda=0.1),
    "liquid_tau": dict(kind="liquid", hidden_size=192, density=0.3,
                       ste_temperature=1.0, entropy_lambda=0.1,
                       liquid_tau_strength=1.0),
}


def make_model(cfg: dict) -> SeqModel:
    if cfg["kind"] == "static":
        cell = STEWithEntropy(
            input_size=1, hidden_size=cfg["hidden_size"],
            density=cfg["density"], ste_temperature=cfg["ste_temperature"],
            entropy_lambda=cfg["entropy_lambda"],
        )
    else:
        cell = LiquidTauSTECfCCell(
            input_size=1, hidden_size=cfg["hidden_size"],
            density=cfg["density"], ste_temperature=cfg["ste_temperature"],
            entropy_lambda=cfg["entropy_lambda"],
            liquid_tau_strength=cfg["liquid_tau_strength"],
        )
    return SeqModel(cell, cfg["hidden_size"], cfg["entropy_lambda"])


def train_one(model, x_train, y_train, x_eval, y_eval,
              epochs, lr, batch_size, device):
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = x_train.shape[0]
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        x_tr = x_train[perm]
        y_tr = y_train[perm]
        epoch_loss = 0.0
        n_batches = 0
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


def collect_diagnostics(model: SeqModel, x_sample: torch.Tensor) -> dict:
    cell = model.cell
    diag = {"n_params": sum(p.numel() for p in model.parameters())}
    with torch.no_grad():
        if isinstance(cell, LiquidTauSTECfCCell):
            _, _, aux = cell(x_sample, return_aux=True)
            diag["tau_temporal_std"] = aux["tau_temporal_std"]
            diag["tau_dynamic_mean"] = aux["tau_dynamic_mean"]
            diag["tau_dynamic_min"] = aux["tau_dynamic_min"]
            diag["tau_dynamic_max"] = aux["tau_dynamic_max"]
        else:
            _, _, aux = cell(x_sample, return_aux=True)
            diag["tau_temporal_std"] = 0.0  # static τ never moves
            diag["tau_dynamic_mean"] = aux["tau_summary"]["mean"]
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
                        default="analysis/liquid_tau_ste_bench.json")
    parser.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bench] device={device}")

    results = {
        "config": {
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "T": args.T, "n_samples": args.n_samples, "seeds": args.seeds,
            "datasets": args.datasets, "modes": args.modes,
        },
        "cells": [],
    }

    for mode_name in args.modes:
        cfg = MODES[mode_name]
        for ds_name in args.datasets:
            for seed in args.seeds:
                torch.manual_seed(seed)
                factory = DATA_FACTORIES[ds_name]
                x, y = factory(T=args.T, n_samples=args.n_samples, seed=seed)
                n_train = int(0.8 * x.shape[0])
                x_train, y_train = x[:n_train], y[:n_train]
                x_eval, y_eval = x[n_train:], y[n_train:]

                model = make_model(cfg)
                t0 = time.time()
                out = train_one(model, x_train, y_train, x_eval, y_eval,
                                epochs=args.epochs, lr=args.lr,
                                batch_size=args.batch_size, device=device)
                elapsed = time.time() - t0
                diag = collect_diagnostics(model, x_eval[:16].to(device))

                cell_result = {
                    "mode": mode_name, "kind": cfg["kind"],
                    "dataset": ds_name, "seed": seed,
                    "test_mse": out["test_mse"],
                    "train_loss_last": out["train_loss_last"],
                    "elapsed_sec": round(elapsed, 2),
                    "diagnostics": diag,
                }
                results["cells"].append(cell_result)
                print(
                    f"[bench] mode={mode_name:12s} ds={ds_name:10s} seed={seed} "
                    f"test_mse={out['test_mse']:.6f} "
                    f"tau_tstd={diag['tau_temporal_std']:.4f} ({elapsed:.1f}s)"
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {out_path}")

    # Summary: mean test_mse by mode × dataset.
    summary = {}
    for cell in results["cells"]:
        summary.setdefault((cell["mode"], cell["dataset"]), []).append(cell["test_mse"])
    print("\n[bench] Mean test_mse by mode × dataset:")
    print(f"{'mode':12s} | " + " | ".join(f"{ds:>12s}" for ds in args.datasets))
    for mode_name in args.modes:
        cols = []
        for ds in args.datasets:
            vals = summary.get((mode_name, ds), [])
            cols.append(sum(vals) / len(vals) if vals else float("nan"))
        cells_str = " | ".join(f"{c:>12.6f}" for c in cols)
        print(f"{mode_name:12s} | {cells_str}")

    # Delta: liquid vs static (negative = liquid better).
    if "static_tau" in args.modes and "liquid_tau" in args.modes:
        print("\n[bench] liquid vs static Δ%% (negative = liquid better):")
        for ds in args.datasets:
            s = summary.get(("static_tau", ds), [])
            l = summary.get(("liquid_tau", ds), [])
            if s and l:
                sm = sum(s) / len(s)
                lm = sum(l) / len(l)
                pct = 100.0 * (lm - sm) / max(abs(sm), 1e-12)
                print(f"  {ds:12s}: static={sm:.6f} liquid={lm:.6f} Δ={pct:+.1f}%")


if __name__ == "__main__":
    main()
