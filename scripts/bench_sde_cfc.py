"""Round 306 — bench for SDE-CfC (Stochastic Differential Equation CfC).

Reference: arXiv:2608.28702 (Stochastic Liquid Deformation Fields, 2026-08-27).

SDE-CfC adds a Gaussian perturbation σ_sde · ε to the pre-sigmoid time gate
inside every CfC cell, training-only, exact CfC limit at eval.

5 cells × 3 datasets × 2 seeds = 30 cells, 50 epochs each.

Conditions:
  - cfc            : vanilla CfC baseline
  - sde_005        : σ_sde = 0.05  (r192 input-noise match)
  - sde_010        : σ_sde = 0.10  (2× sweep)
  - sde_020        : σ_sde = 0.20  (4× sweep)
  - sde_learnable  : σ_sde learned via softplus, init 0.05

Datasets: sin_irr (D=2, T=32, missing_rate=0.3), structured_irr, random_irr.

Hypotheses:
  - small σ_sde (0.05-0.10) helps toy_sin (gate-noise regularises
    the temporal gate, like r192 input noise gave toy_sin -24%);
  - large σ_sde (0.20) hurts everywhere (saturates the sigmoid);
  - learnable σ_sde should drift toward the optimum of the grid sweep.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.cfc import CfCNetwork
from lnn.core.sde_cfc import SDECfCNetwork


def make_sin_irr(B=32, T=32, D=2, missing_rate=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).expand(B, T)
    x1 = torch.sin(t) + 0.05 * torch.randn(B, T, generator=g)
    x2 = torch.cos(t) + 0.05 * torch.randn(B, T, generator=g)
    x = torch.stack([x1, x2], dim=-1)
    mask = (torch.rand(B, T, D, generator=g) > missing_rate).float()
    x = x * mask
    y = torch.sin(t + math.pi / 4).unsqueeze(-1)
    return x, y


def make_structured_irr(B=32, T=32, D=2, missing_rate=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).expand(B, T)
    # Two-frequency structured signal
    x1 = torch.sin(2 * t) + 0.1 * torch.sin(7 * t)
    x2 = torch.cos(3 * t) + 0.1 * torch.cos(11 * t)
    x = torch.stack([x1 + 0.05 * torch.randn(B, T, generator=g),
                     x2 + 0.05 * torch.randn(B, T, generator=g)], dim=-1)
    mask = (torch.rand(B, T, D, generator=g) > missing_rate).float()
    x = x * mask
    y = (torch.sin(2 * t) * torch.cos(3 * t)).unsqueeze(-1)
    return x, y


def make_random_irr(B=32, T=32, D=2, missing_rate=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, D, generator=g)
    mask = (torch.rand(B, T, D, generator=g) > missing_rate).float()
    x = x * mask
    y = torch.randn(B, T, 1, generator=g)
    return x, y


def make_cfc(input_size, hidden_size, output_size, num_layers=1):
    return CfCNetwork(input_size, hidden_size, output_size, num_layers=num_layers)


def make_sde(input_size, hidden_size, output_size, num_layers=1, sigma=0.05, learnable=False):
    return SDECfCNetwork(
        input_size, hidden_size, output_size,
        num_layers=num_layers,
        sigma_sde=sigma,
        learnable_sigma=learnable,
    )


def train_and_eval(model, x_train, y_train, x_eval, y_eval, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        y_pred = model(x_train)
        loss = F.mse_loss(y_pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        y_pred_eval = model(x_eval)
        eval_mse = F.mse_loss(y_pred_eval, y_eval).item()
    return eval_mse


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--out", default="results/bench_sde_cfc.json")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.epochs = 20
        args.seeds = 1

    datasets = {
        "toy_sin": make_sin_irr,
        "structured_irr": make_structured_irr,
        "random_irr": make_random_irr,
    }
    conds = ["cfc", "sde_005", "sde_010", "sde_020", "sde_learnable"]

    rows = []
    t0 = time.time()
    for ds_name, mk in datasets.items():
        for seed in range(args.seeds):
            x_train, y_train = mk(seed=seed)
            x_eval, y_eval = mk(seed=seed + 100)
            for cond in conds:
                torch.manual_seed(seed)
                model = None
                if cond == "cfc":
                    model = make_cfc(2, 16, 1)
                elif cond == "sde_005":
                    model = make_sde(2, 16, 1, sigma=0.05)
                elif cond == "sde_010":
                    model = make_sde(2, 16, 1, sigma=0.10)
                elif cond == "sde_020":
                    model = make_sde(2, 16, 1, sigma=0.20)
                elif cond == "sde_learnable":
                    model = make_sde(2, 16, 1, sigma=0.05, learnable=True)
                assert model is not None

                mse = train_and_eval(model, x_train, y_train, x_eval, y_eval, args.epochs, 1e-3)
                row = {
                    "dataset": ds_name,
                    "seed": seed,
                    "cond": cond,
                    "mse": mse,
                    "params": count_params(model),
                }
                if hasattr(model, "cells") and hasattr(model.cells[0], "current_sigma"):
                    row["final_sigma_sde"] = model.cells[0].current_sigma()
                rows.append(row)
                print(f"[{ds_name} seed={seed} {cond}] mse={mse:.4f}"
                      + (f" σ={row.get('final_sigma_sde', 0):.4f}" if 'final_sigma_sde' in row else ""))

    elapsed = time.time() - t0
    summary: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        summary.setdefault(r["dataset"], {}).setdefault(r["cond"], []).append(r["mse"])

    # Compute per-dataset mean and deltas vs cfc
    final = {"elapsed_sec": elapsed, "epochs": args.epochs, "seeds": args.seeds, "rows": rows, "summary": {}}
    print("\n=== Mean MSE per dataset / cond ===")
    for ds, by_cond in summary.items():
        final["summary"][ds] = {}
        cfc_mean = sum(by_cond.get("cfc", [])) / max(1, len(by_cond.get("cfc", [])))
        for cond, vals in by_cond.items():
            m = sum(vals) / len(vals)
            delta = (m - cfc_mean) / cfc_mean * 100 if cfc_mean > 0 else 0.0
            final["summary"][ds][cond] = {"mean": m, "delta_pct": delta, "n": len(vals)}
            print(f"  {ds:18s} {cond:18s} mean={m:.4f}  Δ={delta:+6.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nWrote {args.out} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
