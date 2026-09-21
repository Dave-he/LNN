"""Round 307 — bench for MDN-CfC (Mixture Density Network head on CfC backbone).

Reference: arXiv:2603.27058 (Liquid Networks with MDN Heads, 2026-03-28).

5 cells × 3 datasets × 2 seeds = 30 cells, 50 epochs.

Conditions:
  - mdn_k1 (baseline): MDN-CfC with K=1 mixture  → recovers MSE-CfC limit
  - mdn_k3         : MDN-CfC with K=3 mixtures
  - mdn_k5         : MDN-CfC with K=5 mixtures (paper default)
  - mse_cfc        : vanilla CfC + Linear head, MSE loss
  - mse_cfc_big    : CfC with hidden=32, MSE loss (size-matched param count)

Datasets: toy_sin, structured_irr, random_irr (D=2, T=32, missing_rate=0.3).

Hypotheses:
  - mdn_k1 ≈ mse_cfc  (K=1 Gaussian reduces to MSE head)
  - mdn_k3,5 ≥ mdn_k1 on structured_irr (multi-mode target)
  - mdn_k5 ≈ mse_cfc_big in params but should win on distribution quality
  - random_irr → all close (no structure to exploit)
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
from lnn.core.mdn import mdn_mean
from lnn.core.mdn_cfc import MDNCfCNetwork, mdn_cfc_loss


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
    x1 = torch.sin(2 * t) + 0.1 * torch.sin(7 * t)
    x2 = torch.cos(3 * t) + 0.1 * torch.cos(11 * t)
    x = torch.stack([x1 + 0.05 * torch.randn(B, T, generator=g),
                     x2 + 0.05 * torch.randn(B, T, generator=g)], dim=-1)
    mask = (torch.rand(B, T, D, generator=g) > missing_rate).float()
    x = x * mask
    # Two-mode target: bimodal on the product of two frequencies — gives the
    # mixture head something to capture that MSE can't.
    y_main = (torch.sin(2 * t) * torch.cos(3 * t)).unsqueeze(-1)
    y_alt = (-torch.sin(2 * t) * torch.cos(3 * t)).unsqueeze(-1)
    flip = (torch.rand(B, T, 1, generator=g) > 0.5).float()
    y = flip * y_main + (1.0 - flip) * y_alt
    return x, y


def make_random_irr(B=32, T=32, D=2, missing_rate=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, D, generator=g)
    mask = (torch.rand(B, T, D, generator=g) > missing_rate).float()
    x = x * mask
    y = torch.randn(B, T, 1, generator=g)
    return x, y


def make_mdn(input_size, hidden_size, output_size, num_layers=1, k=5):
    return MDNCfCNetwork(
        input_size, hidden_size, output_size,
        num_layers=num_layers, num_mixtures=k,
    )


def make_mse(input_size, hidden_size, output_size, num_layers=1):
    return CfCNetwork(input_size, hidden_size, output_size, num_layers=num_layers)


def train_and_eval_mdn(model, x_train, y_train, x_eval, y_eval, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = mdn_cfc_loss(model, x_train, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        y_pred_eval = model.predict(x_eval)
        eval_mse = F.mse_loss(y_pred_eval, y_eval).item()
    return eval_mse


def train_and_eval_mse(model, x_train, y_train, x_eval, y_eval, epochs, lr):
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
    p.add_argument("--out", default="results/bench_mdn_cfc.json")
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
    conds = ["mdn_k1", "mdn_k3", "mdn_k5", "mse_cfc", "mse_cfc_big"]

    rows = []
    t0 = time.time()
    for ds_name, mk in datasets.items():
        for seed in range(args.seeds):
            x_train, y_train = mk(seed=seed)
            x_eval, y_eval = mk(seed=seed + 100)
            for cond in conds:
                torch.manual_seed(seed)
                model = None
                if cond == "mdn_k1":
                    model = make_mdn(2, 16, 1, k=1)
                elif cond == "mdn_k3":
                    model = make_mdn(2, 16, 1, k=3)
                elif cond == "mdn_k5":
                    model = make_mdn(2, 16, 1, k=5)
                elif cond == "mse_cfc":
                    model = make_mse(2, 16, 1)
                elif cond == "mse_cfc_big":
                    model = make_mse(2, 32, 1)
                assert model is not None

                if cond.startswith("mdn_"):
                    mse = train_and_eval_mdn(model, x_train, y_train, x_eval, y_eval, args.epochs, 1e-3)
                else:
                    mse = train_and_eval_mse(model, x_train, y_train, x_eval, y_eval, args.epochs, 1e-3)

                rows.append({
                    "dataset": ds_name,
                    "seed": seed,
                    "cond": cond,
                    "mse": mse,
                    "params": count_params(model),
                })
                print(f"[{ds_name} seed={seed} {cond}] mse={mse:.4f} params={count_params(model)}")

    elapsed = time.time() - t0
    summary: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        summary.setdefault(r["dataset"], {}).setdefault(r["cond"], []).append(r["mse"])

    final = {"elapsed_sec": elapsed, "epochs": args.epochs, "seeds": args.seeds, "rows": rows, "summary": {}}
    print("\n=== Mean MSE per dataset / cond ===")
    for ds, by_cond in summary.items():
        final["summary"][ds] = {}
        mse_mean = sum(by_cond.get("mse_cfc", [])) / max(1, len(by_cond.get("mse_cfc", [])))
        for cond, vals in by_cond.items():
            m = sum(vals) / len(vals)
            delta = (m - mse_mean) / mse_mean * 100 if mse_mean > 0 else 0.0
            final["summary"][ds][cond] = {"mean": m, "delta_pct": delta, "n": len(vals)}
            print(f"  {ds:18s} {cond:18s} mean={m:.4f}  Δ={delta:+6.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nWrote {args.out} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()