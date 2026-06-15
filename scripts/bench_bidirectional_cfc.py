"""Round 144 — Bench Bidirectional CfC on 3 datasets.

Tests whether forward+backward CfC helps our 1D time-series modelling.

Cells: 3 conditions × 3 datasets × 2 seeds = 18 cells
Datasets: sin_irr, structured_irr, random_irr
Conditions:
- cfc (baseline, vanilla unidirectional CfC)
- bidi_concat (bidirectional, concat forward+backward)
- bidi_weighted (bidirectional, learned per-step alpha weighting)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.cfc import CfCNetwork
from lnn.core.bidirectional_cfc import BidirectionalCfCStackedNetwork


def make_sin_irr(B: int, T: int, D: int = 2, missing_rate: float = 0.3) -> torch.Tensor:
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
    x = torch.zeros(B, T, D)
    for i in range(D):
        phase = i * 0.5
        x[:, :, i] = torch.sin(t.squeeze(-1) + phase)
    mask = torch.rand(B, T, D) < missing_rate
    x[mask] = float("nan")
    return x


def make_structured_irr(B: int, T: int, D: int = 2, missing_rate: float = 0.3) -> torch.Tensor:
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
    x = torch.zeros(B, T, D)
    for i in range(D):
        regime = (torch.arange(T) >= T // 2).float()
        x[:, :, i] = torch.sin(t.squeeze(-1) * (1.0 + regime)) + i * 0.3
    mask = torch.rand(B, T, D) < missing_rate
    x[mask] = float("nan")
    return x


def make_random_irr(B: int, T: int, D: int = 2, missing_rate: float = 0.3) -> torch.Tensor:
    x = torch.randn(B, T, D).cumsum(dim=1) * 0.1
    mask = torch.rand(B, T, D) < missing_rate
    x[mask] = float("nan")
    return x


def make_target(x: torch.Tensor) -> torch.Tensor:
    y = x[:, :, 0]
    return y.unsqueeze(-1)


DATASETS = {
    "sin_irr": make_sin_irr,
    "structured_irr": make_structured_irr,
    "random_irr": make_random_irr,
}


def make_net(cond: str, input_size: int = 2, hidden_size: int = 16, output_size: int = 1):
    if cond == "cfc":
        return CfCNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=2, return_sequences=True,
        )
    elif cond == "bidi_concat":
        return BidirectionalCfCStackedNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=2, merge_mode="concat", return_sequences=True,
        )
    elif cond == "bidi_weighted":
        return BidirectionalCfCStackedNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=2, merge_mode="weighted", return_sequences=True,
        )
    else:
        raise ValueError(cond)


def train_one(cond: str, dataset: str, seed: int, epochs: int = 30,
              B: int = 8, T: int = 32) -> Dict:
    torch.manual_seed(seed)
    net = make_net(cond)

    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    X_train = DATASETS[dataset](B, T)
    y_train = make_target(X_train)
    X_test = DATASETS[dataset](B, T)
    y_test = make_target(X_test)
    y_train_clean = torch.nan_to_num(y_train, nan=0.0)
    y_test_clean = torch.nan_to_num(y_test, nan=0.0)
    X_train_clean = torch.nan_to_num(X_train, nan=0.0)
    X_test_clean = torch.nan_to_num(X_test, nan=0.0)

    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        out = net(X_train_clean)
        mask = ~torch.isnan(y_train)
        if mask.sum() > 0:
            loss = F.mse_loss(out[mask], y_train_clean[mask])
        else:
            loss = F.mse_loss(out, y_train_clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    net.eval()
    with torch.no_grad():
        out_test = net(X_test_clean)
        mask = ~torch.isnan(y_test)
        if mask.sum() > 0:
            test_mse = F.mse_loss(out_test[mask], y_test_clean[mask]).item()
        else:
            test_mse = F.mse_loss(out_test, y_test_clean).item()

    n_total_params = sum(p.numel() for p in net.parameters())

    return {
        "cond": cond,
        "dataset": dataset,
        "seed": seed,
        "epochs": epochs,
        "final_loss": losses[-1],
        "test_mse": test_mse,
        "n_total_params": n_total_params,
    }


CONDITIONS = [
    "cfc",
    "bidi_concat",
    "bidi_weighted",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="Quick smoke run")
    parser.add_argument("--output", type=str, default="results/bench_bidirectional_cfc.json")
    args = parser.parse_args()

    if args.smoke:
        epochs = 1
        seeds = 1
    else:
        epochs = args.epochs
        seeds = args.seeds

    results = []
    cells = list(DATASETS.keys())
    conds = CONDITIONS
    total = len(cells) * len(conds) * seeds
    print(f"=== Round 144 Bidi-CfC bench: {len(cells)} datasets × {len(conds)} conds × {seeds} seeds = {total} cells, {epochs} epochs ===")
    t0 = time.time()
    for dataset in cells:
        for cond in conds:
            for seed in range(seeds):
                r = train_one(cond, dataset, seed, epochs=epochs)
                results.append(r)
                elapsed = time.time() - t0
                params = f"params={r['n_total_params']}"
                print(
                    f"  [{len(results)}/{total}] {dataset}/{cond}/s{seed} "
                    f"-> test_mse={r['test_mse']:.4f} loss={r['final_loss']:.4f} "
                    f"{params} ({elapsed:.0f}s)"
                )

    summary = {}
    for r in results:
        key = (r["cond"], r["dataset"])
        if key not in summary:
            summary[key] = {"test_mse": [], "final_loss": [], "n_total_params": []}
        summary[key]["test_mse"].append(r["test_mse"])
        summary[key]["final_loss"].append(r["final_loss"])
        summary[key]["n_total_params"].append(r["n_total_params"])

    print("\n=== Summary (test_mse mean ± std) ===")
    print(f"{'cond':<14} | " + " | ".join(f"{d:<14}" for d in cells))
    for cond in conds:
        row = [cond]
        for dataset in cells:
            vals = summary[(cond, dataset)]["test_mse"]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            row.append(f"{mean:.4f}±{std:.4f}")
        print(" | ".join(row))

    print("\n=== Parameter count per cond ===")
    for cond in conds:
        params = summary[(cond, cells[0])]["n_total_params"]
        print(f"  {cond:<14} n_params={params[0]}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": results, "summary": {f"{k[0]}|{k[1]}": v for k, v in summary.items()}}, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
