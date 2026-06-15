"""Round 109 — Dynamic TMoE bench (PRD #10-71).

Compares dynamic expert pool vs fixed pool on 3 datasets.

Cells: 3 datasets × 4 conditions × 2 seeds = 24 cells (we'll do 12 for time)
Conditions:
  - baseline_fixed: K=4 fixed, no drift detection
  - dynamic_add:    init K=2, add on drift, max K=6, no prune
  - dynamic_full:   init K=2, add on drift, prune least-used, max K=6
  - dynamic_tiny:   init K=2, max K=3 (capacity-constrained)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.dynamic_tmoe import (
    DynamicExpertPoolConfig,
    DynamicTMoEConfig,
    DynamicTMoECfCNetwork,
    TemporalMemoryRouterConfig,
)


# ---------------------------------------------------------------------------
# Dataset generators (same as round 102-108)
# ---------------------------------------------------------------------------


def make_sin_irr(B: int, T: int, D: int = 2, missing_rate: float = 0.3) -> torch.Tensor:
    """Sin with per-channel phase, with random NaNs."""
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1)
    t = t.expand(B, T, 1)
    x = torch.zeros(B, T, D)
    for i in range(D):
        phase = i * 0.5
        x[:, :, i] = torch.sin(t.squeeze(-1) + phase)
    # Add NaN
    mask = torch.rand(B, T, D) < missing_rate
    x[mask] = float("nan")
    return x


def make_structured_irr(B: int, T: int, D: int = 2, missing_rate: float = 0.3) -> torch.Tensor:
    """Regime-switching sin: first half low-freq, second half high-freq."""
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
    x = torch.zeros(B, T, D)
    for i in range(D):
        regime = (torch.arange(T) >= T // 2).float()
        x[:, :, i] = torch.sin(t.squeeze(-1) * (1.0 + regime)) + i * 0.3
    mask = torch.rand(B, T, D) < missing_rate
    x[mask] = float("nan")
    return x


def make_random_irr(B: int, T: int, D: int = 2, missing_rate: float = 0.3) -> torch.Tensor:
    """Cumulative Gaussian noise with missing values."""
    x = torch.randn(B, T, D).cumsum(dim=1) * 0.1
    mask = torch.rand(B, T, D) < missing_rate
    x[mask] = float("nan")
    return x


def make_target(x: torch.Tensor) -> torch.Tensor:
    """Predict next-step value (B, T, 1) — sum of channels."""
    y = x[:, :, 0]  # (B, T)
    return y.unsqueeze(-1)


DATASETS = {
    "sin_irr": make_sin_irr,
    "structured_irr": make_structured_irr,
    "random_irr": make_random_irr,
}


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def make_net(cond: str, input_size: int = 2, hidden_size: int = 16, output_size: int = 1) -> DynamicTMoECfCNetwork:
    """Build net for a given condition. All dynamic conditions start at K=4
    (same as baseline) for a fair capacity comparison — we test the
    dynamic mechanism, not capacity."""
    if cond == "baseline_fixed":
        cfg = DynamicTMoEConfig(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            pool=DynamicExpertPoolConfig(init_size=4, max_size=4, min_size=4),
            router=TemporalMemoryRouterConfig(memory_dim=8, anomaly_dim=4, top_k=2),
            drift_threshold=10.0,  # disable drift detection
            prune_every=100000,  # disable pruning
        )
    elif cond == "dynamic_add":
        cfg = DynamicTMoEConfig(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            pool=DynamicExpertPoolConfig(init_size=4, max_size=8, min_size=4),
            router=TemporalMemoryRouterConfig(memory_dim=8, anomaly_dim=4, top_k=2),
            drift_threshold=0.05,
            prune_every=100000,  # no prune
        )
    elif cond == "dynamic_full":
        cfg = DynamicTMoEConfig(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            pool=DynamicExpertPoolConfig(init_size=4, max_size=8, min_size=4),
            router=TemporalMemoryRouterConfig(memory_dim=8, anomaly_dim=4, top_k=2),
            drift_threshold=0.05,
            prune_every=50,  # prune less aggressively
        )
    elif cond == "dynamic_tiny":
        cfg = DynamicTMoEConfig(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            pool=DynamicExpertPoolConfig(init_size=4, max_size=4, min_size=4),
            router=TemporalMemoryRouterConfig(memory_dim=8, anomaly_dim=4, top_k=2),
            drift_threshold=0.5,  # higher threshold → rarely fire
            prune_every=50,
        )
    else:
        raise ValueError(cond)
    return DynamicTMoECfCNetwork(input_size=input_size, hidden_size=hidden_size,
                                  output_size=output_size, config=cfg)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one(
    cond: str,
    dataset: str,
    seed: int,
    epochs: int = 50,
    B: int = 8,
    T: int = 32,
) -> Dict:
    """Train one cell, return metrics."""
    torch.manual_seed(seed)
    net = make_net(cond)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    X_train = DATASETS[dataset](B, T)
    y_train = make_target(X_train)
    X_test = DATASETS[dataset](B, T)
    y_test = make_target(X_test)

    # NaN-aware target: zero-fill
    y_train_clean = torch.nan_to_num(y_train, nan=0.0)
    y_test_clean = torch.nan_to_num(y_test, nan=0.0)

    losses = []
    drift_counts = []
    pool_sizes = []
    for epoch in range(epochs):
        opt.zero_grad()
        out, info = net(X_train, reset_state=(epoch == 0))
        # NaN-aware MSE
        mask = ~torch.isnan(y_train)
        if mask.sum() > 0:
            loss = F.mse_loss(out[mask], y_train_clean[mask])
        else:
            loss = F.mse_loss(out, y_train_clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        drift_counts.append(info["n_drifts"])
        pool_sizes.append(info["pool_size_final"])

    # Eval
    net.eval()
    with torch.no_grad():
        out_test, info_test = net(X_test, reset_state=True)
        mask = ~torch.isnan(y_test)
        if mask.sum() > 0:
            test_mse = F.mse_loss(out_test[mask], y_test_clean[mask]).item()
        else:
            test_mse = F.mse_loss(out_test, y_test_clean).item()

    # Get utilization
    util = net.get_utilization()

    return {
        "cond": cond,
        "dataset": dataset,
        "seed": seed,
        "epochs": epochs,
        "final_loss": losses[-1],
        "test_mse": test_mse,
        "n_drifts": info_test["n_drifts"],
        "pool_size_initial": info_test["pool_size_initial"],
        "pool_size_final": info_test["pool_size_final"],
        "n_adds": info_test["n_adds"],
        "n_prunes": info_test["n_prunes"],
        "routing_H": util["routing_H"],
        "active_fraction": util["active_fraction"],
        "max_min": util["max_min"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", type=str, default="results/bench_dynamic_tmoe.json")
    args = ap.parse_args()

    conds = ["baseline_fixed", "dynamic_add", "dynamic_full", "dynamic_tiny"]
    datasets = ["sin_irr", "structured_irr", "random_irr"]

    results: List[Dict] = []
    t0 = time.time()
    for cond in conds:
        for ds in datasets:
            for seed in range(args.seeds):
                r = train_one(cond, ds, seed, epochs=args.epochs)
                results.append(r)
                elapsed = time.time() - t0
                print(f"[{elapsed:6.1f}s] {cond:>16} {ds:>14} seed={seed} "
                      f"test_mse={r['test_mse']:.4f} pool={r['pool_size_initial']}→{r['pool_size_final']} "
                      f"drifts={r['n_drifts']} H={r['routing_H']:.3f}")
    total = time.time() - t0
    print(f"\n=== Done {len(results)} cells in {total:.1f}s ===\n")

    # Aggregate
    print(f"{'cond':>16} {'dataset':>14} {'test_mse':>10} {'pool_init':>10} {'pool_final':>10} "
          f"{'drifts':>7} {'H':>6} {'active':>7}")
    for r in results:
        print(f"{r['cond']:>16} {r['dataset']:>14} {r['test_mse']:>10.4f} "
              f"{r['pool_size_initial']:>10} {r['pool_size_final']:>10} "
              f"{r['n_drifts']:>7} {r['routing_H']:>6.3f} {r['active_fraction']:>7.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "epochs": args.epochs, "seeds": args.seeds}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
