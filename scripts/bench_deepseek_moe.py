"""Round 113 — DeepSeekMoE Shared Expert Isolation bench (PRD #10-75).

Compares DeepSeekMoE (shared + routed experts, additive residual) vs
plain CfC and FAME on 3 datasets.  Tests the audit pattern that
mechanisms that modify the recurrent state mixing are dangerous, and
that an additive residual (shared path) is the natural safe way to
add MoE diversity.

Cells: 3 datasets × 5 conditions × 2 seeds = 30 cells
Conditions:
  - baseline_cfc:  standard CfC, no MoE (control)
  - fame_k3_t1:    FAME K=3 top_k=1 (round 78, sparse token-choice)
  - deepseek_1s_3r_t1: DeepSeek 1 shared + 3 routed, top_k=1
  - deepseek_1s_3r_t2: DeepSeek 1 shared + 3 routed, top_k=2
  - deepseek_2s_3r_t2: DeepSeek 2 shared + 3 routed, top_k=2
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.cfc import CfCNetwork
from lnn.core.deepseek_moe import DeepSeekCfCNetwork, deepseek_utilization
from lnn.core.fame_cfc import FAMECfCNetwork


# ---------------------------------------------------------------------------
# Dataset generators (same as rounds 102-112)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Net factory
# ---------------------------------------------------------------------------


def make_net(cond: str, input_size: int = 2, hidden_size: int = 16, output_size: int = 1):
    if cond == "baseline_cfc":
        return CfCNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=2,
            return_sequences=True,
        )
    elif cond == "fame_k3_t1":
        return FAMECfCNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=2,
            return_sequences=True,
            n_experts=3,
            top_k=1,
        )
    elif cond == "deepseek_1s_3r_t1":
        return DeepSeekCfCNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=2,
            return_sequences=True,
            n_shared=1,
            n_routed=3,
            top_k=1,
        )
    elif cond == "deepseek_1s_3r_t2":
        return DeepSeekCfCNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=2,
            return_sequences=True,
            n_shared=1,
            n_routed=3,
            top_k=2,
        )
    elif cond == "deepseek_2s_3r_t2":
        return DeepSeekCfCNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=2,
            return_sequences=True,
            n_shared=2,
            n_routed=3,
            top_k=2,
        )
    else:
        raise ValueError(cond)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one(cond: str, dataset: str, seed: int, epochs: int = 50,
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

    losses = []
    for _epoch in range(epochs):
        opt.zero_grad()
        out = net(X_train)
        mask = ~torch.isnan(y_train)
        if mask.sum() > 0:
            loss = F.mse_loss(out[mask], y_train_clean[mask])
        else:
            loss = F.mse_loss(out, y_train_clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    # Eval
    net.eval()
    with torch.no_grad():
        out_test = net(X_test)
        mask = ~torch.isnan(y_test)
        if mask.sum() > 0:
            test_mse = F.mse_loss(out_test[mask], y_test_clean[mask]).item()
        else:
            test_mse = F.mse_loss(out_test, y_test_clean).item()

    # DeepSeek utilization diagnostic
    shared_util = 0.0
    routed_util_mean = 0.0
    if cond.startswith("deepseek_"):
        n_shared = 0
        n_routed = 0
        if cond == "deepseek_1s_3r_t1" or cond == "deepseek_1s_3r_t2":
            n_shared, n_routed = 1, 3
        elif cond == "deepseek_2s_3r_t2":
            n_shared, n_routed = 2, 3
        # Run a forward pass to populate last_shared_util / last_g
        with torch.no_grad():
            _ = net(X_train)
        # Inspect the last cell of the last layer
        last_cell = net.cells[-1]
        diag = deepseek_utilization(last_cell)
        shared_util = float(diag["shared_util"].mean().item()) if diag["shared_util"].numel() > 0 else 0.0
        routed_util_mean = float(diag["routed_util"].mean().item()) if diag["routed_util"].numel() > 0 else 0.0

    return {
        "cond": cond,
        "dataset": dataset,
        "seed": seed,
        "epochs": epochs,
        "final_loss": losses[-1],
        "test_mse": test_mse,
        "shared_util": shared_util,
        "routed_util_mean": routed_util_mean,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


CONDITIONS = [
    "baseline_cfc",
    "fame_k3_t1",
    "deepseek_1s_3r_t1",
    "deepseek_1s_3r_t2",
    "deepseek_2s_3r_t2",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke run (1 epoch)")
    parser.add_argument("--output", type=str,
                        default="results/bench_deepseek_moe.json")
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
    print(f"=== Round 113 DeepSeekMoE bench: {len(cells)} datasets × {len(conds)} conds × {seeds} seeds = {total} cells, {epochs} epochs ===")
    t0 = time.time()
    for dataset in cells:
        for cond in conds:
            for seed in range(seeds):
                r = train_one(cond, dataset, seed, epochs=epochs)
                results.append(r)
                elapsed = time.time() - t0
                print(
                    f"  [{len(results)}/{total}] {dataset}/{cond}/s{seed} "
                    f"-> test_mse={r['test_mse']:.4f} loss={r['final_loss']:.4f} "
                    f"shared={r['shared_util']:.2f} routed={r['routed_util_mean']:.3f} ({elapsed:.0f}s)"
                )

    # Aggregate by (cond, dataset)
    summary = {}
    for r in results:
        key = (r["cond"], r["dataset"])
        if key not in summary:
            summary[key] = {"test_mse": [], "final_loss": []}
        summary[key]["test_mse"].append(r["test_mse"])
        summary[key]["final_loss"].append(r["final_loss"])

    # Pretty print
    print("\n=== Summary (test_mse mean ± std) ===")
    print(f"{'cond':<22} | " + " | ".join(f"{d:<14}" for d in cells))
    for cond in conds:
        row = [cond]
        for dataset in cells:
            vals = summary[(cond, dataset)]["test_mse"]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            row.append(f"{mean:.4f}±{std:.4f}")
        print(" | ".join(f"{c:<14}" if i == 0 else f"{c:<22}" if False else c for i, c in enumerate(row)))

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": results, "summary": {f"{k[0]}|{k[1]}": v for k, v in summary.items()}}, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
