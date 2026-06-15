"""Round 127 — K_r (routed expert count) sweep on triple hybrid K_s=2.

Tests whether the natural counterpart to round 125's K_s sweep
helps: more routed experts (K_r ∈ {2, 3, 4, 6}) on the triple
hybrid with K_s=2 (round 125 best for structured). The K_r
dimension is the natural symmetry of K_s — we tested shared
multiplicity but NOT routed multiplicity.

Hypothesis: K_r=4 or K_r=6 might help structured further, since
the LoRA-DAG-Shared's routed experts can capture more diverse
patterns with more capacity.

Cells: 3 datasets × 5 conditions × 2 seeds = 30 cells
Conditions (all K_s=2):
  - lora_dag_shared_kr2_ks2: K_r=2
  - lora_dag_shared_kr3_ks2: K_r=3 (round 125 baseline)
  - lora_dag_shared_kr4_ks2: K_r=4
  - lora_dag_shared_kr6_ks2: K_r=6
  - baseline_cfc:             standard CfC (control)
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
from lnn.core.lora_dag_shared_moe import (
    LoRADAGSharedMoECfCNetwork,
    lora_dag_shared_moe_utilization,
)


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
    if cond == "baseline_cfc":
        return CfCNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=2, return_sequences=True,
        )
    elif cond == "kr2_ks2":
        return LoRADAGSharedMoECfCNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=2, return_sequences=True,
            n_experts=2, top_k=2, rank=4, alpha=1.0,
            n_dag_iterations=1, router_type="learned",
            use_shared=True, n_shared=2,
        )
    elif cond == "kr3_ks2":
        return LoRADAGSharedMoECfCNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=2, return_sequences=True,
            n_experts=3, top_k=3, rank=4, alpha=1.0,
            n_dag_iterations=1, router_type="learned",
            use_shared=True, n_shared=2,
        )
    elif cond == "kr4_ks2":
        return LoRADAGSharedMoECfCNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=2, return_sequences=True,
            n_experts=4, top_k=4, rank=4, alpha=1.0,
            n_dag_iterations=1, router_type="learned",
            use_shared=True, n_shared=2,
        )
    elif cond == "kr6_ks2":
        return LoRADAGSharedMoECfCNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=2, return_sequences=True,
            n_experts=6, top_k=6, rank=4, alpha=1.0,
            n_dag_iterations=1, router_type="learned",
            use_shared=True, n_shared=2,
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

    losses = []
    for _ in range(epochs):
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

    net.eval()
    with torch.no_grad():
        out_test = net(X_test)
        mask = ~torch.isnan(y_test)
        if mask.sum() > 0:
            test_mse = F.mse_loss(out_test[mask], y_test_clean[mask]).item()
        else:
            test_mse = F.mse_loss(out_test, y_test_clean).item()

    routing_entropy = 0.0
    n_total_params = sum(p.numel() for p in net.parameters())

    with torch.no_grad():
        _ = net(X_train)
    if cond.startswith("kr") and hasattr(net, "cells"):
        for cell in net.cells:
            if hasattr(cell, "last_expert_util") and cell.last_expert_util is not None:
                diag = lora_dag_shared_moe_utilization(cell)
                routing_entropy = float(diag["routing_entropy"])
                break

    return {
        "cond": cond,
        "dataset": dataset,
        "seed": seed,
        "epochs": epochs,
        "final_loss": losses[-1],
        "test_mse": test_mse,
        "routing_entropy": routing_entropy,
        "n_total_params": n_total_params,
    }


CONDITIONS = [
    "baseline_cfc",
    "kr2_ks2",
    "kr3_ks2",
    "kr4_ks2",
    "kr6_ks2",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke run (1 epoch)")
    parser.add_argument("--output", type=str,
                        default="results/bench_kr_sweep.json")
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
    print(f"=== Round 127 K_r sweep: {len(cells)} datasets × {len(conds)} conds × {seeds} seeds = {total} cells, {epochs} epochs ===")
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
                    f"H={r['routing_entropy']:.3f} {params} ({elapsed:.0f}s)"
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
    print(f"{'cond':<18} | " + " | ".join(f"{d:<14}" for d in cells))
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
        print(f"  {cond:<18} n_params={params[0]}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": results, "summary": {f"{k[0]}|{k[1]}": v for k, v in summary.items()}}, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
