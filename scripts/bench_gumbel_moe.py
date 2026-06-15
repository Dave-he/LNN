"""Round 117 — Gumbel-Softmax MoE bench (PRD #10-79).

Compares stochastic Gumbel-Softmax routing (Jang et al. 2017, ICLR 2017,
arXiv:1611.01144) with annealed temperature against plain CfC, FAME, and
Sigmoid MoE on 3 datasets.  Tests whether stochastic exploration during
training improves over deterministic routing.

Cells: 3 datasets × 5 conditions × 2 seeds = 30 cells
Conditions:
  - baseline_cfc:     standard CfC, no MoE (control)
  - fame_k3_t1:       FAME K=3 top_k=1 (round 78 baseline, softmax)
  - sigmoid_k3_dense: Sigmoid MoE K=3 dense (round 116 winner)
  - gumbel_k3_high:   Gumbel MoE K=3 T=1.0 constant (no annealing)
  - gumbel_k3_anneal: Gumbel MoE K=3 T annealed 1.0 → 0.1
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
from lnn.core.fame_cfc import FAMECfCNetwork
from lnn.core.gumbel_moe import GumbelMoECfCNetwork, gumbel_moe_utilization
from lnn.core.sigmoid_moe import SigmoidMoECfCNetwork


# ---------------------------------------------------------------------------
# Dataset generators (same as rounds 102-116)
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
    elif cond == "sigmoid_k3_dense":
        return SigmoidMoECfCNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=2,
            return_sequences=True,
            n_experts=3,
            top_k=0,  # dense
        )
    elif cond == "gumbel_k3_high":
        return GumbelMoECfCNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=2,
            return_sequences=True,
            n_experts=3,
            temperature=1.0,
            anneal_rate=1.0,  # no annealing
            min_temperature=1.0,
        )
    elif cond == "gumbel_k3_anneal":
        return GumbelMoECfCNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=2,
            return_sequences=True,
            n_experts=3,
            temperature=1.0,
            anneal_rate=0.95,
            min_temperature=0.1,
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
    for epoch in range(epochs):
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
        # Anneal Gumbel temperature after each epoch (if applicable)
        if cond.startswith("gumbel_"):
            net.anneal_step()

    # Eval
    net.eval()
    with torch.no_grad():
        out_test = net(X_test)
        mask = ~torch.isnan(y_test)
        if mask.sum() > 0:
            test_mse = F.mse_loss(out_test[mask], y_test_clean[mask]).item()
        else:
            test_mse = F.mse_loss(out_test, y_test_clean).item()

    # Gumbel MoE utilization diagnostic
    routing_entropy = 0.0
    expert_util = []
    final_temperature = 0.0
    if cond.startswith("gumbel_"):
        with torch.no_grad():
            _ = net(X_train)
        for cell in net.cells:
            if hasattr(cell, "last_expert_util") and cell.last_expert_util is not None:
                diag = gumbel_moe_utilization(cell)
                routing_entropy = float(diag["routing_entropy"].item())
                expert_util = diag["expert_util"].tolist()
                final_temperature = float(diag["temperature"])
                break  # use first layer's diagnostic

    return {
        "cond": cond,
        "dataset": dataset,
        "seed": seed,
        "epochs": epochs,
        "final_loss": losses[-1],
        "test_mse": test_mse,
        "routing_entropy": routing_entropy,
        "expert_util": expert_util,
        "final_temperature": final_temperature,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


CONDITIONS = [
    "baseline_cfc",
    "fame_k3_t1",
    "sigmoid_k3_dense",
    "gumbel_k3_high",
    "gumbel_k3_anneal",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke run (1 epoch)")
    parser.add_argument("--output", type=str,
                        default="results/bench_gumbel_moe.json")
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
    print(f"=== Round 117 Gumbel MoE bench: {len(cells)} datasets × {len(conds)} conds × {seeds} seeds = {total} cells, {epochs} epochs ===")
    t0 = time.time()
    for dataset in cells:
        for cond in conds:
            for seed in range(seeds):
                r = train_one(cond, dataset, seed, epochs=epochs)
                results.append(r)
                elapsed = time.time() - t0
                ent = f"H={r['routing_entropy']:.2f}" if r["routing_entropy"] > 0 else ""
                temp = f"T={r['final_temperature']:.2f}" if r["final_temperature"] > 0 else ""
                print(
                    f"  [{len(results)}/{total}] {dataset}/{cond}/s{seed} "
                    f"-> test_mse={r['test_mse']:.4f} loss={r['final_loss']:.4f} {ent} {temp} ({elapsed:.0f}s)"
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
        print(" | ".join(row))

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": results, "summary": {f"{k[0]}|{k[1]}": v for k, v in summary.items()}}, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
