"""Round 126 — Mixture-of-Recursions (MoR) bench for CfC.

Tests whether per-timestep variable recursion depth helps the
triple hybrid (LoRA-DAG-Shared, round 124) and standalone CfC.
The MoR idea (arXiv:2507.10524): each token (here, each timestep)
gets a router-predicted recursion depth ∈ {1, 2, ..., max_depth}.
We use continuous relaxation: h_new = sum_d w_d * h_d where w_d is
the softmax router weight.

Cells: 3 datasets × 9 conditions × 2 seeds = 54 cells
Conditions:
  - baseline_cfc:           standard CfC (control)
  - mor_d1:                 MoR with max_depth=1 (warm-start regression check)
  - mor_d2:                 MoR with max_depth=2
  - mor_d3:                 MoR with max_depth=3
  - mor_d4:                 MoR with max_depth=4
  - lora_dag_shared_ks1:    round 124 best sin config (control)
  - lora_dag_shared_ks2:    round 125 best structured config (control)
  - mor_d3_lora_dag_ks1:    MoR max_depth=3 on top of LoRA-DAG-Shared K_s=1
  - mor_d3_lora_dag_ks2:    MoR max_depth=3 on top of LoRA-DAG-Shared K_s=2
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
from lnn.core.lora_dag_shared_moe import LoRADAGSharedMoECfCNetwork
from lnn.core.mor_cfc import MoRCfCNetwork, mor_router_summary


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
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=2, return_sequences=True,
        )
    elif cond == "mor_d1":
        return MoRCfCNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=2, return_sequences=True, max_depth=1,
        )
    elif cond == "mor_d2":
        return MoRCfCNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=2, return_sequences=True, max_depth=2,
        )
    elif cond == "mor_d3":
        return MoRCfCNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=2, return_sequences=True, max_depth=3,
        )
    elif cond == "mor_d4":
        return MoRCfCNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=2, return_sequences=True, max_depth=4,
        )
    elif cond == "lora_dag_shared_ks1":
        return LoRADAGSharedMoECfCNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=2, return_sequences=True,
            n_experts=3, top_k=3, rank=4, alpha=1.0,
            n_dag_iterations=1, router_type="learned",
            use_shared=True, n_shared=1,
        )
    elif cond == "lora_dag_shared_ks2":
        return LoRADAGSharedMoECfCNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=2, return_sequences=True,
            n_experts=3, top_k=3, rank=4, alpha=1.0,
            n_dag_iterations=1, router_type="learned",
            use_shared=True, n_shared=2,
        )
    elif cond == "mor_d3_lora_dag_ks1":
        # The "5-axis hybrid" — would require a new cell. For now, use MoR-only
        # as a stand-in: replace shared pathway with MoR d=3
        # We test the orthogonal "5-axis" by stacking: layer 1 = MoR d=3,
        # layer 2 = LoRA-DAG-Shared K_s=1.
        # This is a simpler test of orthogonality.
        return _make_5axis(input_size, hidden_size, output_size, max_depth=3, n_shared=1)
    elif cond == "mor_d3_lora_dag_ks2":
        return _make_5axis(input_size, hidden_size, output_size, max_depth=3, n_shared=2)
    else:
        raise ValueError(cond)


class FiveAxisNetwork(torch.nn.Module):
    """Sequential: MoR d=3 layer 1, LoRA-DAG-Shared layer 2.

    Tests whether the 5th orthogonal dimension (recursion depth) composes
    with the 4-axis triple hybrid.
    """

    def __init__(self, input_size, hidden_size, output_size, max_depth, n_shared):
        super().__init__()
        self.mor = MoRCfCNetwork(
            input_size=input_size, hidden_size=hidden_size, output_size=hidden_size,
            num_layers=1, return_sequences=True, max_depth=max_depth,
        )
        self.lora_dag = LoRADAGSharedMoECfCNetwork(
            input_size=hidden_size, hidden_size=hidden_size, output_size=output_size,
            num_layers=1, return_sequences=True,
            n_experts=3, top_k=3, rank=4, alpha=1.0,
            n_dag_iterations=1, router_type="learned",
            use_shared=True, n_shared=n_shared,
        )

    def forward(self, x, dt=None, mask=None):
        x = torch.nan_to_num(x, nan=0.0)
        h_mor = self.mor(x, dt=dt)  # [B, T, H]
        return self.lora_dag(h_mor, dt=dt)


def _make_5axis(input_size, hidden_size, output_size, max_depth, n_shared):
    return FiveAxisNetwork(input_size, hidden_size, output_size, max_depth, n_shared)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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

    # Eval
    net.eval()
    with torch.no_grad():
        out_test = net(X_test)
        mask = ~torch.isnan(y_test)
        if mask.sum() > 0:
            test_mse = F.mse_loss(out_test[mask], y_test_clean[mask]).item()
        else:
            test_mse = F.mse_loss(out_test, y_test_clean).item()

    # MoR router diagnostic
    mor_depth = []
    n_total_params = sum(p.numel() for p in net.parameters())

    if cond.startswith("mor_d") or cond.startswith("mor_d3_lora"):
        with torch.no_grad():
            _ = net(X_train)
        if hasattr(net, "cells"):  # standalone MoR
            for cell in net.cells:
                s = mor_router_summary(cell)
                if s["mean_depth_weights"]:
                    mor_depth.append(s["mean_depth_weights"])
        elif hasattr(net, "mor"):  # 5-axis
            for cell in net.mor.cells:
                s = mor_router_summary(cell)
                if s["mean_depth_weights"]:
                    mor_depth.append(s["mean_depth_weights"])

    return {
        "cond": cond,
        "dataset": dataset,
        "seed": seed,
        "epochs": epochs,
        "final_loss": losses[-1],
        "test_mse": test_mse,
        "n_total_params": n_total_params,
        "mor_router_depth_weights": mor_depth,
    }


CONDITIONS = [
    "baseline_cfc",
    "mor_d1",
    "mor_d2",
    "mor_d3",
    "mor_d4",
    "lora_dag_shared_ks1",
    "lora_dag_shared_ks2",
    "mor_d3_lora_dag_ks1",
    "mor_d3_lora_dag_ks2",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke run (1 epoch)")
    parser.add_argument("--output", type=str,
                        default="results/bench_mor_cfc.json")
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
    print(f"=== Round 126 MoR bench: {len(cells)} datasets × {len(conds)} conds × {seeds} seeds = {total} cells, {epochs} epochs ===")
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

    # Aggregate
    summary = {}
    for r in results:
        key = (r["cond"], r["dataset"])
        if key not in summary:
            summary[key] = {"test_mse": [], "final_loss": [], "n_total_params": []}
        summary[key]["test_mse"].append(r["test_mse"])
        summary[key]["final_loss"].append(r["final_loss"])
        summary[key]["n_total_params"].append(r["n_total_params"])

    print("\n=== Summary (test_mse mean ± std) ===")
    print(f"{'cond':<28} | " + " | ".join(f"{d:<14}" for d in cells))
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
        print(f"  {cond:<28} n_params={params[0]}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": results, "summary": {f"{k[0]}|{k[1]}": v for k, v in summary.items()}}, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
