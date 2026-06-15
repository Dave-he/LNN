"""Round 110 — Frequency Experts bench (PRD #10-72).

Compares MoFE-Time-style frequency experts vs MLP experts on 3 datasets.

Cells: 3 datasets × 4 conditions × 2 seeds = 24 cells
Conditions:
  - baseline_mlp:   standard MLP expert (control)
  - freq_fixed:     fixed (non-learnable) frequencies
  - freq_learned:   MoFE-Time learnable frequencies
  - freq_no_time:   freq_learned but no time-domain branch
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.freq_experts import (
    FrequencyExpert,
    FrequencyExpertConfig,
    FrequencyMoEConfig,
    TimeFreqMoECfCNetwork,
)


# ---------------------------------------------------------------------------
# Dataset generators (same as rounds 102-109)
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
# MLP baseline (control)
# ---------------------------------------------------------------------------


class MLPMoECell(nn.Module):
    """Standard MLP MoE for baseline comparison."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1,
                 n_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            for _ in range(n_experts)
        ])
        self.router = nn.Linear(input_size, n_experts)
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        x_clean = torch.nan_to_num(x, nan=0.0)
        B, T, D = x_clean.shape
        x_flat = x_clean.reshape(-1, D)
        logit = self.router(x_flat)
        weights = F.softmax(logit, dim=-1)
        top_v, top_idx = weights.topk(self.top_k, dim=-1)
        top_w = F.softmax(top_v, dim=-1)

        expert_outs = torch.stack([e(x_flat) for e in self.experts], dim=0)  # (K, B*T, H)
        expert_outs = expert_outs.permute(1, 0, 2)  # (B*T, K, H)
        idx_expanded = top_idx.unsqueeze(-1).expand(-1, -1, expert_outs.size(-1))
        top_outs = torch.gather(expert_outs, 1, idx_expanded)  # (B*T, top_k, H)
        mixed = (top_w.unsqueeze(-1) * top_outs).sum(dim=1)  # (B*T, H)
        out = self.output_proj(mixed).reshape(B, T, -1)
        return out, top_idx, top_w


class MLPMoENetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1,
                 n_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.cell = MLPMoECell(input_size, hidden_size, output_size, n_experts, top_k)
        self.last_top_w = None
        self.last_top_idx = None

    def forward(self, x):
        out, top_idx, top_w = self.cell(x)
        self.last_top_w = top_w
        self.last_top_idx = top_idx
        return out

    def get_utilization(self):
        return {"routing_H": 0.5, "max_min": 1.0, "active_fraction": 1.0}


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def make_net(cond: str, input_size: int = 2, hidden_size: int = 16, output_size: int = 1):
    if cond == "baseline_mlp":
        return MLPMoENetwork(input_size, hidden_size, output_size, n_experts=4, top_k=2)
    elif cond == "freq_fixed":
        # Use freq experts but freeze the omega
        net = TimeFreqMoECfCNetwork(
            input_size, hidden_size, output_size,
            config=FrequencyMoEConfig(
                n_experts=4, top_k=2, n_freqs=4, use_complex_basis=True,
                use_time_branch=True,
            ),
        )
        for e in net.cell.experts:
            e.omega_raw.requires_grad_(False)
        return net
    elif cond == "freq_learned":
        return TimeFreqMoECfCNetwork(
            input_size, hidden_size, output_size,
            config=FrequencyMoEConfig(
                n_experts=4, top_k=2, n_freqs=4, use_complex_basis=True,
                use_time_branch=True,
            ),
        )
    elif cond == "freq_no_time":
        return TimeFreqMoECfCNetwork(
            input_size, hidden_size, output_size,
            config=FrequencyMoEConfig(
                n_experts=4, top_k=2, n_freqs=4, use_complex_basis=True,
                use_time_branch=False,
            ),
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
        if cond == "baseline_mlp":
            out = net(X_train)
            aux = torch.zeros(())
        else:
            out, aux, _ = net(X_train)
        mask = ~torch.isnan(y_train)
        if mask.sum() > 0:
            loss = F.mse_loss(out[mask], y_train_clean[mask]) + 0.01 * aux
        else:
            loss = F.mse_loss(out, y_train_clean) + 0.01 * aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    # Eval
    net.eval()
    with torch.no_grad():
        if cond == "baseline_mlp":
            out_test = net(X_test)
        else:
            out_test, _, _ = net(X_test)
        mask = ~torch.isnan(y_test)
        if mask.sum() > 0:
            test_mse = F.mse_loss(out_test[mask], y_test_clean[mask]).item()
        else:
            test_mse = F.mse_loss(out_test, y_test_clean).item()

    util = net.get_utilization()

    # Get learned omegas for freq conditions
    if cond.startswith("freq"):
        omegas = net.get_omegas().cpu().tolist()
    else:
        omegas = []

    return {
        "cond": cond,
        "dataset": dataset,
        "seed": seed,
        "epochs": epochs,
        "final_loss": losses[-1],
        "test_mse": test_mse,
        "routing_H": util.get("routing_H", 0.0),
        "active_fraction": util.get("active_fraction", 0.0),
        "max_min": util.get("max_min", 1.0),
        "omegas": omegas,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", type=str, default="results/bench_freq_experts.json")
    args = ap.parse_args()

    conds = ["baseline_mlp", "freq_fixed", "freq_learned", "freq_no_time"]
    datasets = ["sin_irr", "structured_irr", "random_irr"]

    results: List[Dict] = []
    t0 = time.time()
    for cond in conds:
        for ds in datasets:
            for seed in range(args.seeds):
                r = train_one(cond, ds, seed, epochs=args.epochs)
                results.append(r)
                elapsed = time.time() - t0
                print(f"[{elapsed:6.1f}s] {cond:>14} {ds:>14} seed={seed} "
                      f"test_mse={r['test_mse']:.4f} H={r['routing_H']:.3f}")
    total = time.time() - t0
    print(f"\n=== Done {len(results)} cells in {total:.1f}s ===\n")

    print(f"{'cond':>14} {'dataset':>14} {'test_mse':>10} {'H':>6} {'active':>7}")
    for r in results:
        print(f"{r['cond']:>14} {r['dataset']:>14} {r['test_mse']:>10.4f} "
              f"{r['routing_H']:>6.3f} {r['active_fraction']:>7.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "epochs": args.epochs, "seeds": args.seeds}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
