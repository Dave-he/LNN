"""Smoke bench for per-expert gradient magnitude (PRD #10-50, 2026-06-15, round 88).

Compare per-expert gradient H vs per-expert empirical H (utilization
EMA) on the same hard cases used in round 84-87:

- 2 conditions × 3 datasets × 3 orth λ ∈ {0.1, 1.0, 10.0}
- 2 H modes: per-expert empirical (utilization) vs per-expert gradient

Per cell we report:
- E_agg_emp: aggregated empirical E (round 83/87)
- E_agg_grad: aggregated gradient E (round 87)
- per_expert_grad: [K] tensor of per-expert gradient norms
- per_expert_util: [K] utilization EMA values
- dead_by_util: count of dead experts by utilization (< 1% threshold)
- dead_by_grad: count of dead experts by gradient (< 1e-6 threshold)
- max_min_ratio_grad: spread of per-expert gradient magnitudes
- max_min_ratio_util: spread of per-expert utilization

Hypotheses:
- H1 (per-expert H_grad detects dead experts): per-expert H_grad is
  0 for truly dead experts, even when aggregated H_grad is high
- H2 (per-expert H_grad and utilization can disagree on WHICH
  experts are dead): gradient and utilization are different signals
  and may flag different experts

We test on 3 synthetic datasets (toy_sin, random, structured).

Run:
    .venv312/bin/python scripts/bench_per_expert_gradient.py --quick
    .venv312/bin/python scripts/bench_per_expert_gradient.py            # full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lnn.core.fame_cfc import FAMECfCCell
from lnn.core.orthogonality import orthogonality_loss


# ---------------------------------------------------------------------------
# Synthetic datasets (same as round 83/84/85/86/87)
# ---------------------------------------------------------------------------

def make_sin_dataset(n_samples: int = 32, seq_len: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    t = np.linspace(0, 4 * np.pi, seq_len + 1).astype(np.float32)
    x_np = np.sin(t[1:])[None, :].repeat(n_samples, axis=0)
    y_np = np.sin(t[1:] + 0.1)[:, None].repeat(n_samples, axis=0).T
    x = torch.tensor(x_np).unsqueeze(-1)
    y = torch.tensor(y_np).unsqueeze(-1)
    x = x + 0.05 * torch.randn_like(x)
    return x, y


def make_random_dataset(n_samples: int = 32, seq_len: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.standard_normal((n_samples, seq_len, 1)).astype(np.float32))
    y = torch.tensor(rng.standard_normal((n_samples, seq_len, 1)).astype(np.float32))
    return x, y


def make_structured_dataset(n_samples: int = 32, seq_len: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, seq_len).astype(np.float32)
    x_np = (np.sin(2.0 * t) + 2.0 * np.cos(0.5 * t))[None, :].repeat(n_samples, axis=0)
    y_np = (np.sin(2.0 * t + 0.1) + 2.0 * np.cos(0.5 * t + 0.05))[None, :].repeat(n_samples, axis=0)
    x = torch.tensor(x_np).unsqueeze(-1) + 0.3 * torch.randn(n_samples, seq_len, 1)
    y = torch.tensor(y_np).unsqueeze(-1) + 0.3 * torch.randn(n_samples, seq_len, 1)
    return x, y


DATASETS = {
    "toy_sin": make_sin_dataset,
    "random": make_random_dataset,
    "structured": make_structured_dataset,
}


# ---------------------------------------------------------------------------
# Training (per condition)
# ---------------------------------------------------------------------------

def train_one_with_per_expert(
    cell: FAMECfCCell,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    orth_lambda: float,
    lr: float = 1e-2,
) -> dict:
    """Train cell and collect per-expert E at each epoch."""
    params = [p for p in cell.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    cell.train()
    history = {
        "loss": [],
        "per_expert_grad": [],  # [K] list of lists
        "per_expert_util": [],  # [K] list of lists
        "dead_by_grad": [],     # int per epoch
        "dead_by_util": [],     # int per epoch
    }
    for epoch in range(epochs):
        opt.zero_grad()
        h = torch.zeros(x.shape[0], cell.hidden_size)
        task_loss_acc = 0.0
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            y_t = y[:, t, :]
            h_new, outs = cell.forward_with_aux(x_t, h, dt=1.0)
            task_loss_t = ((h_new - y_t) ** 2).mean()
            task_loss_acc = task_loss_acc + task_loss_t
            h = h_new
        task_loss = task_loss_acc / x.shape[1]
        # Compute orth loss (no rescaling — we want raw gradient signal).
        h_new, outs = cell.forward_with_aux(x[:, -1, :], h, dt=1.0)
        orth_loss = orthogonality_loss(outs, lambda_coeff=orth_lambda) if orth_lambda > 0 else torch.tensor(0.0)
        total_loss = task_loss + orth_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        history["loss"].append(float(total_loss.item()))
        # Per-expert gradient diagnostic (round 88).
        # Recompute forward to get fresh logits (after step).
        cell.train()
        h2 = torch.zeros(x.shape[0], cell.hidden_size)
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            h2, _ = cell.forward_with_aux(x_t, h2, dt=1.0)
        # Build a fresh task_loss that depends on the new routing.
        h2_new, _ = cell.forward_with_aux(x[:, -1, :], h2, dt=1.0)
        fresh_task_loss = (h2_new ** 2).mean() * 0.0 + 1.0  # trivial loss
        # Use a real task_loss (with grad to experts) — for simplicity,
        # we use the *post-step* h2_new vs a target.
        target = torch.zeros_like(h2_new)
        fresh_task_loss = ((h2_new - target) ** 2).mean()
        diag = cell.moe_ecology_diagnostic(
            B=orth_lambda, task_loss=fresh_task_loss, per_expert=True,
        )
        history["per_expert_grad"].append(diag["per_expert_grad_list"])
        history["dead_by_grad"].append(diag["dead_by_grad"])
        # Utilization (per-expert empirical) — from same diag.
        history["per_expert_util"].append(diag["utilization"])
        history["dead_by_util"].append(diag["dead_experts"])
    return history


def cell_factory(orth_lambda: float, lambda_safe: float = 0.001):
    """Build cell (no ecology gates, so we can isolate per-expert H)."""
    return FAMECfCCell(
        input_size=1, hidden_size=8, n_experts=3, top_k=1,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results/bench_per_expert_gradient.json")
    args = p.parse_args()
    epochs = 2 if args.quick else 5
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out = {"epochs": epochs, "wall_time_s": 0.0, "runs": []}
    for orth_lambda in (0.1, 1.0, 10.0):
        run = {"orth_lambda": orth_lambda, "datasets": {}}
        for ds_name, ds_fn in DATASETS.items():
            x, y = ds_fn(n_samples=32, seq_len=16, seed=0)
            torch.manual_seed(0)
            cell = cell_factory(orth_lambda)
            h = train_one_with_per_expert(
                cell, x, y, epochs=epochs, orth_lambda=orth_lambda,
            )
            # Per-cell summary (last epoch).
            run["datasets"][ds_name] = {
                "loss_final": h["loss"][-1],
                "per_expert_grad_last": h["per_expert_grad"][-1],
                "per_expert_util_last": h["per_expert_util"][-1],
                "dead_by_grad_last": h["dead_by_grad"][-1],
                "dead_by_util_last": h["dead_by_util"][-1],
                "max_min_ratio_grad": max(h["per_expert_grad"][-1]) / (min(h["per_expert_grad"][-1]) + 1e-8),
                "max_min_ratio_util": max(h["per_expert_util"][-1]) / (min(h["per_expert_util"][-1]) + 1e-8),
            }
        # Pretty print.
        print(f"\n[λ={orth_lambda}]")
        for ds_name in DATASETS:
            r = run["datasets"][ds_name]
            print(
                f"  {ds_name:11s}: loss={r['loss_final']:.4f}  "
                f"per_grad={['%.2e' % g for g in r['per_expert_grad_last']]}  "
                f"per_util={['%.3f' % u for u in r['per_expert_util_last']]}  "
                f"dead_grad={r['dead_by_grad_last']}  dead_util={r['dead_by_util_last']}"
            )
        out["runs"].append(run)
    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
