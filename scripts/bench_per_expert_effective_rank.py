"""Round 95 bench (PRD #10-57) — per-expert effective rank.

Direct test of the FAME (arXiv:2606.08896) "diverse experts" claim
and the MR-MoE (arXiv:2606.12240) "multi-rate expert specialization"
claim, by measuring the **per-expert weight effective rank** after
training on three datasets of increasing structure:

- toy_sin:    smooth, predictable — moderate structure
- structured: regime-switching — high structure (FAME's ideal)
- random:     uniform noise — no structure (control)

For each (dataset × model × condition) cell we measure:
- per_expert_eff_rank: list of K floats
- diversity_ratio: max/min ratio (1.0 = uniform, >1.5 = diverse)
- expert_utilization: fraction of steps each expert is selected
                      (FAME only — uses last_g from forward)

Cells:
  3 datasets × 2 models (FAME / MR-MoE) × 2 conditions (trained / init)
  × 3 seeds = 36 cells

Outputs results/bench_per_expert_effective_rank.json

Run:
    .venv312/bin/python scripts/bench_per_expert_effective_rank.py --quick
    .venv312/bin/python scripts/bench_per_expert_effective_rank.py        # full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lnn.core.effective_rank import expert_diversity_summary
from lnn.core.fame_cfc import FAMECfCCell
from lnn.core.mr_moe_cfc import MRMoECfCCell


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def make_toy_sin(T: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """f(t) = sin(2π t) + 0.5 sin(10π t)."""
    torch.manual_seed(seed)
    t = torch.linspace(0, 1, T)
    y = torch.sin(2 * np.pi * t) + 0.5 * torch.sin(10 * np.pi * t)
    return t, y


def make_structured(T: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-regime signal: slow sine in [0, 0.5], fast sawtooth in [0.5, 1]."""
    torch.manual_seed(seed)
    t = torch.linspace(0, 1, T)
    y = torch.zeros_like(t)
    regime1 = t < 0.5
    y[regime1] = torch.sin(2 * np.pi * t[regime1])
    y[~regime1] = torch.sign(torch.sin(20 * np.pi * t[~regime1]))
    return t, y


def make_random(T: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure noise — control, no structure to learn."""
    torch.manual_seed(seed)
    t = torch.linspace(0, 1, T)
    y = torch.randn(T)
    return t, y


DATASETS = {
    "toy_sin": make_toy_sin,
    "structured": make_structured,
    "random": make_random,
}


# ---------------------------------------------------------------------------
# Cell trainers
# ---------------------------------------------------------------------------

def make_fame(K: int = 5, top_k: int = 2) -> FAMECfCCell:
    return FAMECfCCell(input_size=1, hidden_size=8, n_experts=K, top_k=top_k)


def make_mr_moe(K: int = 5) -> MRMoECfCCell:
    return MRMoECfCCell(input_size=1, hidden_size=8, n_experts=K)


FACTORIES = {
    "FAME": make_fame,
    "MR-MoE": make_mr_moe,
}


def train_cell(
    cell: nn.Module,
    t: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
) -> None:
    """Stateless training: at each step, hidden state h is reset to 0.

    This is the same setup as rounds 92-94 — the goal is to train
    the cell weights, not to test its recurrent memory.
    """
    opt = torch.optim.Adam(cell.parameters(), lr=lr)
    h0 = torch.zeros(1, 8)
    for _ in range(epochs):
        opt.zero_grad()
        h = h0
        outs = []
        for ti in t:
            x = ti.reshape(1, 1)
            h = cell(x, h, dt=1.0)
            outs.append(h)
        y_pred = torch.stack(outs, dim=0).squeeze(1).mean(dim=-1)
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()
        opt.step()


def utilization_from_fame(cell: FAMECfCCell, t: torch.Tensor) -> list[float]:
    """Compute expert utilization from FAME's last_g on a forward pass.

    FAME stores its routing weights in cell.last_g (top-K sparse).  We
    accumulate them across all steps and report the fraction of steps
    each expert was selected.
    """
    counts = [0] * cell.n_experts
    n_steps = 0
    h = torch.zeros(1, cell.hidden_size)
    with torch.no_grad():
        for ti in t:
            x = ti.reshape(1, 1)
            _ = cell(x, h, dt=1.0)
            if hasattr(cell, "last_g") and cell.last_g is not None:
                g = cell.last_g.detach().squeeze(0).cpu()
                # top-K sparse: only top_k entries are non-zero
                for k, gk in enumerate(g):
                    if gk > 0:
                        counts[k] += 1
            n_steps += 1
    if n_steps == 0:
        return [0.0] * cell.n_experts
    return [c / n_steps for c in counts]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out", default="results/bench_per_expert_effective_rank.json")
    args = p.parse_args()
    epochs = args.epochs if args.epochs is not None else (30 if args.quick else 100)
    n_seeds = args.seeds
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out: dict = {
        "epochs": epochs,
        "n_seeds": n_seeds,
        "wall_time_s": 0.0,
        "datasets": {},
    }

    T = 64

    for ds_name, ds_fn in DATASETS.items():
        ds_out: dict = {}
        for model_name, factory in FACTORIES.items():
            model_out: dict = {"trained": [], "init": []}
            for seed in range(n_seeds):
                # Trained cell.
                torch.manual_seed(seed)
                np.random.seed(seed)
                t, y = ds_fn(T, seed=seed)
                cell = factory()
                # Snapshot ranks at init BEFORE training.
                init_summary = expert_diversity_summary(cell)
                train_cell(cell, t, y, epochs=epochs, lr=1e-2)
                trained_summary = expert_diversity_summary(cell)
                utilization: list[float] | None = None
                if model_name == "FAME":
                    utilization = utilization_from_fame(cell, t)
                model_out["trained"].append({
                    "per_expert_eff_rank": trained_summary["per_expert"],
                    "diversity_ratio": trained_summary["diversity_ratio"],
                    "mean": trained_summary["mean"],
                    "min": trained_summary["min"],
                    "max": trained_summary["max"],
                    "std": trained_summary["std"],
                    "n_dead": trained_summary["n_dead"],
                    "utilization": utilization,
                })
                model_out["init"].append({
                    "per_expert_eff_rank": init_summary["per_expert"],
                    "diversity_ratio": init_summary["diversity_ratio"],
                    "mean": init_summary["mean"],
                    "min": init_summary["min"],
                    "max": init_summary["max"],
                    "std": init_summary["std"],
                })
            # Aggregate.
            def agg(field: str, condition: str) -> tuple[float, float]:
                vals = [s[field] for s in model_out[condition] if s.get(field) is not None and s[field] != float("inf")]
                if not vals:
                    return 0.0, 0.0
                return float(np.mean(vals)), float(np.std(vals))

            ds_out[model_name] = {
                "trained": {
                    "diversity_ratio_mean_std": agg("diversity_ratio", "trained"),
                    "mean_eff_rank_mean_std": agg("mean", "trained"),
                    "max_eff_rank_mean_std": agg("max", "trained"),
                    "min_eff_rank_mean_std": agg("min", "trained"),
                    "n_dead_mean_std": agg("n_dead", "trained"),
                    "per_seed": model_out["trained"],
                },
                "init": {
                    "diversity_ratio_mean_std": agg("diversity_ratio", "init"),
                    "mean_eff_rank_mean_std": agg("mean", "init"),
                    "max_eff_rank_mean_std": agg("max", "init"),
                    "min_eff_rank_mean_std": agg("min", "init"),
                    "per_seed": model_out["init"],
                },
            }
        out["datasets"][ds_name] = ds_out

    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))

    # Pretty print.
    print(f"\n=== Round 95 per-expert effective rank bench (epochs={epochs}, seeds={n_seeds}) ===\n")
    print(f"{'dataset':12s} | {'model':8s} | {'cond':8s} | {'div_ratio':>10s} | {'mean':>6s} | {'min':>6s} | {'max':>6s} | {'n_dead':>6s}")
    print("-" * 88)
    for ds_name in DATASETS:
        for model_name in FACTORIES:
            for cond in ("init", "trained"):
                cell = out["datasets"][ds_name][model_name][cond]
                dr_m, dr_s = cell["diversity_ratio_mean_std"]
                m_m, _ = cell["mean_eff_rank_mean_std"]
                mn_m, _ = cell.get("min_eff_rank_mean_std", (0.0, 0.0))
                mx_m, _ = cell.get("max_eff_rank_mean_std", (0.0, 0.0))
                nd_m = cell.get("n_dead_mean_std", (0.0, 0.0))[0]
                dr_str = f"{dr_m:5.2f}±{dr_s:4.2f}"
                print(f"{ds_name:12s} | {model_name:8s} | {cond:8s} | {dr_str:>10s} | {m_m:6.2f} | {mn_m:6.2f} | {mx_m:6.2f} | {nd_m:6.2f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
