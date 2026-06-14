"""Round 100 bench (PRD #10-62) — SNNL for expert disentanglement.

Direct test of arXiv:2603.26734 (Agarap & Azcarraga, March 2026) —
*Mixture of Experts with Soft Nearest Neighbor Loss: Resolving Expert
Collapse via Representation Disentanglement*.

We apply SNNL to the per-expert hidden state means of a small MoE
model (FAMECfC with K=4 experts) and measure whether it promotes
expert diversity at the feature level.

For each of 3 datasets (toy_sin, structured, random), we compare:
- baseline (no aux)
- +SNNL λ=0.001
- +orth λ=0.001 (round 80, activation orthogonality)
- +SNNL+orth combined

Cells: 1 model × 3 datasets × 4 conditions × 2 seeds = 24 cells

For each cell measure:
- task_loss
- pairwise_weight_similarity (round 90 metric)
- expert_diversity_ratio (round 95 metric)
- mean_eff_rank (round 94 metric)

H1: +SNNL has lower weight_similarity than baseline.
H2: +SNNL task loss within ±10%.
H3: combined SNNL+orth is the most diverse.

Run:
    .venv312/bin/python scripts/bench_snnl_expert_disentanglement.py --quick
    .venv312/bin/python scripts/bench_snnl_expert_disentanglement.py        # full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lnn.core.fame_cfc import FAMECfCCell
from lnn.core.moe_ecology import weight_space_overlap
from lnn.core.effective_rank import (
    per_expert_effective_rank,
    expert_diversity_ratio,
)
from lnn.core.orthogonality import orthogonality_loss
from lnn.core.snnl import soft_nearest_neighbor_loss


# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------

def target_fn(t: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * np.pi * t) + 0.5 * np.sin(10 * np.pi * t)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def make_toy_sin(T: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    t = torch.linspace(0, 1, T)
    y = target_fn(t)
    return t, y


def make_structured(T: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    t = torch.linspace(0, 1, T)
    y = torch.zeros_like(t)
    regime1 = t < 0.5
    y[regime1] = torch.sin(2 * np.pi * t[regime1])
    y[~regime1] = torch.sign(torch.sin(20 * np.pi * t[~regime1]))
    return t, y


def make_random(T: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
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
# FAME model wrapper
# ---------------------------------------------------------------------------

class FAMEModel(nn.Module):
    """FAMECfC with K experts and a head. Returns (y_pred, per_timestep_expert_features).

    per_timestep_expert_features: (T*K, d) tensor — flat collection of
    per-expert hidden states for every (timestep, expert) pair. Labels
    are the binned input regime for each timestep (positive vs negative
    target value).
    """

    def __init__(self, n_experts: int = 4) -> None:
        super().__init__()
        self.cell = FAMECfCCell(input_size=1, hidden_size=16, n_experts=n_experts)
        self.head = nn.Linear(16, 1)
        self.n_experts = n_experts

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # (T, K, H) per-timestep per-expert states
        per_ts = []
        outs = []
        h = torch.zeros(1, self.cell.hidden_size)
        for ti in t:
            x_t = ti.reshape(1, 1)
            h, expert_outs = self.cell.forward_with_aux(x_t, h, dt=1.0)
            outs.append(self.head(h))
            # Stack K expert outputs at this timestep
            per_ts.append(torch.stack([eo.squeeze(0) for eo in expert_outs], dim=0))
        y = torch.cat(outs, dim=-1).squeeze(0)
        per_timestep = torch.stack(per_ts, dim=0)  # (T, K, H)
        # Flatten to (T*K, H)
        features = per_timestep.reshape(-1, self.cell.hidden_size)
        # Labels: bin the input t into 2 classes (low vs high)
        # t is in [0, 1], so threshold at 0.5
        labels = (t > 0.5).long()  # (T,)
        # Expand labels to (T, K) then flatten
        labels = labels.unsqueeze(1).expand(-1, self.n_experts).reshape(-1)
        return y, features, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

SNNL_LAMBDA = 0.001
ORTH_LAMBDA = 0.001
TEMPERATURE = 0.5


def train_model(
    model: FAMEModel,
    t: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    use_snnl: bool,
    use_orth: bool,
) -> tuple[float, float, float, float]:
    """Train and return (task_loss, weight_sim, diversity_ratio, mean_eff_rank)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        y_pred, features, labels = model(t)
        task_loss = ((y_pred - y) ** 2).mean()
        total_loss = task_loss
        if use_snnl:
            # SNNL on per-(timestep, expert) features with regime labels
            snnl = soft_nearest_neighbor_loss(
                features, labels, temperature=TEMPERATURE,
            )
            total_loss = total_loss + SNNL_LAMBDA * snnl
        if use_orth:
            # Orthogonality between per-expert hidden states
            # Use the per-expert mean across timesteps (batch dim 0)
            per_expert_means = []
            for k in range(model.n_experts):
                per_expert_means.append(features[k::model.n_experts].mean(dim=0).unsqueeze(0))
            orth = orthogonality_loss(per_expert_means, lambda_coeff=ORTH_LAMBDA)
            total_loss = total_loss + orth
        total_loss.backward()
        opt.step()
    # Final measurement
    with torch.no_grad():
        y_pred, features, labels = model(t)
        final_task = float(((y_pred - y) ** 2).mean().item())
        # Weight overlap: collect first 2D weight matrix from each expert
        weights = []
        for k in range(model.n_experts):
            for name, p in model.cell.experts[k].named_parameters():
                if "weight" in name and p.dim() == 2:
                    weights.append(p)
                    break
        if weights:
            weight_sim = weight_space_overlap(weights)
        else:
            weight_sim = 1.0
        # Per-expert effective rank (over per-expert hidden state trajectories)
        ranks = per_expert_effective_rank(model.cell)
        diversity = expert_diversity_ratio(ranks)
        mean_er = float(np.mean(ranks))
    return final_task, float(weight_sim), float(diversity), mean_er


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--out", default="results/bench_snnl_expert_disentanglement.json")
    args = p.parse_args()
    epochs = 30 if args.quick else 100
    n_seeds = args.seeds
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out: dict = {
        "epochs": epochs,
        "n_seeds": n_seeds,
        "snnl_lambda": SNNL_LAMBDA,
        "orth_lambda": ORTH_LAMBDA,
        "temperature": TEMPERATURE,
        "wall_time_s": 0.0,
        "datasets": {},
    }
    T = 64

    for ds_name, ds_fn in DATASETS.items():
        ds_out: dict = {}
        for cond in ("baseline", "snnl", "orth", "snnl_orth"):
            use_snnl = cond in ("snnl", "snnl_orth")
            use_orth = cond in ("orth", "snnl_orth")
            cond_out: list[dict] = []
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                t, y = ds_fn(T, seed=seed)
                model = FAMEModel(n_experts=4)
                task_loss, weight_sim, diversity, mean_er = train_model(
                    model, t, y, epochs=epochs, lr=1e-2,
                    use_snnl=use_snnl, use_orth=use_orth,
                )
                cond_out.append({
                    "task_loss": task_loss,
                    "weight_sim": weight_sim,
                    "diversity_ratio": diversity,
                    "mean_eff_rank": mean_er,
                })
            def agg(field: str) -> tuple[float, float]:
                vals = [s[field] for s in cond_out if s[field] is not None]
                if not vals:
                    return 0.0, 0.0
                return float(np.mean(vals)), float(np.std(vals))
            ds_out[cond] = {
                "task_loss_mean_std": agg("task_loss"),
                "weight_sim_mean_std": agg("weight_sim"),
                "diversity_ratio_mean_std": agg("diversity_ratio"),
                "mean_eff_rank_mean_std": agg("mean_eff_rank"),
                "per_seed": cond_out,
            }
        out["datasets"][ds_name] = ds_out

    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))

    # Pretty print
    print(f"\n=== Round 100 SNNL expert disentanglement bench "
          f"(epochs={epochs}, seeds={n_seeds}, λ_snnl={SNNL_LAMBDA}, "
          f"λ_orth={ORTH_LAMBDA}, T={TEMPERATURE}) ===\n")
    print(f"{'dataset':12s} | {'cond':10s} | {'task_loss':>10s} | {'wgt_sim':>10s} | "
          f"{'div_ratio':>10s} | {'mean_er':>8s}")
    print("-" * 90)
    for ds_name in DATASETS:
        for cond in ("baseline", "snnl", "orth", "snnl_orth"):
            c = out["datasets"][ds_name][cond]
            tl_m, _ = c["task_loss_mean_std"]
            ws_m, _ = c["weight_sim_mean_std"]
            dv_m, _ = c["diversity_ratio_mean_std"]
            er_m, _ = c["mean_eff_rank_mean_std"]
            print(f"{ds_name:12s} | {cond:10s} | {tl_m:10.4f} | {ws_m:10.4f} | "
                  f"{dv_m:10.4f} | {er_m:8.3f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
