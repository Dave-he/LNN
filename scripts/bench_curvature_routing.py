"""Round 101 bench (PRD #10-63) — Ollivier-Ricci Curvature routing signal.

Direct test of arXiv:2603.22317 (Cao et al., March 2026) — *Geometric
Mixture-of-Experts with Curvature-Guided Adaptive Routing* (GeoMoE).

We implement ORC as a routing regularizer and apply it to a small MoE
model (FAMECfC with K=4 experts). The hypothesis is that the curvature
of the expert manifold captures a different geometric property than
weight/activation orthogonality (rounds 80, 97).

For each of 3 datasets (toy_sin, structured, random), we compare:
- baseline (no aux)
- +ORC λ=0.001 (round 101)
- +orth λ=0.001 (round 80, activation orthogonality)
- +ORC+orth combined

Cells: 1 model × 3 datasets × 4 conditions × 2 seeds = 24 cells

For each cell measure:
- task_loss
- mean_ollivier_ricci (round 101 metric — the new signal)
- expert_diversity_ratio (round 95)
- mean_eff_rank (round 94)

H1: +ORC has higher mean_orc than baseline.
H2: +ORC has higher diversity_ratio than baseline.
H3: task loss within ±10% at λ=0.001.

Run:
    .venv312/bin/python scripts/bench_curvature_routing.py --quick
    .venv312/bin/python scripts/bench_curvature_routing.py        # full
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
from lnn.core.curvature import mean_ollivier_ricci


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

    per_timestep_expert_features: (T, K, H) tensor — per-timestep per-expert
    states. We compute ORC on the per-expert mean (K, H) to capture the
    curvature of the expert manifold.
    """

    def __init__(self, n_experts: int = 4) -> None:
        super().__init__()
        self.cell = FAMECfCCell(input_size=1, hidden_size=16, n_experts=n_experts)
        self.head = nn.Linear(16, 1)
        self.n_experts = n_experts

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        per_ts = []
        outs = []
        h = torch.zeros(1, self.cell.hidden_size)
        for ti in t:
            x_t = ti.reshape(1, 1)
            h, expert_outs = self.cell.forward_with_aux(x_t, h, dt=1.0)
            outs.append(self.head(h))
            per_ts.append(torch.stack([eo.squeeze(0) for eo in expert_outs], dim=0))
        y = torch.cat(outs, dim=-1).squeeze(0)
        per_timestep = torch.stack(per_ts, dim=0)  # (T, K, H)
        return y, per_timestep


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

ORC_LAMBDA = 0.001
ORTH_LAMBDA = 0.001
ORC_K = 2  # k-NN parameter for ORC
SINKHORN_ITERS = 5  # small N=4 → fast Sinkhorn


def train_model(
    model: FAMEModel,
    t: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    use_orc: bool,
    use_orth: bool,
) -> tuple[float, float, float, float, float]:
    """Train and return (task_loss, mean_orc, weight_sim, diversity_ratio, mean_eff_rank)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        y_pred, per_timestep = model(t)
        task_loss = ((y_pred - y) ** 2).mean()
        total_loss = task_loss
        if use_orc or use_orth:
            # Compute per-expert mean across timesteps: (K, H)
            per_expert_means = per_timestep.mean(dim=0)  # (K, H)
            if use_orc:
                # Penalty: lambda * (1 - mean ORC) — encourages high ORC
                m_orc = mean_ollivier_ricci(
                    per_expert_means, k=ORC_K, sinkhorn_iters=SINKHORN_ITERS,
                )
                total_loss = total_loss + ORC_LAMBDA * (1.0 - m_orc)
            if use_orth:
                # Orthogonality between per-expert mean features
                orth = orthogonality_loss(
                    [per_expert_means[k].unsqueeze(0) for k in range(model.n_experts)],
                    lambda_coeff=ORTH_LAMBDA,
                )
                total_loss = total_loss + orth
        total_loss.backward()
        opt.step()
    # Final measurement
    with torch.no_grad():
        y_pred, per_timestep = model(t)
        final_task = float(((y_pred - y) ** 2).mean().item())
        # ORC of the final per-expert manifold
        per_expert_means = per_timestep.mean(dim=0)
        m_orc = mean_ollivier_ricci(
            per_expert_means, k=ORC_K, sinkhorn_iters=SINKHORN_ITERS,
        )
        # Weight overlap
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
        # Per-expert effective rank
        ranks = per_expert_effective_rank(model.cell)
        diversity = expert_diversity_ratio(ranks)
        mean_er = float(np.mean(ranks))
    return (
        final_task, float(m_orc), float(weight_sim),
        float(diversity), mean_er,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--out", default="results/bench_curvature_routing.json")
    args = p.parse_args()
    epochs = 30 if args.quick else 100
    n_seeds = args.seeds
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out: dict = {
        "epochs": epochs,
        "n_seeds": n_seeds,
        "orc_lambda": ORC_LAMBDA,
        "orth_lambda": ORTH_LAMBDA,
        "orc_k": ORC_K,
        "sinkhorn_iters": SINKHORN_ITERS,
        "wall_time_s": 0.0,
        "datasets": {},
    }
    T = 64

    for ds_name, ds_fn in DATASETS.items():
        ds_out: dict = {}
        for cond in ("baseline", "orc", "orth", "orc_orth"):
            use_orc = cond in ("orc", "orc_orth")
            use_orth = cond in ("orth", "orc_orth")
            cond_out: list[dict] = []
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                t, y = ds_fn(T, seed=seed)
                model = FAMEModel(n_experts=4)
                result = train_model(
                    model, t, y, epochs=epochs, lr=1e-2,
                    use_orc=use_orc, use_orth=use_orth,
                )
                task_loss, m_orc, weight_sim, diversity, mean_er = result
                cond_out.append({
                    "task_loss": task_loss,
                    "mean_orc": m_orc,
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
                "mean_orc_mean_std": agg("mean_orc"),
                "weight_sim_mean_std": agg("weight_sim"),
                "diversity_ratio_mean_std": agg("diversity_ratio"),
                "mean_eff_rank_mean_std": agg("mean_eff_rank"),
                "per_seed": cond_out,
            }
        out["datasets"][ds_name] = ds_out

    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))

    # Pretty print
    print(f"\n=== Round 101 Curvature routing bench "
          f"(epochs={epochs}, seeds={n_seeds}, λ_orc={ORC_LAMBDA}, "
          f"λ_orth={ORTH_LAMBDA}, k={ORC_K}, sinkhorn={SINKHORN_ITERS}) ===\n")
    print(f"{'dataset':12s} | {'cond':10s} | {'task_loss':>10s} | {'mean_orc':>10s} | "
          f"{'wgt_sim':>10s} | {'div_ratio':>10s} | {'mean_er':>8s}")
    print("-" * 100)
    for ds_name in DATASETS:
        for cond in ("baseline", "orc", "orth", "orc_orth"):
            c = out["datasets"][ds_name][cond]
            tl_m, _ = c["task_loss_mean_std"]
            orc_m, _ = c["mean_orc_mean_std"]
            ws_m, _ = c["weight_sim_mean_std"]
            dv_m, _ = c["diversity_ratio_mean_std"]
            er_m, _ = c["mean_eff_rank_mean_std"]
            print(f"{ds_name:12s} | {cond:10s} | {tl_m:10.4f} | {orc_m:10.4f} | "
                  f"{ws_m:10.4f} | {dv_m:10.4f} | {er_m:8.3f}")
        print()


if __name__ == "__main__":
    main()
