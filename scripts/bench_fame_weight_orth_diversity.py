"""Round 97 bench (PRD #10-59) — FAME + weight-level orthogonality.

Direct follow-up to round 96: if the round 80 activation-level
orthogonality_loss does NOT increase weight diversity, does a
weight-level penalty (||W_i W_j^T||_F^2 / ||W_i||_F · ||W_j||_F)?

We compare 4 conditions:
- baseline:        no orth
- +act_orth:       round 80 orthogonality_loss (λ=0.001)
- +wt_orth:        round 97 weight_orthogonality_loss (λ=0.001)
- +both:           both penalties

For each we measure: diversity_ratio (weight), task_loss, activation_cos_sim.

Cells: 3 datasets × 4 conditions × 3 seeds = 36 cells

Run:
    .venv312/bin/python scripts/bench_fame_weight_orth_diversity.py --quick
    .venv312/bin/python scripts/bench_fame_weight_orth_diversity.py        # full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lnn.core.effective_rank import expert_diversity_summary
from lnn.core.fame_cfc import FAMECfCCell
from lnn.core.moe_ecology import activation_space_overlap
from lnn.core.orthogonality import orthogonality_loss, weight_orthogonality_loss


# ---------------------------------------------------------------------------
# Datasets (same as rounds 95, 96)
# ---------------------------------------------------------------------------

def make_toy_sin(T: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    t = torch.linspace(0, 1, T)
    y = torch.sin(2 * np.pi * t) + 0.5 * torch.sin(10 * np.pi * t)
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
# Training
# ---------------------------------------------------------------------------

LAMBDA = 0.001  # round 83 safe setting


def train_fame(
    cell: FAMECfCCell,
    t: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    mode: str,
) -> float:
    """Train FAME cell stateless.  mode ∈ {"baseline", "act", "wt", "both"}."""
    opt = torch.optim.Adam(cell.parameters(), lr=lr)
    h0 = torch.zeros(1, cell.hidden_size)
    final_task_loss = 0.0
    for _ in range(epochs):
        opt.zero_grad()
        h = h0
        outs: list[torch.Tensor] = []
        all_per_expert: list[list[torch.Tensor]] = [[] for _ in range(cell.n_experts)]
        for ti in t:
            x = ti.reshape(1, 1)
            if mode in ("act", "both"):
                h_new, expert_outs = cell.forward_with_aux(x, h, dt=1.0)
                for k in range(cell.n_experts):
                    all_per_expert[k].append(expert_outs[k].squeeze(0))
            else:
                h_new = cell(x, h, dt=1.0)
            outs.append(h_new)
            h = h_new
        h_traj = torch.stack(outs, dim=0).squeeze(1)
        y_pred_traj = h_traj.mean(dim=-1)
        task_loss = ((y_pred_traj - y) ** 2).mean()
        total_loss = task_loss
        if mode in ("act", "both"):
            per_expert_traj = [torch.stack(eks, dim=0) for eks in all_per_expert]
            total_loss = total_loss + orthogonality_loss(per_expert_traj, lambda_coeff=LAMBDA)
        if mode in ("wt", "both"):
            total_loss = total_loss + cell.compute_weight_orth_loss(lambda_coeff=LAMBDA)
        total_loss.backward()
        opt.step()
        final_task_loss = float(task_loss.item())
    return final_task_loss


def measure_activation_diversity(cell: FAMECfCCell, t: torch.Tensor) -> float:
    """Mean pairwise |cos_sim| of expert hidden states (lower = more diverse)."""
    h0 = torch.zeros(1, cell.hidden_size)
    h = h0
    per_expert_traj: list[torch.Tensor] = [torch.zeros(len(t), cell.hidden_size) for _ in range(cell.n_experts)]
    with torch.no_grad():
        for t_idx, ti in enumerate(t):
            x = ti.reshape(1, 1)
            _, eo = cell.forward_with_aux(x, h, dt=1.0)
            for k in range(cell.n_experts):
                per_expert_traj[k][t_idx] = eo[k].squeeze(0)
            h = cell(x, h, dt=1.0)
    return activation_space_overlap(per_expert_traj)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="results/bench_fame_weight_orth_diversity.json")
    args = p.parse_args()
    epochs = 30 if args.quick else 100
    n_seeds = args.seeds
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out: dict = {
        "epochs": epochs,
        "n_seeds": n_seeds,
        "lambda": LAMBDA,
        "wall_time_s": 0.0,
        "datasets": {},
    }
    T = 64
    modes = ("baseline", "act", "wt", "both")

    for ds_name, ds_fn in DATASETS.items():
        ds_out: dict = {}
        for mode in modes:
            cond_out: list[dict] = []
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                t, y = ds_fn(T, seed=seed)
                cell = FAMECfCCell(input_size=1, hidden_size=8, n_experts=5, top_k=2)
                task_loss = train_fame(cell, t, y, epochs=epochs, lr=1e-2, mode=mode)
                div_summary = expert_diversity_summary(cell)
                act_div = measure_activation_diversity(cell, t)
                cond_out.append({
                    "diversity_ratio": div_summary["diversity_ratio"],
                    "mean_eff_rank": div_summary["mean"],
                    "task_loss": task_loss,
                    "activation_cos_sim": act_div,
                })
            def agg(field: str) -> tuple[float, float]:
                vals = [s[field] for s in cond_out if s[field] is not None]
                if not vals:
                    return 0.0, 0.0
                return float(np.mean(vals)), float(np.std(vals))
            ds_out[mode] = {
                "diversity_ratio_mean_std": agg("diversity_ratio"),
                "mean_eff_rank_mean_std": agg("mean_eff_rank"),
                "task_loss_mean_std": agg("task_loss"),
                "activation_cos_sim_mean_std": agg("activation_cos_sim"),
                "per_seed": cond_out,
            }
        out["datasets"][ds_name] = ds_out

    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\n=== Round 97 FAME+weight_orth diversity bench (epochs={epochs}, seeds={n_seeds}, λ={LAMBDA}) ===\n")
    print(f"{'dataset':12s} | {'cond':10s} | {'div_ratio':>12s} | {'mean_eff':>10s} | {'task_loss':>12s} | {'act_cos':>10s}")
    print("-" * 90)
    for ds_name in DATASETS:
        for mode in modes:
            c = out["datasets"][ds_name][mode]
            dr_m, dr_s = c["diversity_ratio_mean_std"]
            m_m, _ = c["mean_eff_rank_mean_std"]
            tl_m, _ = c["task_loss_mean_std"]
            ac_m, _ = c["activation_cos_sim_mean_std"]
            dr_str = f"{dr_m:5.2f}±{dr_s:4.2f}"
            print(f"{ds_name:12s} | {mode:10s} | {dr_str:>12s} | {m_m:10.2f} | {tl_m:12.4f} | {ac_m:10.4f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
