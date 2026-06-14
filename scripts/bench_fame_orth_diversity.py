"""Round 96 bench (PRD #10-58) — Test FAME with orthogonality (H4 from round 95).

Direct test of the round 80 mechanism: does ``orthogonality_loss``
(PRD #10-37, arXiv:2606.03631 AnchorMoE) at the safe λ=0.001 setting
(round 83 confirmed safe) actually increase FAME's expert diversity?

We compare:
- FAME-baseline: no orth loss (the round 95 baseline)
- FAME+orth(λ=0.001): train with orth loss added to task loss

For each we measure:
- per_expert_eff_rank (round 95 tool)
- diversity_ratio (max/min)
- task_loss (final MSE)
- orth_loss (auxiliary, FAME+orth only)
- activation_diversity: mean pairwise cos_sim of expert outputs
  (lower = more diverse, round 90's activation_space_overlap family)

Cells:
  3 datasets × 2 conditions × 3 seeds = 18 cells

Run:
    .venv312/bin/python scripts/bench_fame_orth_diversity.py --quick
    .venv312/bin/python scripts/bench_fame_orth_diversity.py        # full
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
from lnn.core.orthogonality import orthogonality_loss


# ---------------------------------------------------------------------------
# Datasets (same as round 95)
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
# Training (stateless, same as round 95)
# ---------------------------------------------------------------------------

ORTH_LAMBDA = 0.001  # round 83: safe setting


def train_fame(
    cell: FAMECfCCell,
    t: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    use_orth: bool,
) -> float:
    """Train FAME cell stateless (h reset each step). Returns final task loss.

    If ``use_orth``, adds orth loss at λ=0.001 to the task loss.
    The orth loss is computed from the SAME forward pass as the task
    loss so gradients flow back to the expert weights.
    """
    opt = torch.optim.Adam(cell.parameters(), lr=lr)
    h0 = torch.zeros(1, cell.hidden_size)
    final_task_loss = 0.0
    for _ in range(epochs):
        opt.zero_grad()
        h = h0
        outs: list[torch.Tensor] = []
        # Collect per-step expert outputs across the trajectory so
        # we can compute the orth loss over the full sequence.  For
        # each step, expert_outs is K × [1, H]; we store the [H] view.
        all_per_expert: list[list[torch.Tensor]] = [[] for _ in range(cell.n_experts)]
        for ti in t:
            x = ti.reshape(1, 1)
            if use_orth:
                h_new, expert_outs = cell.forward_with_aux(x, h, dt=1.0)
            else:
                h_new = cell(x, h, dt=1.0)
            outs.append(h_new)
            if use_orth:
                for k in range(cell.n_experts):
                    all_per_expert[k].append(expert_outs[k].squeeze(0))  # [H]
            h = h_new
        # Aggregate: MSE between mean-projection trajectory and target.
        h_traj = torch.stack(outs, dim=0).squeeze(1)  # [T, H]
        y_pred_traj = h_traj.mean(dim=-1)  # [T]
        task_loss = ((y_pred_traj - y) ** 2).mean()
        if use_orth:
            # Stack per-expert trajectories: each is [T, H], requires_grad=True.
            per_expert_traj = [torch.stack(eks, dim=0) for eks in all_per_expert]
            orth_aux = orthogonality_loss(per_expert_traj, lambda_coeff=ORTH_LAMBDA)
            total_loss = task_loss + orth_aux
        else:
            total_loss = task_loss
        total_loss.backward()
        opt.step()
        final_task_loss = float(task_loss.item())
    return final_task_loss


def measure_activation_diversity(
    cell: FAMECfCCell,
    t: torch.Tensor,
) -> float:
    """Measure mean pairwise |cos_sim| of expert hidden states on the trajectory.

    Lower = more diverse (orthogonal experts produce different
    representations).  Uses round 90's ``activation_space_overlap``
    which returns a single float.
    """
    h0 = torch.zeros(1, cell.hidden_size)
    h = h0
    per_expert_trajectory: list[torch.Tensor] = [torch.zeros(len(t), cell.hidden_size) for _ in range(cell.n_experts)]
    with torch.no_grad():
        for t_idx, ti in enumerate(t):
            x = ti.reshape(1, 1)
            _, eo = cell.forward_with_aux(x, h, dt=1.0)
            # eo is K × [1, H]
            for k in range(cell.n_experts):
                per_expert_trajectory[k][t_idx] = eo[k].squeeze(0)
            h = cell(x, h, dt=1.0)
    return activation_space_overlap(per_expert_trajectory)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="results/bench_fame_orth_diversity.json")
    args = p.parse_args()
    epochs = 30 if args.quick else 100
    n_seeds = args.seeds
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out: dict = {
        "epochs": epochs,
        "n_seeds": n_seeds,
        "orth_lambda": ORTH_LAMBDA,
        "wall_time_s": 0.0,
        "datasets": {},
    }
    T = 64

    for ds_name, ds_fn in DATASETS.items():
        ds_out: dict = {}
        for cond in ("baseline", "orth"):
            use_orth = cond == "orth"
            cond_out: list[dict] = []
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                t, y = ds_fn(T, seed=seed)
                cell = FAMECfCCell(
                    input_size=1, hidden_size=8,
                    n_experts=5, top_k=2,
                )
                task_loss = train_fame(cell, t, y, epochs=epochs, lr=1e-2, use_orth=use_orth)
                div_summary = expert_diversity_summary(cell)
                act_div = measure_activation_diversity(cell, t)
                cond_out.append({
                    "diversity_ratio": div_summary["diversity_ratio"],
                    "mean_eff_rank": div_summary["mean"],
                    "min_eff_rank": div_summary["min"],
                    "max_eff_rank": div_summary["max"],
                    "std_eff_rank": div_summary["std"],
                    "per_expert_eff_rank": div_summary["per_expert"],
                    "task_loss": task_loss,
                    "activation_cos_sim": act_div,
                })
            # Aggregate
            def agg(field: str) -> tuple[float, float]:
                vals = [s[field] for s in cond_out if s[field] is not None]
                if not vals:
                    return 0.0, 0.0
                return float(np.mean(vals)), float(np.std(vals))
            ds_out[cond] = {
                "diversity_ratio_mean_std": agg("diversity_ratio"),
                "mean_eff_rank_mean_std": agg("mean_eff_rank"),
                "task_loss_mean_std": agg("task_loss"),
                "activation_cos_sim_mean_std": agg("activation_cos_sim"),
                "per_seed": cond_out,
            }
        out["datasets"][ds_name] = ds_out

    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))

    # Pretty print
    print(f"\n=== Round 96 FAME+orth diversity bench (epochs={epochs}, seeds={n_seeds}, λ={ORTH_LAMBDA}) ===\n")
    print(f"{'dataset':12s} | {'cond':10s} | {'div_ratio':>12s} | {'mean_eff':>10s} | {'task_loss':>12s} | {'act_cos':>10s}")
    print("-" * 90)
    for ds_name in DATASETS:
        for cond in ("baseline", "orth"):
            c = out["datasets"][ds_name][cond]
            dr_m, dr_s = c["diversity_ratio_mean_std"]
            m_m, _ = c["mean_eff_rank_mean_std"]
            tl_m, _ = c["task_loss_mean_std"]
            ac_m, _ = c["activation_cos_sim_mean_std"]
            dr_str = f"{dr_m:5.2f}±{dr_s:4.2f}"
            print(f"{ds_name:12s} | {cond:10s} | {dr_str:>12s} | {m_m:10.2f} | {tl_m:12.4f} | {ac_m:10.4f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
