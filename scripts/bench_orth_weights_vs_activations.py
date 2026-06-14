"""Round 90 audit bench (PRD #10-52, response to arXiv:2601.00457 Kim 2026).

Compares activation-space and weight-space orth overlap under round 80's
``orthogonality_loss`` (acts on activations) at multiple λ strengths.

2 conditions × 3 datasets × 4 orth λ ∈ {0, 0.1, 1.0, 10.0} = 24 cells.
For each cell we report:
- loss_final
- activation_space_overlap (last epoch)
- weight_space_overlap (last epoch)
- E_emp_last
- max_min_ratio_grad

Hypotheses:
- H1 (Kim disconnect): weight_overlap INCREASES with λ
- H2 (our target): activation_overlap DECREASES with λ
- H3 (no disconnect): weight_overlap and activation_overlap are negatively correlated
- H4 (clean signal): loss_final improves monotonically with λ

Run:
    .venv312/bin/python scripts/bench_orth_weights_vs_activations.py --quick
    .venv312/bin/python scripts/bench_orth_weights_vs_activations.py        # full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lnn.core.fame_cfc import FAMECfCCell
from lnn.core.moe_ecology import (
    activation_space_overlap,
    weight_space_overlap,
)


# ---------------------------------------------------------------------------
# Synthetic datasets (same as round 83-89)
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
# Collect expert weight matrices (per-expert W from FAME routing)
# ---------------------------------------------------------------------------

def collect_expert_weights(cell: FAMECfCCell) -> list[torch.Tensor]:
    """Return a list of per-expert weight matrices (K of them, same shape).

    Each expert in FAMECfCCell has its own ``f_gate.0.weight`` of shape
    (out, in). We collect one canonical per-expert weight per expert.
    """
    named_params = dict(cell.named_parameters())
    per_expert: dict[int, list[torch.Tensor]] = {i: [] for i in range(cell.n_experts)}
    for name, p in named_params.items():
        # Match ``experts.<i>.<...>.weight`` with shape (out, in).
        if name.startswith("experts.") and name.endswith(".weight") and p.dim() == 2:
            parts = name.split(".")
            try:
                idx = int(parts[1])
            except (ValueError, IndexError):
                continue
            if 0 <= idx < cell.n_experts:
                per_expert[idx].append(p.detach().clone())
    # Take the first weight of each expert (canonical f_gate.0.weight).
    out: list[torch.Tensor] = []
    for i in range(cell.n_experts):
        if per_expert[i]:
            out.append(per_expert[i][0])
    if len(out) < 2:
        # Fallback: use the router.
        if "router.router.weight" in named_params:
            r = named_params["router.router.weight"]
            return [r[i].detach().clone() for i in range(r.shape[0])]
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(
    cell: FAMECfCCell,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    orth_lambda: float,
    lr: float = 1e-2,
) -> dict:
    """Train cell, collecting loss + final activation/weight overlaps."""
    params = [p for p in cell.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    cell.train()
    history = {"loss": [], "last_outs": None, "last_expert_weights": None}
    for epoch in range(epochs):
        opt.zero_grad()
        h = torch.zeros(x.shape[0], cell.hidden_size)
        task_loss_acc = 0.0
        last_outs = None
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            y_t = y[:, t, :]
            h_new, outs = cell.forward_with_aux(x_t, h, dt=1.0)
            task_loss_t = ((h_new - y_t) ** 2).mean()
            task_loss_acc = task_loss_acc + task_loss_t
            h = h_new
            last_outs = outs
        task_loss = task_loss_acc / x.shape[1]
        orth_loss = cell.compute_orth_loss(last_outs, user_lambda=orth_lambda)
        total_loss = task_loss + orth_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        history["loss"].append(float(total_loss.item()))
        history["last_outs"] = [o.detach().clone() for o in last_outs]
        history["last_expert_weights"] = collect_expert_weights(cell)
    return history


def cell_factory():
    """Build a plain FAMECfCCell — no gates (round 80 raw orth only)."""
    return FAMECfCCell(
        input_size=1, hidden_size=8, n_experts=3, top_k=1,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results/bench_orth_weights_vs_activations.json")
    args = p.parse_args()
    epochs = 2 if args.quick else 5
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out = {"epochs": epochs, "wall_time_s": 0.0, "runs": []}
    for orth_lambda in (0.0, 0.1, 1.0, 10.0):
        run = {"orth_lambda": orth_lambda, "datasets": {}}
        for ds_name, ds_fn in DATASETS.items():
            x, y = ds_fn(n_samples=32, seq_len=16, seed=0)
            torch.manual_seed(0)
            cell = cell_factory()
            h = train_one(cell, x, y, epochs=epochs, orth_lambda=orth_lambda)
            run["datasets"][ds_name] = {
                "loss_final": h["loss"][-1],
                "activation_overlap": activation_space_overlap(h["last_outs"]),
                "weight_overlap": weight_space_overlap(h["last_expert_weights"]),
                "n_expert_weights": len(h["last_expert_weights"]),
            }
        out["runs"].append(run)
    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))

    # Pretty print: H1/H2 quick check.
    print(f"\n{'λ':>6s} | {'dataset':11s} | {'loss':>8s} | {'act_ov':>8s} | {'wgt_ov':>8s}")
    print("-" * 60)
    for run in out["runs"]:
        for ds_name, r in run["datasets"].items():
            print(
                f"{run['orth_lambda']:>6.2f} | {ds_name:11s} | {r['loss_final']:>8.4f} | "
                f"{r['activation_overlap']:>8.4f} | {r['weight_overlap']:>8.4f}"
            )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
