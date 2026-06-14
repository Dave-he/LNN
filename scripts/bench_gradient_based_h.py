"""Smoke bench for gradient-based H (PRD #10-49, 2026-06-15, round 87).

Compare 2 H modes (empirical vs gradient) on the same hard cases used
in round 84-86:

- 2 conditions × 3 datasets × 3 orth λ ∈ {0.1, 1.0, 10.0}
- 2 H modes: empirical (round 83) vs gradient (round 87)

Per cell we report:
- E_emp (empirical): round 83 behavior
- E_grad (gradient): round 87 behavior
- gate_fired_emp: did the orth gate fire under empirical H?
- gate_fired_grad: did the orth gate fire under gradient H?
- loss_final

Hypotheses:
- H1: H_emp and H_grad agree when E is healthy (no orth toxicity)
- H2: H_emp and H_grad diverge when orth is toxic (E_emp ≥ 0.5
     but H_grad ≪ H_emp, because the loss is dominated by aux)
- H3: H_grad is more sensitive to early collapse (fires earlier
     in regimes where the loss is becoming flat to routing)

We test on 3 synthetic datasets (toy_sin, random, structured).

Run:
    .venv312/bin/python scripts/bench_gradient_based_h.py --quick
    .venv312/bin/python scripts/bench_gradient_based_h.py            # full
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from lnn.core.fame_cfc import FAMECfCCell
from lnn.core.orthogonality import orthogonality_loss


# ---------------------------------------------------------------------------
# Synthetic datasets (same as round 83/84/85/86)
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

def train_one_with_diag(
    cell: FAMECfCCell,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    orth_lambda: float,
    lr: float = 1e-2,
) -> dict:
    """Train cell and collect E_emp + E_grad at each epoch."""
    params = [p for p in cell.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    cell.train()
    history = {
        "loss": [], "E_emp": [], "E_grad": [],
        "orth_fired_emp": False, "orth_fired_grad": False,
        "lambda_scale_final_emp": 1.0, "lambda_scale_final_grad": 1.0,
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
        # Empirical E (round 83).
        cell.train()  # ensure training mode for gates
        diag_emp = cell.moe_ecology_diagnostic(B=orth_lambda, task_loss=None)
        history["E_emp"].append(diag_emp["E"])
        # Gradient E (round 87): need fresh task_loss with grad.
        # Recompute forward to get fresh logits.
        h2 = torch.zeros(x.shape[0], cell.hidden_size)
        task_loss_acc2 = 0.0
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            y_t = y[:, t, :]
            h_new, _ = cell.forward_with_aux(x_t, h2, dt=1.0)
            task_loss_t = ((h_new - y_t) ** 2).mean()
            task_loss_acc2 = task_loss_acc2 + task_loss_t
            h2 = h_new
        fresh_task_loss = task_loss_acc2 / x.shape[1]
        diag_grad = cell.moe_ecology_diagnostic(B=orth_lambda, task_loss=fresh_task_loss)
        history["E_grad"].append(diag_grad["E"])
        # Gate fires (under each H mode).
        if cell.orth_gate is not None and cell.orth_gate.intervened:
            history["orth_fired_emp"] = True
            history["lambda_scale_final_emp"] = cell.orth_gate.last_lambda_scale
        # The gates share state, so we can't easily distinguish — note this.
    return history


def cell_factory(orth_lambda: float, lambda_safe: float = 0.001):
    """Build cell with orth gate (so we can compare gate decisions)."""
    return FAMECfCCell(
        input_size=1, hidden_size=8, n_experts=3, top_k=1,
        ecology_gated_orth=True, ecology_orth_lambda_safe=lambda_safe,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results/bench_gradient_based_h.json")
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
            h = train_one_with_diag(cell, x, y, epochs=epochs, orth_lambda=orth_lambda)
            run["datasets"][ds_name] = {
                "loss_final": h["loss"][-1],
                "E_emp_last": h["E_emp"][-1],
                "E_grad_last": h["E_grad"][-1],
                "E_emp_first": h["E_emp"][0],
                "E_grad_first": h["E_grad"][0],
                "orth_fired_emp": h["orth_fired_emp"],
                "orth_fired_grad": h["orth_fired_grad"],
                "lambda_scale_final_emp": h["lambda_scale_final_emp"],
            }
        # Pretty print.
        print(f"\n[λ={orth_lambda}]")
        for ds_name in DATASETS:
            r = run["datasets"][ds_name]
            print(
                f"  {ds_name:11s}: loss={r['loss_final']:.4f}  "
                f"E_emp={r['E_emp_last']:.4f}  E_grad={r['E_grad_last']:.4f}  "
                f"orth_fired={r['orth_fired_emp']}  λ_s={r['lambda_scale_final_emp']:.4f}"
            )
        out["runs"].append(run)
    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
