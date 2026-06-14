"""Smoke bench for ecology-gated φ-balancing (PRD #10-43, 2026-06-14).

Three cells × three datasets, with three conditions:

A) baseline (no φ, no orth)
B) always-on φ (round 81, η=0.05)
C) **ecology-gated φ** (auto-enable when E < 0.5)

To force E to drop below 0.5, we inject orth λ=1.0 (the paper's
threshold region, per round 83 B).  We measure:
- final test loss
- whether the gate fired (and at which step)
- E trajectory
- dead_experts trajectory

Run:
    .venv312/bin/python scripts/bench_ecology_gated.py --quick
    .venv312/bin/python scripts/bench_ecology_gated.py            # full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lnn.core.fame_cfc import FAMECfCCell
from lnn.core.moe_ecology import MoEEcologyMonitor


# ---------------------------------------------------------------------------
# Synthetic datasets (same as round 83 B)
# ---------------------------------------------------------------------------

def make_sin_dataset(n_samples: int = 32, seq_len: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
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
# Training
# ---------------------------------------------------------------------------

def train_with_gated_diagnostic(
    cell: FAMECfCCell,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 5,
    orth_lambda: float = 0.0,
    lr: float = 1e-2,
) -> dict:
    """Train cell and run the ecology diagnostic each epoch.

    orth_lambda > 0 forces E to drop below 0.5 (per round 83 B).
    Returns dict with final loss, E trajectory, gate trajectory, dead.
    """
    params = [p for p in cell.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    cell.train()
    history = {"loss": [], "E": [], "dead": [], "gate_fired_step": -1}
    for epoch in range(epochs):
        opt.zero_grad()
        h = torch.zeros(x.shape[0], cell.hidden_size)
        loss = 0.0
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            y_t = y[:, t, :]
            if orth_lambda > 0:
                h_new, outs = cell.forward_with_aux(x_t, h, dt=1.0)
                from lnn.core.orthogonality import orthogonality_loss
                aux = orthogonality_loss(outs, lambda_coeff=orth_lambda)
            else:
                h_new = cell.forward(x_t, h, dt=1.0)
                aux = 0.0
            loss_t = ((h_new - y_t) ** 2).mean() + aux
            loss = loss + loss_t
            h = h_new
        loss = loss / x.shape[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        history["loss"].append(float(loss.item()))
        # Compute diagnostic and gate state.
        diag = cell.moe_ecology_diagnostic(B=orth_lambda)
        history["E"].append(diag["E"])
        history["dead"].append(diag["dead_experts"])
        if "ecology_gate" in diag and diag["ecology_gate"]["intervened"]:
            if history["gate_fired_step"] == -1:
                history["gate_fired_step"] = diag["ecology_gate"]["triggered_step"]
    return history


# ---------------------------------------------------------------------------
# Experiment: 3 conditions × 3 datasets
# ---------------------------------------------------------------------------

def run_condition(
    cond_name: str,
    cell_factory,
    datasets: dict,
    epochs: int,
    orth_lambda: float = 1.0,
) -> dict:
    """Run one condition across all 3 datasets."""
    results = {}
    for ds_name, ds_fn in datasets.items():
        x, y = ds_fn(n_samples=32, seq_len=16, seed=0)
        torch.manual_seed(0)
        cell = cell_factory()
        h = train_with_gated_diagnostic(cell, x, y, epochs=epochs, orth_lambda=orth_lambda)
        results[f"ds={ds_name}"] = {
            "loss_final": h["loss"][-1],
            "E_first": h["E"][0] if h["E"] else None,
            "E_last": h["E"][-1] if h["E"] else None,
            "dead_final": h["dead"][-1] if h["dead"] else None,
            "gate_fired_step": h["gate_fired_step"],
        }
    return {"name": cond_name, "results": results}


def make_baseline() -> FAMECfCCell:
    return FAMECfCCell(input_size=1, hidden_size=8, n_experts=3, top_k=1)


def make_always_phi() -> FAMECfCCell:
    return FAMECfCCell(
        input_size=1, hidden_size=8, n_experts=3, top_k=1,
        phi_balance=True, phi_step_size=0.05, ema_alpha=0.05,
    )


def make_gated_phi() -> FAMECfCCell:
    return FAMECfCCell(
        input_size=1, hidden_size=8, n_experts=3, top_k=1,
        ecology_gated_balancing=True, ecology_E_min=0.5,
        phi_step_size=0.05, ema_alpha=0.05,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Fewer epochs for fast smoke test")
    p.add_argument("--out", default="results/bench_ecology_gated.json", help="Output JSON path")
    args = p.parse_args()
    epochs = 2 if args.quick else 5
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # Force E to drop below 0.5 by injecting orth λ=1.0.
    orth_lambda = 1.0
    a = run_condition("A: baseline (no φ)", make_baseline, DATASETS, epochs, orth_lambda)
    print(f"\n[A] {a['name']}  (orth λ=1.0 forces E < 0.5)")
    for k, v in a["results"].items():
        print(f"  {k}: loss={v['loss_final']:.4f}  E_last={v['E_last']:.2f}  dead={v['dead_final']}  gate={v['gate_fired_step']}")
    b = run_condition("B: always-on φ (round 81)", make_always_phi, DATASETS, epochs, orth_lambda)
    print(f"\n[B] {b['name']}  (orth λ=1.0 + always-on φ η=0.05)")
    for k, v in b["results"].items():
        print(f"  {k}: loss={v['loss_final']:.4f}  E_last={v['E_last']:.2f}  dead={v['dead_final']}  gate={v['gate_fired_step']}")
    c = run_condition("C: ecology-gated φ (round 84)", make_gated_phi, DATASETS, epochs, orth_lambda)
    print(f"\n[C] {c['name']}  (orth λ=1.0 + auto-enable φ when E < 0.5)")
    for k, v in c["results"].items():
        print(f"  {k}: loss={v['loss_final']:.4f}  E_last={v['E_last']:.2f}  dead={v['dead_final']}  gate={v['gate_fired_step']}")

    out = {
        "epochs": epochs,
        "orth_lambda": orth_lambda,
        "wall_time_s": round(time.time() - t0, 2),
        "A": a,
        "B": b,
        "C": c,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
