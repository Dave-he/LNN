"""Smoke bench for combined ecology gates (PRD #10-48, 2026-06-15, round 86).

Four conditions × three datasets × three orth λ values:

A) baseline (no gate, user λ)
B) **φ gate only** (EcologyGatedBalancer, round 84)
C) **orth gate only** (EcologyGatedOrth, round 85)
D) **combined** (PRD #10-48, round 86) — both gates co-active

Hypotheses:
- H1 (cumulative): D ≤ min(B, C) — combined best
- H2 (orth dominates): D ≈ C — orth alone is enough
- H3 (φ adds noise): D > C — combined worse than orth alone

We test on 3 synthetic datasets (toy_sin, random, structured) and
3 orth λ ∈ {0.1, 1.0, 10.0} to cover healthy, toxic, and extreme
orth regimes.

Run:
    .venv312/bin/python scripts/bench_combined_gates.py --quick
    .venv312/bin/python scripts/bench_combined_gates.py            # full
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
# Synthetic datasets (same as round 83/84/85)
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

def train_one(cell, x, y, epochs, orth_lambda, lr=1e-2):
    """Train a FAMECfCCell; orth loss applied externally.

    The cell may or may not have ecology gates — they decide whether to
    rescale λ via cell.compute_orth_loss (D) or use orthogonality_loss
    directly (A, B, C).
    """
    params = [p for p in cell.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    cell.train()
    history = {"loss": [], "E": [], "dead": [], "phi_fired": False, "orth_fired": False, "lambda_scale_final": 1.0}
    for epoch in range(epochs):
        opt.zero_grad()
        h = torch.zeros(x.shape[0], cell.hidden_size)
        loss = 0.0
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            y_t = y[:, t, :]
            if orth_lambda > 0:
                h_new, outs = cell.forward_with_aux(x_t, h, dt=1.0)
                # Use cell.compute_orth_loss if the cell has the orth
                # gate (conditions C and D); else plain orthogonality_loss.
                if cell.orth_gate is not None:
                    aux = cell.compute_orth_loss(outs, user_lambda=orth_lambda)
                else:
                    aux = orthogonality_loss(outs, lambda_coeff=orth_lambda)
            else:
                h_new = cell.forward(x_t, h, dt=1.0)
                aux = torch.tensor(0.0)
            loss_t = ((h_new - y_t) ** 2).mean() + aux
            loss = loss + loss_t
            h = h_new
        loss = loss / x.shape[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        history["loss"].append(float(loss.item()))
        diag = cell.moe_ecology_diagnostic(B=orth_lambda)
        history["E"].append(diag["E"])
        history["dead"].append(diag["dead_experts"])
        if cell.ecology_gate is not None and cell.ecology_gate.intervened:
            history["phi_fired"] = True
        if cell.orth_gate is not None and cell.orth_gate.intervened:
            history["orth_fired"] = True
            history["lambda_scale_final"] = cell.orth_gate.last_lambda_scale
    return history


def cell_factory(cond: str, lambda_safe: float = 0.001):
    """Build a cell for the given condition."""
    if cond == "A":
        return FAMECfCCell(input_size=1, hidden_size=8, n_experts=3, top_k=1)
    if cond == "B":
        return FAMECfCCell(
            input_size=1, hidden_size=8, n_experts=3, top_k=1,
            ecology_gated_balancing=True,
        )
    if cond == "C":
        return FAMECfCCell(
            input_size=1, hidden_size=8, n_experts=3, top_k=1,
            ecology_gated_orth=True, ecology_orth_lambda_safe=lambda_safe,
        )
    if cond == "D":
        return FAMECfCCell(
            input_size=1, hidden_size=8, n_experts=3, top_k=1,
            ecology_combined=True, ecology_orth_lambda_safe=lambda_safe,
        )
    raise ValueError(cond)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results/bench_combined_gates.json")
    args = p.parse_args()
    epochs = 2 if args.quick else 5
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out = {"epochs": epochs, "wall_time_s": 0.0, "runs": []}
    for orth_lambda in (0.1, 1.0, 10.0):
        run = {"orth_lambda": orth_lambda, "conditions": {}}
        for cond in ("A", "B", "C", "D"):
            ds_results = {}
            for ds_name, ds_fn in DATASETS.items():
                x, y = ds_fn(n_samples=32, seq_len=16, seed=0)
                torch.manual_seed(0)
                cell = cell_factory(cond)
                h = train_one(cell, x, y, epochs=epochs, orth_lambda=orth_lambda)
                ds_results[ds_name] = {
                    "loss_final": h["loss"][-1],
                    "E_last": h["E"][-1],
                    "dead_final": h["dead"][-1],
                    "phi_fired": h["phi_fired"],
                    "orth_fired": h["orth_fired"],
                    "lambda_scale_final": h["lambda_scale_final"],
                }
            run["conditions"][cond] = ds_results
        # Pretty print.
        print(f"\n[λ={orth_lambda}]")
        for ds_name in DATASETS:
            a = run["conditions"]["A"][ds_name]
            b = run["conditions"]["B"][ds_name]
            c = run["conditions"]["C"][ds_name]
            d = run["conditions"]["D"][ds_name]
            print(
                f"  {ds_name:11s}: A={a['loss_final']:.4f}  "
                f"B={b['loss_final']:.4f}(φ={b['phi_fired']})  "
                f"C={c['loss_final']:.4f}(orth={c['orth_fired']},λ_s={c['lambda_scale_final']:.4f})  "
                f"D={d['loss_final']:.4f}(φ={d['phi_fired']},orth={d['orth_fired']},λ_s={d['lambda_scale_final']:.4f})"
            )
        out["runs"].append(run)
    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
