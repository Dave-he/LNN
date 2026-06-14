"""Smoke bench for ecology-gated orth rescaling (PRD #10-44, 2026-06-15).

Two conditions × three datasets × three orth λ values:

A) baseline (no gate, user λ)
B) **ecology-gated orth** (auto-rescale λ → 0.001 when E<0.5)

To force E < 0.5, we use high orth λ ∈ {0.1, 1.0, 10.0}.

Hypothesis: at λ=1.0 and λ=10.0, the gate should rescale λ down to
0.001, recovering the round 80 default behavior.  At λ=0.1, E is
borderline (depending on dataset), so the gate may or may not fire.

Run:
    .venv312/bin/python scripts/bench_ecology_gated_orth.py --quick
    .venv312/bin/python scripts/bench_ecology_gated_orth.py            # full
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
# Synthetic datasets (same as round 83 B / round 84)
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

def train_one(
    cell: FAMECfCCell,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 5,
    orth_lambda: float = 0.0,
    use_gated: bool = False,
    lr: float = 1e-2,
) -> dict:
    """Train cell with or without ecology-gated orth rescaling.

    use_gated=True → use cell.compute_orth_loss() (applies gate).
    use_gated=False → use orthogonality_loss() directly (no gate).
    """
    params = [p for p in cell.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    cell.train()
    history = {"loss": [], "E": [], "dead": [], "gate_triggered_step": -1, "lambda_scale_final": 1.0}
    for epoch in range(epochs):
        opt.zero_grad()
        h = torch.zeros(x.shape[0], cell.hidden_size)
        loss = 0.0
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            y_t = y[:, t, :]
            if orth_lambda > 0:
                h_new, outs = cell.forward_with_aux(x_t, h, dt=1.0)
                if use_gated:
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
        # Diagnostic.
        diag = cell.moe_ecology_diagnostic(B=orth_lambda)
        history["E"].append(diag["E"])
        history["dead"].append(diag["dead_experts"])
        if "ecology_gate_orth" in diag and diag["ecology_gate_orth"]["intervened"]:
            if history["gate_triggered_step"] == -1:
                history["gate_triggered_step"] = diag["ecology_gate_orth"]["triggered_step"]
            history["lambda_scale_final"] = diag["ecology_gate_orth"]["lambda_scale"]
    return history


# ---------------------------------------------------------------------------
# Experiment: 2 conditions × 3 datasets × 3 lambdas
# ---------------------------------------------------------------------------

def run_condition(cond_name: str, cell_factory, datasets: dict, orth_lambda: float, epochs: int) -> dict:
    results = {}
    for ds_name, ds_fn in datasets.items():
        x, y = ds_fn(n_samples=32, seq_len=16, seed=0)
        torch.manual_seed(0)
        cell = cell_factory()
        h = train_one(cell, x, y, epochs=epochs, orth_lambda=orth_lambda, use_gated=False)
        results[f"ds={ds_name}"] = {
            "loss_final": h["loss"][-1],
            "E_first": h["E"][0] if h["E"] else None,
            "E_last": h["E"][-1] if h["E"] else None,
            "dead_final": h["dead"][-1] if h["dead"] else None,
            "gate_fired": h["gate_triggered_step"] != -1,
            "lambda_scale_final": h["lambda_scale_final"],
        }
    return {"name": cond_name, "results": results}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Fewer epochs for fast smoke test")
    p.add_argument("--out", default="results/bench_ecology_gated_orth.json", help="Output JSON path")
    args = p.parse_args()
    epochs = 2 if args.quick else 5
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out = {"epochs": epochs, "wall_time_s": 0.0, "runs": []}
    for orth_lambda in (0.1, 1.0, 10.0):
        # A: baseline
        a = run_condition(
            f"A: baseline λ={orth_lambda}",
            lambda: FAMECfCCell(input_size=1, hidden_size=8, n_experts=3, top_k=1),
            DATASETS, orth_lambda, epochs,
        )
        # B: gated (use_gated=True via train_one)
        def make_gated() -> FAMECfCCell:
            return FAMECfCCell(
                input_size=1, hidden_size=8, n_experts=3, top_k=1,
                ecology_gated_orth=True, ecology_orth_lambda_safe=0.001,
            )
        # For B, we need use_gated=True — re-implement inline
        torch.manual_seed(0)
        b_results = {}
        for ds_name, ds_fn in DATASETS.items():
            x, y = ds_fn(n_samples=32, seq_len=16, seed=0)
            torch.manual_seed(0)
            cell = make_gated()
            h = train_one(cell, x, y, epochs=epochs, orth_lambda=orth_lambda, use_gated=True)
            b_results[f"ds={ds_name}"] = {
                "loss_final": h["loss"][-1],
                "E_first": h["E"][0] if h["E"] else None,
                "E_last": h["E"][-1] if h["E"] else None,
                "dead_final": h["dead"][-1] if h["dead"] else None,
                "gate_fired": h["gate_triggered_step"] != -1,
                "lambda_scale_final": h["lambda_scale_final"],
            }
        b = {"name": f"B: gated orth (auto-rescale to 0.001) λ={orth_lambda}", "results": b_results}
        # Print.
        print(f"\n[λ={orth_lambda}]")
        for k in a["results"]:
            va, vb = a["results"][k], b["results"][k]
            print(f"  {k}: A loss={va['loss_final']:.4f} E_last={va['E_last']:.2f} dead={va['dead_final']}  |  B loss={vb['loss_final']:.4f} E_last={vb['E_last']:.2f} dead={vb['dead_final']} gate_fired={vb['gate_fired']} λ_scale={vb['lambda_scale_final']}")
        out["runs"].append({"orth_lambda": orth_lambda, "A": a, "B": b})
    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
