"""Smoke bench for MoE Ecology diagnostic (PRD #10-42, 2026-06-14).

Two experiments:

A) **E trajectory on the 16-cell grid** (round 79): for each
   (K, top_k, n_tau) in {2,3,5} × {1,2} × {1,2}, train a tiny
   FAMECfCCell on toy sin for 5 epochs, log E and dead_count per step.

B) **Ortho toxicity test** (arXiv:2605.06415 finding 2): K=3 top_k=1
   on three synthetic datasets (toy sin, random gaussian, structured
   sin+2*cos+noise), λ ∈ {0, 0.001, 0.01, 0.1, 1.0}.  We compare
   final test loss and final E to test the hypothesis that
   orthogonality helps on structured data but hurts on random data.

Run:
    .venv312/bin/python scripts/bench_moe_ecology.py --quick
    .venv312/bin/python scripts/bench_moe_ecology.py            # full
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
# Synthetic datasets
# ---------------------------------------------------------------------------

def make_sin_dataset(n_samples: int = 64, seq_len: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-frequency sin with mild noise (round 73-79 baseline)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, seq_len + 1).astype(np.float32)
    x_np = np.sin(t[1:])[None, :].repeat(n_samples, axis=0)
    y_np = np.sin(t[1:] + 0.1)[:, None].repeat(n_samples, axis=0).T
    x = torch.tensor(x_np).unsqueeze(-1)  # [N, T, 1]
    y = torch.tensor(y_np).unsqueeze(-1)  # [N, T, 1]
    x = x + 0.05 * torch.randn_like(x)
    return x, y


def make_random_dataset(n_samples: int = 64, seq_len: int = 32, dim: int = 1, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure Gaussian noise — no learnable structure."""
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.standard_normal((n_samples, seq_len, dim)).astype(np.float32))
    y = torch.tensor(rng.standard_normal((n_samples, seq_len, dim)).astype(np.float32))
    return x, y


def make_structured_dataset(n_samples: int = 64, seq_len: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """sin + 2*cos with 30% Gaussian noise (multi-frequency structured)."""
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
    use_orth: bool = False,
    lr: float = 1e-2,
    monitor: MoEEcologyMonitor | None = None,
    log_every: int = 1,
) -> dict:
    """Train a cell on (x, y) and return loss/E trajectory."""
    params = [p for p in cell.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    cell.train()
    history = {"loss": [], "E": [], "dead": []}
    for epoch in range(epochs):
        opt.zero_grad()
        h = torch.zeros(x.shape[0], cell.hidden_size)
        loss = 0.0
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            y_t = y[:, t, :]
            if use_orth:
                h_new, outs = cell.forward_with_aux(x_t, h, dt=1.0)
                from lnn.core.orthogonality import orthogonality_loss
                aux = orthogonality_loss(outs, lambda_coeff=orth_lambda)
            else:
                h_new = cell.forward(x_t, h, dt=1.0)
                aux = 0.0
            loss_t = ((h_new - y_t) ** 2).mean() + aux
            loss = loss + loss_t
            h = h_new
            if monitor is not None and t % log_every == 0:
                # Step the ecology monitor with the cell's last_g.
                if hasattr(cell, "last_g") and cell.last_g is not None:
                    monitor.step(cell.last_g, T=1.0, B=orth_lambda if use_orth else 0.0)
        loss = loss / x.shape[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        history["loss"].append(float(loss.item()))
        if monitor is not None and monitor.E_history:
            history["E"].append(monitor.E_history[-1])
            history["dead"].append(monitor.dead_history[-1])
    return history


# ---------------------------------------------------------------------------
# Experiment A: E trajectory on the 16-cell grid
# ---------------------------------------------------------------------------

def experiment_a(epochs: int = 5) -> dict:
    """Sweep (K × top_k × n_tau) on toy sin, log E trajectory."""
    x, y = make_sin_dataset(n_samples=32, seq_len=16, seed=0)
    grid = []
    for K in (2, 3, 5):
        for top_k in (1, 2):
            for n_tau in (1, 2):
                if top_k > K:
                    continue
                grid.append((K, top_k, n_tau))
    results = {}
    for K, top_k, n_tau in grid:
        torch.manual_seed(0)
        cell = FAMECfCCell(
            input_size=1, hidden_size=8, n_experts=K, top_k=top_k,
            n_tau_per_expert=n_tau, tau_scales=(0.1, 1.0, 10.0)[:n_tau] if n_tau <= 3 else (0.1, 1.0, 10.0),
        )
        monitor = MoEEcologyMonitor(n_experts=K, ema_alpha=0.1)
        h = train_one(cell, x, y, epochs=epochs, monitor=monitor, log_every=4)
        summary = monitor.summary()
        results[f"K={K},top_k={top_k},n_tau={n_tau}"] = {
            "loss_final": h["loss"][-1],
            "E_mean": summary["E_mean"],
            "E_last": summary["E_last"],
            "dead_final": summary["dead_experts"],
            "utilization": [round(u, 3) for u in summary["utilization"]],
        }
    return {"name": "A: 16-cell grid E trajectory", "results": results}


# ---------------------------------------------------------------------------
# Experiment B: ortho toxicity
# ---------------------------------------------------------------------------

def experiment_b(epochs: int = 5) -> dict:
    """Test arXiv:2605.06415 finding: ortho is dataset-dependent."""
    K, top_k = 3, 1
    lambdas = (0.0, 0.001, 0.01, 0.1, 1.0)
    datasets = ("toy_sin", "random", "structured")
    results = {}
    for ds_name in datasets:
        x, y = DATASETS[ds_name](n_samples=32, seq_len=16, seed=0)
        for lam in lambdas:
            torch.manual_seed(0)
            cell = FAMECfCCell(
                input_size=1, hidden_size=8, n_experts=K, top_k=top_k,
            )
            monitor = MoEEcologyMonitor(n_experts=K, ema_alpha=0.1)
            use_orth = lam > 0
            h = train_one(
                cell, x, y, epochs=epochs,
                orth_lambda=lam, use_orth=use_orth, monitor=monitor, log_every=4,
            )
            summary = monitor.summary()
            results[f"ds={ds_name},lambda={lam}"] = {
                "loss_final": h["loss"][-1],
                "E_last": summary["E_last"],
                "E_mean": summary["E_mean"],
                "dead_final": summary["dead_experts"],
                "utilization": [round(u, 3) for u in summary["utilization"]],
            }
    return {"name": "B: Ortho toxicity test (3 datasets × 5 lambdas)", "results": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Fewer epochs for fast smoke test")
    p.add_argument("--out", default="results/bench_moe_ecology.json", help="Output JSON path")
    args = p.parse_args()
    epochs = 2 if args.quick else 5
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    a = experiment_a(epochs=epochs)
    print(f"\n[A] {a['name']}")
    for k, v in a["results"].items():
        print(f"  {k}: loss={v['loss_final']:.4f}  E_last={v['E_last']:.2f}  dead={v['dead_final']}  util={v['utilization']}")
    b = experiment_b(epochs=epochs)
    print(f"\n[B] {b['name']}")
    for k, v in b["results"].items():
        print(f"  {k}: loss={v['loss_final']:.4f}  E_last={v['E_last']:.2f}  dead={v['dead_final']}  util={v['utilization']}")

    out = {
        "epochs": epochs,
        "wall_time_s": round(time.time() - t0, 2),
        "A": a,
        "B": b,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
