"""Smoke bench for causality-gated orth policy (PRD #10-51, 2026-06-15, round 89).

Compare 2 conditions on the same hard cases used in round 84-88:

- 2 conditions × 3 datasets × 3 orth λ ∈ {0.1, 1.0, 10.0}
- Conditions:
  (A) ecology_gated_orth=True (round 85, observational E-based)
  (B) ecology_gated_orth=True + causality_gated_orth=True (round 89)

Per cell we report:
- loss_final
- E_emp_last
- max_min_ratio_grad_last
- orth_fired (E-gate): bool
- causality_fired (per-expert grad): bool
- effective_lambda

Hypotheses:
- H1: causality gate fires in cells where E gate doesn't (per-expert
     imbalance > 10 even when E ≥ 0.5)
- H2: orth+causality never worse than orth alone (safe superset)
- H3: causality gate catches early collapse before E drops below 0.5

Run:
    .venv312/bin/python scripts/bench_causality_gated_orth.py --quick
    .venv312/bin/python scripts/bench_causality_gated_orth.py            # full
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
# Synthetic datasets (same as round 83/84/85/86/87/88)
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

def train_one(
    cell: FAMECfCCell,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    orth_lambda: float,
    use_causality: bool,
    lr: float = 1e-2,
) -> dict:
    """Train cell, collect E_emp, max_min_ratio_grad, gate-firing, loss."""
    params = [p for p in cell.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    cell.train()
    history = {
        "loss": [],
        "E_emp": [],
        "max_min_ratio_grad": [],
        "orth_fired": False,
        "causality_fired": False,
        "lambda_eff": orth_lambda,
    }
    for epoch in range(epochs):
        opt.zero_grad()
        h = torch.zeros(x.shape[0], cell.hidden_size)
        task_loss_acc = 0.0
        last_outs: list[torch.Tensor] = []
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            y_t = y[:, t, :]
            h_new, outs = cell.forward_with_aux(x_t, h, dt=1.0)
            task_loss_t = ((h_new - y_t) ** 2).mean()
            task_loss_acc = task_loss_acc + task_loss_t
            h = h_new
            last_outs = outs  # keep latest outs (same graph as task_loss)
        task_loss = task_loss_acc / x.shape[1]
        # Use compute_orth_loss_causality if enabled, else compute_orth_loss.
        # Use last_outs (same graph as task_loss) so the per-expert gradient
        # diagnostic can flow from task_loss to last_router_logits.
        if use_causality and cell.causality_gate is not None:
            orth_loss = cell.compute_orth_loss_causality(
                last_outs, user_lambda=orth_lambda, task_loss=task_loss,
            )
        else:
            orth_loss = cell.compute_orth_loss(last_outs, user_lambda=orth_lambda)
        total_loss = task_loss + orth_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        history["loss"].append(float(total_loss.item()))
        # Diagnostic for last epoch.
        cell.train()
        h2 = torch.zeros(x.shape[0], cell.hidden_size)
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            h2, _ = cell.forward_with_aux(x_t, h2, dt=1.0)
        h2_new, _ = cell.forward_with_aux(x[:, -1, :], h2, dt=1.0)
        fresh_task_loss = ((h2_new - torch.zeros_like(h2_new)) ** 2).mean()
        diag = cell.moe_ecology_diagnostic(
            B=orth_lambda, task_loss=fresh_task_loss, per_expert=True,
        )
        history["E_emp"].append(diag["E"] if isinstance(diag["E"], float) else diag["E"])
        history["max_min_ratio_grad"].append(diag.get("max_min_ratio", 1.0))
        # Gate states — use the post-training value (already intervened
        # by the end of this epoch).
        if cell.orth_gate is not None and cell.orth_gate.intervened:
            history["orth_fired"] = True
        if cell.causality_gate is not None and cell.causality_gate.intervened:
            history["causality_fired"] = True
        if cell.orth_gate is not None and cell.orth_gate.intervened:
            history["lambda_eff"] = cell.orth_gate.last_lambda_scale
        # For causality, also check the gate's "intervened" (sticky)
        # and the current cell.causality_gate.last_ratio.
    return history


def cell_factory(use_causality: bool, lambda_safe: float = 0.001):
    """Build cell with ecology_gated_orth (always on for fair A/B)."""
    return FAMECfCCell(
        input_size=1, hidden_size=8, n_experts=3, top_k=1,
        ecology_gated_orth=True,
        ecology_orth_lambda_safe=lambda_safe,
        causality_gated_orth=use_causality,
        causality_ratio_threshold=10.0,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results/bench_causality_gated_orth.json")
    args = p.parse_args()
    epochs = 2 if args.quick else 5
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out = {"epochs": epochs, "wall_time_s": 0.0, "runs": []}
    for orth_lambda in (0.1, 1.0, 10.0):
        run = {"orth_lambda": orth_lambda, "conditions": {}}
        for cond_name, use_c in [("orth_only", False), ("orth_causality", True)]:
            run["conditions"][cond_name] = {"datasets": {}}
            for ds_name, ds_fn in DATASETS.items():
                x, y = ds_fn(n_samples=32, seq_len=16, seed=0)
                torch.manual_seed(0)
                cell = cell_factory(use_c)
                h = train_one(
                    cell, x, y, epochs=epochs,
                    orth_lambda=orth_lambda, use_causality=use_c,
                )
                run["conditions"][cond_name]["datasets"][ds_name] = {
                    "loss_final": h["loss"][-1],
                    "E_emp_last": h["E_emp"][-1],
                    "max_min_ratio_grad": h["max_min_ratio_grad"][-1],
                    "orth_fired": h["orth_fired"],
                    "causality_fired": h["causality_fired"],
                    "lambda_eff": h["lambda_eff"],
                }
        # Pretty print.
        print(f"\n[λ={orth_lambda}]")
        for ds_name in DATASETS:
            r_orth = run["conditions"]["orth_only"]["datasets"][ds_name]
            r_cau = run["conditions"]["orth_causality"]["datasets"][ds_name]
            print(
                f"  {ds_name:11s}: orth loss={r_orth['loss_final']:.4f} (fired={r_orth['orth_fired']}) | "
                f"orth+cau loss={r_cau['loss_final']:.4f} (cau_fired={r_cau['causality_fired']}) | "
                f"ratio={r_orth['max_min_ratio_grad']:.2f}"
            )
        out["runs"].append(run)
    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
