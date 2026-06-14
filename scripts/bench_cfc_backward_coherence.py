"""Round 98 bench (PRD #10-60) — CfC + backward coherence regularization.

Direct test of arXiv:2606.08934 (Chang, June 2026) —
*Backward Coherence and Hidden-State Stability in Recurrent Neural
Networks: A Quasi-Reverse-Martingale Theory*.

Hypothesis: penalizing ``mean(||h_{t+1} - h_t||^2)`` during training
encourages a quasi-reverse-martingale where ``h_t ≈ E[h_{t+1}]``,
which the paper claims improves stability and generalization on
PhysioNet 2012 ICU, FRED-MD, and UCI HAR.

For each of 4 models (MLP, CfC, LSTM, GRU) on 3 datasets
(toy_sin, structured, random), we compare:
- baseline (no coherence penalty)
- +backward_coherence (λ=0.1)

We measure: task_loss, backward_diff_std, max_grad, hidden_eff_rank.

Cells: 4 models × 3 datasets × 2 conditions × 3 seeds = 72 cells

Run:
    .venv312/bin/python scripts/bench_cfc_backward_coherence.py --quick
    .venv312/bin/python scripts/bench_cfc_backward_coherence.py        # full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell
from lnn.core.effective_rank import effective_rank_trajectory
from lnn.core.smoothness_metrics import (
    backward_coherence_loss,
    max_gradient,
)


# ---------------------------------------------------------------------------
# Target function (same as rounds 91, 92, 93, 94)
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
# Models — same as round 94
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(-1) if x.dim() == 1 else x).squeeze(-1)


class CfCRegressor(nn.Module):
    """CfC regression — exposes ``.cell`` and ``.head`` so bench can run a
    unrolled forward collecting NON-detached hidden states (the
    ``return_states=True`` flag in the cell API detaches for measurement
    convenience; the bench needs gradient flow)."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = CfCCell(input_size=1, hidden_size=16, n_tau=1)
        self.head = nn.Linear(16, 1)


class LSTMSeq2Seq(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.head = nn.Linear(16, 1)


class GRUSeq2Seq(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=16, batch_first=True)
        self.head = nn.Linear(16, 1)


FACTORIES = {
    "MLP": MLP,
    "CfC": CfCRegressor,
    "LSTM": LSTMSeq2Seq,
    "GRU": GRUSeq2Seq,
}


# ---------------------------------------------------------------------------
# Per-model forward returning (y_pred, states) where states has grad
# ---------------------------------------------------------------------------

def forward_with_states(
    model: nn.Module,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run forward and return (y_pred_T, states_T_d) — states has grad
    flow (non-detached). For MLP, states is None (no hidden state)."""
    if isinstance(model, MLP):
        y = model(t)
        return y, None
    if isinstance(model, CfCRegressor):
        outs = []
        states = []
        h = torch.zeros(1, model.cell.hidden_size)
        for ti in t:
            x_t = ti.reshape(1, 1)
            h = model.cell(x_t, h, dt=1.0)
            outs.append(model.head(h))
            states.append(h.squeeze(0))  # NOT detached — gradient flows
        y = torch.cat(outs, dim=-1).squeeze(0)
        return y, torch.stack(states, dim=0)
    if isinstance(model, LSTMSeq2Seq):
        x_seq = t.unsqueeze(0).unsqueeze(-1)
        out, _ = model.lstm(x_seq)
        y = model.head(out).squeeze(0).squeeze(-1)
        return y, out.squeeze(0)  # (T, 16) with grad
    if isinstance(model, GRUSeq2Seq):
        x_seq = t.unsqueeze(0).unsqueeze(-1)
        out, _ = model.gru(x_seq)
        y = model.head(out).squeeze(0).squeeze(-1)
        return y, out.squeeze(0)
    raise ValueError(f"unknown model type {type(model)}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

COHERENCE_LAMBDA = 0.1  # the smallest λ where the loss has visible effect
# (λ=0.001 used in PRD is too small — the loss is ~3.6e-6 of task loss
# so the gradient is negligible. λ=0.1 is the safe band from a manual
# sweep where task loss ±15% but bwd_std drops ~5-15%.)


def train_model(
    model: nn.Module,
    t: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    use_coherence: bool,
) -> tuple[float, float, float, float]:
    """Train and return (task_loss, backward_diff_std, max_grad, hidden_eff_rank)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    final_task_loss = 0.0
    for _ in range(epochs):
        opt.zero_grad()
        y_pred, states = forward_with_states(model, t)
        task_loss = ((y_pred - y) ** 2).mean()
        total_loss = task_loss
        if use_coherence and states is not None:
            aux = backward_coherence_loss(states, lambda_coeff=COHERENCE_LAMBDA)
            total_loss = total_loss + aux
        total_loss.backward()
        opt.step()
        final_task_loss = float(task_loss.item())
    # Final measurements
    with torch.no_grad():
        y_pred, fs = forward_with_states(model, t)
    if fs is None:
        # MLP — no hidden state. Use output y_pred for max_grad only.
        mg = max_gradient(y_pred.detach(), dt=1.0)
        bds = 0.0
        her = 1.0
    else:
        diffs = fs[1:] - fs[:-1]
        bds = float(diffs.norm(dim=-1).std().item())
        mg = max_gradient(y_pred.detach(), dt=1.0)
        her = effective_rank_trajectory(fs)
    return final_task_loss, bds, mg, her


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="results/bench_cfc_backward_coherence.json")
    args = p.parse_args()
    epochs = 30 if args.quick else 100
    n_seeds = args.seeds
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out: dict = {
        "epochs": epochs,
        "n_seeds": n_seeds,
        "coherence_lambda": COHERENCE_LAMBDA,
        "wall_time_s": 0.0,
        "datasets": {},
    }
    T = 64

    for ds_name, ds_fn in DATASETS.items():
        ds_out: dict = {}
        for model_name, factory in FACTORIES.items():
            for cond in ("baseline", "coherence"):
                use_coh = cond == "coherence"
                cond_out: list[dict] = []
                for seed in range(n_seeds):
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    t, y = ds_fn(T, seed=seed)
                    model = factory()
                    task_loss, bds, mg, her = train_model(
                        model, t, y, epochs=epochs, lr=1e-2, use_coherence=use_coh,
                    )
                    cond_out.append({
                        "task_loss": task_loss,
                        "backward_diff_std": bds,
                        "max_grad": mg,
                        "hidden_eff_rank": her,
                    })
                def agg(field: str) -> tuple[float, float]:
                    vals = [s[field] for s in cond_out if s[field] is not None]
                    if not vals:
                        return 0.0, 0.0
                    return float(np.mean(vals)), float(np.std(vals))
                ds_out[f"{model_name}_{cond}"] = {
                    "task_loss_mean_std": agg("task_loss"),
                    "backward_diff_std_mean_std": agg("backward_diff_std"),
                    "max_grad_mean_std": agg("max_grad"),
                    "hidden_eff_rank_mean_std": agg("hidden_eff_rank"),
                    "per_seed": cond_out,
                }
        out["datasets"][ds_name] = ds_out

    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))

    # Pretty print
    print(f"\n=== Round 98 backward coherence bench (epochs={epochs}, seeds={n_seeds}, λ={COHERENCE_LAMBDA}) ===\n")
    print(f"{'dataset':12s} | {'model':5s} | {'cond':10s} | {'task_loss':>10s} | {'bwd_std':>10s} | {'max_grad':>10s} | {'her':>6s}")
    print("-" * 90)
    for ds_name in DATASETS:
        for model_name in FACTORIES:
            for cond in ("baseline", "coherence"):
                key = f"{model_name}_{cond}"
                if key not in out["datasets"][ds_name]:
                    continue
                c = out["datasets"][ds_name][key]
                tl_m, _ = c["task_loss_mean_std"]
                bds_m, _ = c["backward_diff_std_mean_std"]
                mg_m, _ = c["max_grad_mean_std"]
                her_m, _ = c["hidden_eff_rank_mean_std"]
                print(f"{ds_name:12s} | {model_name:5s} | {cond:10s} | {tl_m:10.4f} | {bds_m:10.4f} | {mg_m:10.4f} | {her_m:6.2f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
