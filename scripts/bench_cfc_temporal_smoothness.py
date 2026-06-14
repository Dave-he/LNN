"""Round 91 bench (PRD #10-53, response to arXiv:2606.07670 Li/Pal/Tan 2026).

Tests the claim that CfC cells produce smoother outputs in t than
equivalent MLPs because of the closed-form time-constant inductive bias.

Two models, ~matched parameter count:
- MLP(1 → 16 → 16 → 1) with ReLU, ~200 params
- CfCCell(1, 16) + Linear(16, 1) head, ~200 params

Test function: f(t) = sin(2π t) + 0.5 sin(10π t) + 0.1 noise
- Train on 64 sparse points in [0, 1]
- Evaluate on 256 dense points in [0, 1] (interpolation)
- Evaluate on 64 points in [1.0, 1.2] (extrapolation)

Metrics per model:
- MSE (interpolation)
- TV, l2_deriv, max_grad (interpolation smoothness)
- OOD-MSE (extrapolation)

5 seeds per model → mean ± std.

Run:
    .venv312/bin/python scripts/bench_cfc_temporal_smoothness.py --quick
    .venv312/bin/python scripts/bench_cfc_temporal_smoothness.py        # full
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
from lnn.core.smoothness_metrics import smoothness_summary


# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------

def target_fn(t: torch.Tensor) -> torch.Tensor:
    """f(t) = sin(2π t) + 0.5 sin(10π t) + small noise."""
    return torch.sin(2 * np.pi * t) + 0.5 * torch.sin(10 * np.pi * t)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Tiny MLP: 1 → 16 → 16 → 1, ReLU, ~200 params."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t.unsqueeze(-1) if t.dim() == 1 else t).squeeze(-1)


class CfCRegressor(nn.Module):
    """CfCCell(1, 16) + Linear(16, 1) head, single-step stateless.

    For each t, reset h=0 and run one CfC step. This is the apples-to-
    apples comparison with the MLP: both are pure functions of t.
    The CfC's smoothness property comes from the closed-form time-
    constant solution of the ODE (Hasani et al. 2021), not from state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cell = CfCCell(input_size=1, hidden_size=16, n_tau=1)
        self.head = nn.Linear(16, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """For each t value, run a stateless CfC step (h=0) and read head."""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        outs = []
        for ti in t.unbind(dim=-1):
            x_t = ti.reshape(1, 1)
            h0 = torch.zeros(1, self.cell.hidden_size)
            h_new = self.cell(x_t, h0, dt=1.0)
            outs.append(self.head(h_new))
        return torch.cat(outs, dim=-1).squeeze(0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_eval(
    model: nn.Module,
    n_train: int = 64,
    n_eval: int = 256,
    n_ood: int = 64,
    epochs: int = 100,
    lr: float = 1e-2,
    seed: int = 0,
) -> dict:
    """Train model on sparse t grid, return metrics on dense eval + OOD."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Sparse training grid (deterministic).
    t_train = torch.linspace(0, 1, n_train)
    y_train = target_fn(t_train)
    # Dense eval grid (interpolation).
    t_eval = torch.linspace(0, 1, n_eval)
    y_eval = target_fn(t_eval)
    # OOD grid (extrapolation).
    t_ood = torch.linspace(1.0, 1.2, n_ood)
    y_ood = target_fn(t_ood)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        y_pred = model(t_train)
        loss = ((y_pred - y_train) ** 2).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        # Interpolation: dense grid.
        y_pred_eval = model(t_eval)
        mse_eval = float(((y_pred_eval - y_eval) ** 2).mean().item())
        smooth_eval = smoothness_summary(y_pred_eval, dt=1.0 / (n_eval - 1))
        # Extrapolation: OOD grid.
        y_pred_ood = model(t_ood)
        mse_ood = float(((y_pred_ood - y_ood) ** 2).mean().item())

    return {
        "mse_eval": mse_eval,
        "ood_mse": mse_ood,
        "tv": smooth_eval["tv"],
        "l2_deriv": smooth_eval["l2_deriv"],
        "max_grad": smooth_eval["max_grad"],
        "n_params": sum(p.numel() for p in model.parameters()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default="results/bench_cfc_temporal_smoothness.json")
    args = p.parse_args()
    epochs = 30 if args.quick else 100
    n_seeds = args.seeds
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out = {
        "epochs": epochs,
        "n_seeds": n_seeds,
        "wall_time_s": 0.0,
        "models": {},
    }
    for name, factory in [("MLP", MLP), ("CfC", CfCRegressor)]:
        per_seed = []
        for seed in range(n_seeds):
            model = factory()
            res = train_and_eval(model, epochs=epochs, seed=seed)
            res["seed"] = seed
            per_seed.append(res)
        # Aggregate.
        keys = ["mse_eval", "ood_mse", "tv", "l2_deriv", "max_grad"]
        agg = {k: {"mean": float(np.mean([s[k] for s in per_seed])),
                   "std": float(np.std([s[k] for s in per_seed]))}
               for k in keys}
        agg["n_params"] = per_seed[0]["n_params"]
        agg["per_seed"] = per_seed
        out["models"][name] = agg

    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))

    # Pretty print.
    print(f"\n{'model':5s} | {'params':>6s} | {'mse_eval':>16s} | {'ood_mse':>16s} | {'tv':>12s} | {'l2_deriv':>12s} | {'max_grad':>12s}")
    print("-" * 100)
    for name in ("MLP", "CfC"):
        m = out["models"][name]
        ms = m["mse_eval"]; mos = m["ood_mse"]
        tv = m["tv"]; ld = m["l2_deriv"]; mg = m["max_grad"]
        def fmt(d: dict) -> str:
            return f"{d['mean']:.4f}±{d['std']:.4f}"
        print(
            f"{name:5s} | {m['n_params']:>6d} | {fmt(ms):>16s} | {fmt(mos):>16s} | "
            f"{fmt(tv):>12s} | {fmt(ld):>12s} | {fmt(mg):>12s}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
