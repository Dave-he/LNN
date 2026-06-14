"""Round 93 bench (PRD #10-55, response to arXiv:2605.27467 Thu/Oo/Supnithi 2026).

This is the **input-side temporal dropout** counterpart to round 92.

Key fix vs the first version: the model takes (t, y_masked) as a 2D
input feature. The masked y is the "observation" — the model must
interpolate through the zeros to predict the dense y. This matches
the paper's clinical scenario where the model sees a sparse,
gap-filled input stream and predicts the underlying function.

In round 92 (target-side), the model was fed the clean t grid and
trained to predict y. In round 93 (input-side), the model is fed
the OBSERVATIONS (t, y_masked) where y_masked has zeros at dropped
positions, and trained to predict the dense y.

Models: MLP, CfC, LSTM, GRU (4)
Dropout p: 0%, 10%, 20%, 40%, 60%, 80% (6)
Seeds: 3 per cell
Total: 72 cells

Per cell:
- mse_eval (dense grid)
- max_grad (smoothness, from round 91)
- mse_degradation_ratio (relative to p=0)

Run:
    .venv312/bin/python scripts/bench_cfc_input_dropout.py --quick
    .venv312/bin/python scripts/bench_cfc_input_dropout.py        # full
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
from lnn.core.smoothness_metrics import max_gradient
from lnn.core.temporal_dropout import input_dropout


# ---------------------------------------------------------------------------
# Target function (same as rounds 91, 92)
# ---------------------------------------------------------------------------

def target_fn(t: torch.Tensor) -> torch.Tensor:
    """f(t) = sin(2π t) + 0.5 sin(10π t)."""
    return torch.sin(2 * np.pi * t) + 0.5 * np.sin(10 * np.pi * t)


# ---------------------------------------------------------------------------
# Models — all take (T, 2) input: [t, y_masked]
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """MLP(2 → 16 → 16 → 1), ReLU, ~353 params. Per-position stateless."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, 2) — returns (T,) predictions."""
        return self.net(x).squeeze(-1)


class CfCRegressor(nn.Module):
    """CfCCell(2, 16) + Linear(16, 1) head, stateless (h=0 each t).

    Unlike round 92 (which fed (1,) features), the cell now sees a
    2D input: [t, y_masked]. This lets the cell use the y_masked
    observation directly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cell = CfCCell(input_size=2, hidden_size=16, n_tau=1)
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, 2) — returns (T,) predictions."""
        if x.dim() == 0:
            x = x.unsqueeze(0)
        outs = []
        for ti in x.unbind(dim=-2):  # iterate over T dim
            x_t = ti.reshape(1, 2)  # (1, 2)
            h0 = torch.zeros(1, self.cell.hidden_size)
            h_new = self.cell(x_t, h0, dt=1.0)
            outs.append(self.head(h_new))
        return torch.cat(outs, dim=-1).squeeze(0)


class LSTMSeq2Seq(nn.Module):
    """LSTM(2, 16) + Linear(16, 1) head, seq2seq."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, 2) — returns (T,) predictions."""
        x_seq = x.unsqueeze(0)  # (1, T, 2)
        out, _ = self.lstm(x_seq)
        return self.head(out).squeeze(0).squeeze(-1)


class GRUSeq2Seq(nn.Module):
    """GRU(2, 16) + Linear(16, 1) head, seq2seq."""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.unsqueeze(0)
        out, _ = self.gru(x_seq)
        return self.head(out).squeeze(0).squeeze(-1)


# ---------------------------------------------------------------------------
# Training: round 93 = input-side dropout, 2D input feature
# ---------------------------------------------------------------------------

def train_and_predict_dense_input_dropout(
    model: nn.Module,
    n_train: int = 64,
    n_eval: int = 256,
    epochs: int = 100,
    lr: float = 1e-2,
    dropout_p: float = 0.0,
    seed: int = 0,
) -> dict:
    """Train with INPUT-side dropout. 2D input (t, y_masked) → 1D output y."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    t_train = torch.linspace(0, 1, n_train)
    y_train = target_fn(t_train)
    t_eval = torch.linspace(0, 1, n_eval)
    y_eval = target_fn(t_eval)

    # Apply input-side dropout to y_train; the model gets (t, y_masked).
    _, y_input_masked = input_dropout(t_train, y_train, p=dropout_p, seed=seed)
    x_train = torch.stack([t_train, y_input_masked], dim=-1)  # (T, 2)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        y_pred_train = model(x_train)
        if y_pred_train.dim() == 0:
            y_pred_train = y_pred_train.unsqueeze(0)
        # Loss is against the FULL target (not the masked one).
        loss = ((y_pred_train - y_train) ** 2).mean()
        loss.backward()
        opt.step()

    # Eval on dense (t, 0) grid: at eval time, no observations, model
    # must extrapolate from the time alone. y_eval is the dense target.
    model.eval()
    with torch.no_grad():
        x_eval = torch.stack([t_eval, torch.zeros_like(t_eval)], dim=-1)
        y_pred_eval = model(x_eval)
    return {
        "y_pred_eval": y_pred_eval.detach(),
        "y_eval": y_eval,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="results/bench_cfc_input_dropout.json")
    args = p.parse_args()
    epochs = 30 if args.quick else 100
    n_seeds = args.seeds
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    factories = {
        "MLP": MLP,
        "CfC": CfCRegressor,
        "LSTM": LSTMSeq2Seq,
        "GRU": GRUSeq2Seq,
    }
    dropout_ps = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]

    t0 = time.time()
    out = {
        "epochs": epochs,
        "n_seeds": n_seeds,
        "wall_time_s": 0.0,
        "models": {},
    }
    for name, factory in factories.items():
        per_p: dict[float, list[dict]] = {p_p: [] for p_p in dropout_ps}
        # First pass: p=0 to get baseline.
        for seed in range(n_seeds):
            model = factory()
            res = train_and_predict_dense_input_dropout(
                model, epochs=epochs, dropout_p=0.0, seed=seed,
            )
            mse = float(((res["y_pred_eval"] - res["y_eval"]) ** 2).mean().item())
            mg = max_gradient(res["y_pred_eval"], dt=1.0 / 255)
            per_p[0.0].append({"mse": mse, "max_grad": mg})
        # Then for p > 0.
        for p_p in dropout_ps[1:]:
            for seed in range(n_seeds):
                model = factory()
                res = train_and_predict_dense_input_dropout(
                    model, epochs=epochs, dropout_p=p_p, seed=seed,
                )
                mse = float(((res["y_pred_eval"] - res["y_eval"]) ** 2).mean().item())
                per_p[p_p].append({"mse": mse})
        # Aggregate per p.
        agg = {}
        for p_p, lst in per_p.items():
            mses = [s["mse"] for s in lst]
            agg[str(p_p)] = {
                "mse_mean": float(np.mean(mses)),
                "mse_std": float(np.std(mses)),
            }
        # max_grad from p=0.
        mg_lst = [s["max_grad"] for s in per_p[0.0]]
        agg["max_grad_at_p0"] = {
            "mean": float(np.mean(mg_lst)),
            "std": float(np.std(mg_lst)),
        }
        # Degradation ratio.
        mse0 = agg["0.0"]["mse_mean"]
        for p_p in dropout_ps[1:]:
            msep = agg[str(p_p)]["mse_mean"]
            agg[str(p_p)]["degradation_ratio"] = msep / mse0 if mse0 > 0 else float("inf")
        agg["n_params"] = sum(p.numel() for p in factory().parameters())
        out["models"][name] = agg

    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))

    # Pretty print.
    print(f"\n{'model':5s} | {'params':>6s} | {'max_grad@0':>10s} | " +
          " | ".join(f"p={p}" for p in dropout_ps))
    print("-" * 100)
    for name in factories:
        m = out["models"][name]
        mg = m["max_grad_at_p0"]["mean"]
        cells = []
        for p_p in dropout_ps:
            mse = m[str(p_p)]["mse_mean"]
            cells.append(f"mse={mse:.4f}")
        print(f"{name:5s} | {m['n_params']:>6d} | {mg:>10.4f} | " + " | ".join(cells))
    print("\nDegradation ratios (mse@p / mse@0):")
    print(f"{'model':5s} | " + " | ".join(f"p={p}" for p in dropout_ps[1:]))
    for name in factories:
        m = out["models"][name]
        ratios = [f"{m[str(p)]['degradation_ratio']:>5.2f}x" for p in dropout_ps[1:]]
        print(f"{name:5s} | " + " | ".join(ratios))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
