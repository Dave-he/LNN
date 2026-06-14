"""Round 94 bench (PRD #10-56, response to arXiv:2606.00243 Williams/Payeur/Lajoie 2026).

Tests the paper's prediction that locality-constrained learning rules
find low-rank solutions. We test whether CfC (which has a smoothness
prior, round 91) finds lower effective rank than MLP/LSTM/GRU when
trained on the same 1D function-fitting task.

For each of 4 models, we measure:
- weight_eff_rank: mean effective rank of the trained weight matrices
- hidden_eff_rank: effective rank of the (T, d) hidden-state trajectory
                  on the dense eval grid

Models: MLP, CfC, LSTM, GRU (4)
Seeds: 3
Total: 12 cells

Run:
    .venv312/bin/python scripts/bench_cfc_effective_rank.py --quick
    .venv312/bin/python scripts/bench_cfc_effective_rank.py        # full
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
from lnn.core.effective_rank import (
    effective_rank,
    effective_rank_trajectory,
)


# ---------------------------------------------------------------------------
# Target function (same as rounds 91, 92, 93)
# ---------------------------------------------------------------------------

def target_fn(t: torch.Tensor) -> torch.Tensor:
    """f(t) = sin(2π t) + 0.5 sin(10π t)."""
    return torch.sin(2 * np.pi * t) + 0.5 * np.sin(10 * np.pi * t)


# ---------------------------------------------------------------------------
# Models — same as rounds 92, 93 (1D input)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """MLP(1 → 16 → 16 → 1), ReLU, ~321 params."""

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
    """CfCCell(1, 16) + Linear(16, 1) head, stateless (h=0 each t)."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = CfCCell(input_size=1, hidden_size=16, n_tau=1)
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0:
            x = x.unsqueeze(0)
        outs = []
        for ti in x.unbind(dim=-1):
            x_t = ti.reshape(1, 1)
            h0 = torch.zeros(1, self.cell.hidden_size)
            h_new = self.cell(x_t, h0, dt=1.0)
            outs.append(self.head(h_new))
        return torch.cat(outs, dim=-1).squeeze(0)


class LSTMSeq2Seq(nn.Module):
    """LSTM(1, 16) + Linear(16, 1) head, seq2seq."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.unsqueeze(0).unsqueeze(-1)
        out, _ = self.lstm(x_seq)
        return self.head(out).squeeze(0).squeeze(-1)


class GRUSeq2Seq(nn.Module):
    """GRU(1, 16) + Linear(16, 1) head, seq2seq."""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.unsqueeze(0).unsqueeze(-1)
        out, _ = self.gru(x_seq)
        return self.head(out).squeeze(0).squeeze(-1)


# ---------------------------------------------------------------------------
# Hidden-state collection utilities
# ---------------------------------------------------------------------------

def collect_mlp_hidden(model: MLP, x: torch.Tensor) -> torch.Tensor:
    """Return (T, hidden_dim) of the activations after the first ReLU.

    We use the activations of the first hidden layer as a proxy for
    the model's "hidden state".
    """
    activations = []

    def hook(_m: nn.Module, _i: torch.Tensor, o: torch.Tensor) -> None:
        activations.append(o.detach().clone())

    h = model.net[1].register_forward_hook(hook)  # ReLU after first Linear
    with torch.no_grad():
        _ = model(x)
    h.remove()
    # x: (T,) → unsqueeze to (T, 1) → out of Linear: (T, 16) → ReLU: (T, 16)
    return activations[0]  # (T, 16)


def collect_cfc_hidden(model: CfCRegressor, x: torch.Tensor) -> torch.Tensor:
    """Return (T, hidden_dim) of CfC stateless hidden states."""
    states = []
    if x.dim() == 0:
        x = x.unsqueeze(0)
    with torch.no_grad():
        for ti in x.unbind(dim=-1):
            x_t = ti.reshape(1, 1)
            h0 = torch.zeros(1, model.cell.hidden_size)
            h_new = model.cell(x_t, h0, dt=1.0)
            states.append(h_new.detach().clone().squeeze(0))
    return torch.stack(states, dim=0)  # (T, hidden_dim)


def collect_lstm_hidden(model: LSTMSeq2Seq, x: torch.Tensor) -> torch.Tensor:
    """Return (T, hidden_dim) of LSTM hidden states (output of LSTM, not the cell state)."""
    x_seq = x.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    with torch.no_grad():
        out, _ = model.lstm(x_seq)
    return out.squeeze(0)  # (T, 16)


def collect_gru_hidden(model: GRUSeq2Seq, x: torch.Tensor) -> torch.Tensor:
    """Return (T, hidden_dim) of GRU hidden states."""
    x_seq = x.unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        out, _ = model.gru(x_seq)
    return out.squeeze(0)  # (T, 16)


# ---------------------------------------------------------------------------
# Weight collection utilities
# ---------------------------------------------------------------------------

def collect_mlp_weights(model: MLP) -> list[torch.Tensor]:
    return [m.weight for m in model.net if isinstance(m, nn.Linear)]


def collect_cfc_weights(model: CfCRegressor) -> list[torch.Tensor]:
    """Return the 2D weight matrices of the CfC cell plus the head.

    For CfC we collect the major learnable matrices: f_gate (gate
    weights), W_tau (time-constant), the input-to-hidden mapping
    that's parameterized, and the head.
    """
    out = [model.head.weight]
    for name, param in model.cell.named_parameters():
        if param.dim() == 2:
            out.append(param)
    return out


def collect_lstm_weights(model: LSTMSeq2Seq) -> list[torch.Tensor]:
    return [
        model.lstm.weight_ih_l0,  # (4*hidden, input)
        model.lstm.weight_hh_l0,  # (4*hidden, hidden)
        model.head.weight,
    ]


def collect_gru_weights(model: GRUSeq2Seq) -> list[torch.Tensor]:
    return [
        model.gru.weight_ih_l0,  # (3*hidden, input)
        model.gru.weight_hh_l0,  # (3*hidden, hidden)
        model.head.weight,
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    n_train: int = 64,
    epochs: int = 100,
    lr: float = 1e-2,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    t_train = torch.linspace(0, 1, n_train)
    y_train = target_fn(t_train)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        y_pred = model(t_train)
        if y_pred.dim() == 0:
            y_pred = y_pred.unsqueeze(0)
        loss = ((y_pred - y_train) ** 2).mean()
        loss.backward()
        opt.step()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="results/bench_cfc_effective_rank.json")
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
    weight_collectors = {
        "MLP": collect_mlp_weights,
        "CfC": collect_cfc_weights,
        "LSTM": collect_lstm_weights,
        "GRU": collect_gru_weights,
    }
    hidden_collectors = {
        "MLP": collect_mlp_hidden,
        "CfC": collect_cfc_hidden,
        "LSTM": collect_lstm_hidden,
        "GRU": collect_gru_hidden,
    }

    t0 = time.time()
    out = {
        "epochs": epochs,
        "n_seeds": n_seeds,
        "wall_time_s": 0.0,
        "models": {},
    }

    n_eval = 256
    t_eval = torch.linspace(0, 1, n_eval)

    for name, factory in factories.items():
        per_seed: list[dict] = []
        for seed in range(n_seeds):
            model = factory()
            train_model(model, epochs=epochs, seed=seed)

            # Collect weights and hidden states on dense eval.
            weights = weight_collectors[name](model)
            states = hidden_collectors[name](model, t_eval)

            per_weight = [effective_rank(W) for W in weights]
            hidden_er = effective_rank_trajectory(states)

            # Compute mse for sanity.
            with torch.no_grad():
                y_pred = model(t_eval)
                if y_pred.dim() == 0:
                    y_pred = y_pred.unsqueeze(0)
            y_eval = target_fn(t_eval)
            mse = float(((y_pred - y_eval) ** 2).mean().item())

            per_seed.append({
                "mse": mse,
                "per_weight_eff_rank": per_weight,
                "mean_weight_eff_rank": float(np.mean(per_weight)),
                "hidden_eff_rank": hidden_er,
                "n_weight_matrices": len(weights),
                "hidden_dim": states.shape[-1],
            })

        # Aggregate.
        weight_ers = [s["mean_weight_eff_rank"] for s in per_seed]
        hidden_ers = [s["hidden_eff_rank"] for s in per_seed]
        mses = [s["mse"] for s in per_seed]
        out["models"][name] = {
            "n_params": sum(p.numel() for p in factory().parameters()),
            "mean_weight_eff_rank_mean": float(np.mean(weight_ers)),
            "mean_weight_eff_rank_std": float(np.std(weight_ers)),
            "hidden_eff_rank_mean": float(np.mean(hidden_ers)),
            "hidden_eff_rank_std": float(np.std(hidden_ers)),
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "per_seed": per_seed,
        }

    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))

    # Pretty print.
    print(f"\n{'model':5s} | {'params':>6s} | {'mse':>8s} | {'weight_eff_rank':>17s} | {'hidden_eff_rank':>16s}")
    print("-" * 80)
    for name in factories:
        m = out["models"][name]
        wstr = f"{m['mean_weight_eff_rank_mean']:>7.3f} ± {m['mean_weight_eff_rank_std']:>5.3f}"
        hstr = f"{m['hidden_eff_rank_mean']:>6.3f} ± {m['hidden_eff_rank_std']:>5.3f}"
        msestr = f"{m['mse_mean']:>6.4f}"
        print(f"{name:5s} | {m['n_params']:>6d} | {msestr:>8s} | {wstr:>17s} | {hstr:>16s}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
