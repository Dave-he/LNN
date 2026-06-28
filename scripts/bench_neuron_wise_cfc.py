#!/usr/bin/env python3
"""Benchmark for NeuronWiseCfCCell (round 263).

Tests whether **per-neuron dynamics** (arXiv:2606.21295 TND analog)
improve over the shared-τ CfC baseline. Closes the structural gap
identified in r257's bridge document after the basin-axis rounds
(r257-r262): we now operate on the *neuron axis* — each neuron has
its own τ, α, and learned sparse neighborhood.

Modes (5 total):
  * baseline                 — vanilla CfC (no branches, shared τ)
  * neuronwise_d03           — density 0.3, key treatment (TND-inspired)
  * neuronwise_d05           — density 0.5, moderate
  * neuronwise_d10           — density 1.0, fully connected
  * neuronwise_d03_shared    — density 0.3, but shared τ (control)

Hypotheses (PRD #10-100):

  H1: NeuronWiseCfCCell (d=0.3) beats plain CfC on toy_sin and
      structured because per-neuron τ allows heterogeneous time-
      scales to emerge from topology.
  H2: The learned neighborhood mask becomes ASYMMETRIC (avg
      off-diagonal density > 0.3 of total off-diag pairs are
      asymmetric).
  H3: Per-neuron τ values span a wide range after training
      (std(τ) > 0.3 × mean(τ)) — neurons develop heterogeneous
      time-scales.
  H4: NeuronWiseCfCCell is a strict superset of single-τ CfC:
      with density=1.0 and uniform τ, it degenerates to a recurrent
      network equivalent to CfC's gate × tanh form.

Bench config:
  * 3 datasets: toy_sin, structured, random (same as r248-r262)
  * hidden_size = 16
  * 100 epochs, lr=1e-2, batch=16, 2 seeds
  * Output: JSON with test_mse, neighbor diagnostics, tau stats.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lnn.core.cfc import CfCCell  # noqa: E402
from lnn.core.neuron_wise_cfc import NeuronWiseCfCCell  # noqa: E402


# ---------------------------------------------------------------------------
# Toy data generators (match r257-262 bench protocol)
# ---------------------------------------------------------------------------
def make_toy_sin(T: int = 64, n_samples: int = 256, d_in: int = 1, seed: int = 0):
    """y(t) = sin(2π t / T); predict next value."""
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, T + 1).unsqueeze(0).repeat(n_samples, 1)  # (N, T+1)
    y = torch.sin(2 * math.pi * t)
    x = y[:, :-1].unsqueeze(-1)  # (N, T, 1)
    y_target = y[:, 1:].unsqueeze(-1)  # (N, T, 1)
    # Ensure d_in matches request.
    if d_in > 1:
        # Add extra channels of low-amplitude noise (irrelevant).
        extra = torch.randn(n_samples, T, d_in - 1, generator=g) * 0.01
        x = torch.cat([x, extra], dim=-1)
    return x, y_target


def make_structured(T: int = 64, n_samples: int = 256, d_in: int = 1, seed: int = 0):
    """Step function with 4 levels; predict next value."""
    g = torch.Generator().manual_seed(seed)
    n_segments = 4
    seg_len = (T + 1) // n_segments
    levels = torch.tensor([0.0, 1.0, -0.5, 0.7])
    y = torch.zeros(n_samples, T + 1)
    for i in range(n_segments):
        start = i * seg_len
        end = (i + 1) * seg_len if i < n_segments - 1 else T + 1
        y[:, start:end] = levels[i % len(levels)]
    # Add tiny noise.
    y = y + torch.randn(n_samples, T + 1, generator=g) * 0.01
    x = y[:, :-1].unsqueeze(-1)
    y_target = y[:, 1:].unsqueeze(-1)
    if d_in > 1:
        extra = torch.randn(n_samples, T, d_in - 1, generator=g) * 0.01
        x = torch.cat([x, extra], dim=-1)
    return x, y_target


def make_random(T: int = 64, n_samples: int = 256, d_in: int = 1, seed: int = 0):
    """Pure noise; predict next value (impossible — serves as ceiling)."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(n_samples, T + 1, generator=g)
    x = y[:, :-1].unsqueeze(-1)
    y_target = y[:, 1:].unsqueeze(-1)
    if d_in > 1:
        extra = torch.randn(n_samples, T, d_in - 1, generator=g) * 0.01
        x = torch.cat([x, extra], dim=-1)
    return x, y_target


DATA_FACTORIES = {
    "toy_sin": make_toy_sin,
    "structured": make_structured,
    "random": make_random,
}


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------
class WrappedCfC(nn.Module):
    """Wrap CfCCell so that output is (B, T, hidden_size).

    CfCCell.forward(x_t, h, dt=1.0) returns just h. We loop over T.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = CfCCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h)
            outputs.append(h)
        return torch.stack(outputs, dim=1), h


class WrappedNeuronWise(nn.Module):
    """Wrap NeuronWiseCfCCell so that output is (B, T, hidden_size)."""

    def __init__(self, input_size, hidden_size, density=0.3, shared_tau=False):
        super().__init__()
        self.cell = NeuronWiseCfCCell(
            input_size=input_size, hidden_size=hidden_size, density=density
        )
        self.hidden_size = hidden_size
        self.shared_tau = shared_tau

    def forward(self, x):
        # For shared-tau control, force all neurons to use the same τ by
        # zeroing the per-neuron tau delta after init.
        if self.shared_tau:
            with torch.no_grad():
                # Reset all tau logits to the same value (mean of current).
                mean_logit = self.cell.tau_per_neuron.mean().item()
                self.cell.tau_per_neuron.fill_(mean_logit)
        return self.cell(x)


class ReadoutHead(nn.Module):
    """Linear readout from hidden_size to 1 (regression target)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, h):
        return self.head(h)


class SeqModel(nn.Module):
    """Encoder + readout."""

    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.head = ReadoutHead(hidden_size)

    def forward(self, x):
        out, _ = self.encoder(x)
        return self.head(out)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def train_one(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> dict:
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = x_train.shape[0]
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        x_tr = x_train[perm]
        y_tr = y_train[perm]
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            xb = x_tr[i : i + batch_size].to(device)
            yb = y_tr[i : i + batch_size].to(device)
            pred = model(xb)
            loss = (pred - yb).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
    # Eval
    model.eval()
    with torch.no_grad():
        pred = model(x_eval.to(device))
        test_mse = float((pred - y_eval.to(device)).pow(2).mean().item())
    return {"test_mse": test_mse, "train_loss_last": losses[-1]}


# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------
MODES = {
    "baseline": dict(kind="baseline"),
    "neuronwise_d03": dict(kind="neuronwise", density=0.3, shared_tau=False),
    "neuronwise_d05": dict(kind="neuronwise", density=0.5, shared_tau=False),
    "neuronwise_d10": dict(kind="neuronwise", density=1.0, shared_tau=False),
    "neuronwise_d03_shared": dict(kind="neuronwise", density=0.3, shared_tau=True),
}


def make_model(mode_cfg: dict, d_in: int, hidden_size: int) -> nn.Module:
    if mode_cfg["kind"] == "baseline":
        enc = WrappedCfC(d_in, hidden_size)
    else:
        enc = WrappedNeuronWise(
            d_in, hidden_size,
            density=mode_cfg["density"],
            shared_tau=mode_cfg.get("shared_tau", False),
        )
    return SeqModel(enc, hidden_size)


def collect_diagnostics(model: SeqModel) -> dict:
    enc = model.encoder
    if isinstance(enc, WrappedNeuronWise):
        cell = enc.cell
        tau = cell.get_tau().detach().cpu().tolist()
        mask = cell.get_neighborhood_mask().detach().cpu().tolist()
        alpha = cell.get_alpha().detach().cpu().tolist()
        return {
            "tau_mean": float(sum(tau) / len(tau)),
            "tau_std": float(torch.tensor(tau).std().item()),
            "tau_min": float(min(tau)),
            "tau_max": float(max(tau)),
            "alpha_mean": float(sum(alpha) / len(alpha)),
            "alpha_std": float(torch.tensor(alpha).std().item()),
            "neighborhood_density": cell.neighborhood_density(),
            "neighborhood_asymmetry": cell.neighborhood_asymmetry(),
        }
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--d-in", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--datasets", nargs="+", default=["toy_sin", "structured", "random"])
    parser.add_argument("--out", type=str, default="analysis/neuron_wise_cfc_bench.json")
    parser.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bench] device={device}")

    results = {
        "config": {
            "epochs": args.epochs, "hidden": args.hidden, "lr": args.lr,
            "batch": args.batch, "T": args.T, "n_samples": args.n_samples,
            "d_in": args.d_in, "seeds": args.seeds, "datasets": args.datasets,
            "modes": args.modes,
        },
        "cells": [],
    }

    for mode_name in args.modes:
        mode_cfg = MODES[mode_name]
        for ds_name in args.datasets:
            for seed in args.seeds:
                t0 = time.time()
                torch.manual_seed(seed)
                x, y = DATA_FACTORIES[ds_name](
                    T=args.T, n_samples=args.n_samples, d_in=args.d_in, seed=seed
                )
                # 80/20 split.
                N = x.shape[0]
                split = int(0.8 * N)
                x_tr, x_ev = x[:split], x[split:]
                y_tr, y_ev = y[:split], y[split:]

                model = make_model(mode_cfg, args.d_in, args.hidden)
                out = train_one(
                    model, x_tr, y_tr, x_ev, y_ev,
                    epochs=args.epochs, lr=args.lr, batch_size=args.batch,
                    device=device,
                )
                diag = collect_diagnostics(model)
                cell_result = {
                    "mode": mode_name,
                    "dataset": ds_name,
                    "seed": seed,
                    "test_mse": out["test_mse"],
                    "train_loss_last": out["train_loss_last"],
                    "elapsed_sec": round(time.time() - t0, 2),
                    "diagnostics": diag,
                }
                results["cells"].append(cell_result)
                print(
                    f"[bench] mode={mode_name:24s} ds={ds_name:12s} "
                    f"seed={seed} test_mse={out['test_mse']:.6f} "
                    f"({cell_result['elapsed_sec']}s)"
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {out_path}")

    # Quick summary by mode × dataset (mean over seeds).
    summary = {}
    for cell in results["cells"]:
        key = (cell["mode"], cell["dataset"])
        summary.setdefault(key, []).append(cell["test_mse"])
    print("\n[bench] Mean test_mse by mode × dataset:")
    print(f"{'mode':24s} | {'toy_sin':>10s} | {'structured':>10s} | {'random':>10s} | mean")
    for mode_name in args.modes:
        cols = []
        for ds in args.datasets:
            vals = summary.get((mode_name, ds), [])
            cols.append(sum(vals) / len(vals) if vals else float("nan"))
        mean = sum(c for c in cols if not math.isnan(c)) / max(len(cols), 1)
        print(
            f"{mode_name:24s} | {cols[0] if len(cols) > 0 else float('nan'):>10.6f} | "
            f"{cols[1] if len(cols) > 1 else float('nan'):>10.6f} | "
            f"{cols[2] if len(cols) > 2 else float('nan'):>10.6f} | {mean:.6f}"
        )


if __name__ == "__main__":
    main()
