#!/usr/bin/env python3
"""Benchmark for STENeuronWiseCfCCell (round 265).

Tests whether the **straight-through estimator (STE)** combines
the best of r263 (hard top-k, true sparsity) and r264 (soft
attention, fully learnable but soft mixing) to beat BOTH.

  - Forward: hard top-k (binary mask, true sparsity like r263)
  - Backward: soft sigmoid (gradients flow to neighbor_logits
    like r264)

Modes (5 total):
  * r263_baseline    — r263 NeuronWiseCfCCell density=0.3 (non-learnable)
  * ste_cold         — STE, τ_ste=0.1 (sharp backward gradient)
  * ste_default      — STE, τ_ste=1.0 (moderate)
  * ste_warm         — STE, τ_ste=5.0 (soft backward gradient)
  * ste_no_init      — STE, τ_ste=1.0, neighbor_logits init=0 (control)

Hypotheses (PRD #10-102):

  H1: STE beats r263 on at least one dataset.
  H2: STE beats r264 (soft attention) on at least one dataset.
  H3: neighbor_logits become structured (std > 0.05 after training).
  H4: STE is a strict superset of r263 (τ_ste → 0) and r264
      (τ_ste → ∞).

Bench config:
  * 3 datasets: toy_sin, structured, random
  * hidden_size = 16
  * 100 epochs, lr=1e-2, batch=16, 2 seeds
  * Loss: MSE only (no auxiliary L1)
  * Output: JSON with test_mse, neighbor_logits stats, mask entropy.
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

from lnn.core.neuron_wise_cfc import NeuronWiseCfCCell  # noqa: E402
from lnn.core.ste_neuron_wise_cfc import STENeuronWiseCfCCell  # noqa: E402


# ---------------------------------------------------------------------------
# Toy data generators (match r263/264 bench protocol)
# ---------------------------------------------------------------------------
def make_toy_sin(T: int = 64, n_samples: int = 256, d_in: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, T + 1).unsqueeze(0).repeat(n_samples, 1)
    y = torch.sin(2 * math.pi * t)
    x = y[:, :-1].unsqueeze(-1)
    y_target = y[:, 1:].unsqueeze(-1)
    if d_in > 1:
        extra = torch.randn(n_samples, T, d_in - 1, generator=g) * 0.01
        x = torch.cat([x, extra], dim=-1)
    return x, y_target


def make_structured(T: int = 64, n_samples: int = 256, d_in: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    n_segments = 4
    seg_len = (T + 1) // n_segments
    levels = torch.tensor([0.0, 1.0, -0.5, 0.7])
    y = torch.zeros(n_samples, T + 1)
    for i in range(n_segments):
        start = i * seg_len
        end = (i + 1) * seg_len if i < n_segments - 1 else T + 1
        y[:, start:end] = levels[i % len(levels)]
    y = y + torch.randn(n_samples, T + 1, generator=g) * 0.01
    x = y[:, :-1].unsqueeze(-1)
    y_target = y[:, 1:].unsqueeze(-1)
    if d_in > 1:
        extra = torch.randn(n_samples, T, d_in - 1, generator=g) * 0.01
        x = torch.cat([x, extra], dim=-1)
    return x, y_target


def make_random(T: int = 64, n_samples: int = 256, d_in: int = 1, seed: int = 0):
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
class WrappedNeuronWise(nn.Module):
    def __init__(self, input_size, hidden_size, density=0.3):
        super().__init__()
        self.cell = NeuronWiseCfCCell(
            input_size=input_size, hidden_size=hidden_size, density=density
        )
        self.hidden_size = hidden_size

    def forward(self, x):
        return self.cell(x)


class WrappedSTE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        density=0.3,
        ste_temperature=1.0,
        zero_init=False,
    ):
        super().__init__()
        self.cell = STENeuronWiseCfCCell(
            input_size=input_size,
            hidden_size=hidden_size,
            density=density,
            ste_temperature=ste_temperature,
        )
        if zero_init:
            with torch.no_grad():
                self.cell.neighbor_logits.zero_()
        self.hidden_size = hidden_size

    def forward(self, x):
        return self.cell(x)


class ReadoutHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, h):
        return self.head(h)


class SeqModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.head = ReadoutHead(hidden_size)

    def forward(self, x):
        out, _ = self.encoder(x)
        return self.head(out)


# ---------------------------------------------------------------------------
# Training
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
    for _ in range(epochs):
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
    model.eval()
    with torch.no_grad():
        pred = model(x_eval.to(device))
        test_mse = float((pred - y_eval.to(device)).pow(2).mean().item())
    return {"test_mse": test_mse, "train_loss_last": losses[-1]}


# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------
MODES = {
    "r263_baseline": dict(kind="neuronwise", density=0.3),
    "ste_cold": dict(kind="ste", density=0.3, ste_temperature=0.1, zero_init=False),
    "ste_default": dict(kind="ste", density=0.3, ste_temperature=1.0, zero_init=False),
    "ste_warm": dict(kind="ste", density=0.3, ste_temperature=5.0, zero_init=False),
    "ste_no_init": dict(kind="ste", density=0.3, ste_temperature=1.0, zero_init=True),
}


def make_model(mode_cfg: dict, d_in: int, hidden_size: int) -> nn.Module:
    if mode_cfg["kind"] == "neuronwise":
        enc = WrappedNeuronWise(d_in, hidden_size, density=mode_cfg["density"])
    else:
        enc = WrappedSTE(
            d_in, hidden_size,
            density=mode_cfg["density"],
            ste_temperature=mode_cfg["ste_temperature"],
            zero_init=mode_cfg["zero_init"],
        )
    return SeqModel(enc, hidden_size)


def collect_diagnostics(model: SeqModel) -> dict:
    enc = model.encoder
    cell = enc.cell
    tau = cell.get_tau().detach().cpu().tolist()
    alpha = cell.get_alpha().detach().cpu().tolist()
    out = {
        "tau_mean": float(sum(tau) / len(tau)),
        "tau_std": float(torch.tensor(tau).std().item()),
        "tau_min": float(min(tau)),
        "tau_max": float(max(tau)),
        "alpha_mean": float(sum(alpha) / len(alpha)),
        "alpha_std": float(torch.tensor(alpha).std().item()),
    }
    if isinstance(enc, WrappedSTE):
        out["neighbor_logits_mean"] = float(cell.neighbor_logits.mean().item())
        out["neighbor_logits_std"] = float(cell.neighbor_logits.std().item())
        out["neighbor_logits_min"] = float(cell.neighbor_logits.min().item())
        out["neighbor_logits_max"] = float(cell.neighbor_logits.max().item())
        out["ste_temperature"] = cell.ste_temperature
        # Hard mask: fraction of 1s in the binary mask.
        hard = cell.get_ste_hard_mask()
        out["mask_ones_fraction"] = float(hard.mean().item())
    else:
        out["neighbor_logits_std"] = float(cell.neighbor_logits.std().item())
        out["mask_density"] = cell.neighborhood_density()
    return out


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
    parser.add_argument(
        "--datasets", nargs="+", default=["toy_sin", "structured", "random"]
    )
    parser.add_argument(
        "--out", type=str, default="analysis/ste_neuron_wise_cfc_bench.json"
    )
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
                    f"[bench] mode={mode_name:18s} ds={ds_name:10s} "
                    f"seed={seed} test_mse={out['test_mse']:.6f} "
                    f"({cell_result['elapsed_sec']}s)"
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {out_path}")

    summary = {}
    for cell in results["cells"]:
        key = (cell["mode"], cell["dataset"])
        summary.setdefault(key, []).append(cell["test_mse"])
    print("\n[bench] Mean test_mse by mode × dataset:")
    print(f"{'mode':18s} | {'toy_sin':>10s} | {'structured':>10s} | {'random':>10s} | mean")
    for mode_name in args.modes:
        cols = []
        for ds in args.datasets:
            vals = summary.get((mode_name, ds), [])
            cols.append(sum(vals) / len(vals) if vals else float("nan"))
        mean = sum(c for c in cols if not math.isnan(c)) / max(len(cols), 1)
        print(
            f"{mode_name:18s} | {cols[0]:>10.6f} | "
            f"{cols[1]:>10.6f} | {cols[2]:>10.6f} | {mean:.6f}"
        )


if __name__ == "__main__":
    main()
