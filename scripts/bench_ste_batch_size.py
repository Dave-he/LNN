#!/usr/bin/env python3
"""Benchmark for STEWithEntropy batch size sweep (round 276).

Tests whether r267/r275 production settings (batch=16, h=192,
T=64, d_in=1, density=0.3, λ=0.1) hold for different batch sizes.

The hypothesis (PRD #10-113): batch size affects gradient noise.
Smaller batch → noisier gradients → stronger regularization but
possibly worse convergence. Larger batch → fewer updates per
epoch → potentially worse learning for same compute.

Modes (5 total):
  * ste_entropy_b4_h192    — NEW (small batch, 64 updates/epoch)
  * ste_entropy_b8_h192    — NEW (medium-small, 32 updates/epoch)
  * ste_entropy_b16_h192   — r267-r275 PRODUCTION
  * ste_entropy_b32_h192   — NEW (large batch, 8 updates/epoch)
  * ste_entropy_b64_h192   — NEW (full batch, 4 updates/epoch)

Hypotheses (PRD #10-113):

  H1: batch=16 is optimal on structured
  H2: smaller batch (4, 8) doesn't hurt structured
  H3: larger batch (32, 64) ≈ batch=16 on structured
  H4: top1_frac preserved across batch sizes
  H5: smaller batch reduces seed variance

Bench config:
  * 5 modes × 3 datasets × 3 seeds = 45 cells
  * 100 epochs, lr=1e-2
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

from lnn.core.ste_entropy_neuron_wise_cfc import STEWithEntropy  # noqa: E402


# ---------------------------------------------------------------------------
# Toy data generators (single-channel)
# ---------------------------------------------------------------------------
def make_toy_sin(T: int = 64, n_samples: int = 256, seed: int = 0):
    t = torch.linspace(0, 1, T + 1).unsqueeze(0).repeat(n_samples, 1)
    y = torch.sin(2 * math.pi * t)
    x = y[:, :-1].unsqueeze(-1)
    y_target = y[:, 1:].unsqueeze(-1)
    return x, y_target


def make_structured(T: int = 64, n_samples: int = 256, seed: int = 0):
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
    return x, y_target


def make_random(T: int = 64, n_samples: int = 256, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(n_samples, T + 1, generator=g)
    x = y[:, :-1].unsqueeze(-1)
    y_target = y[:, 1:].unsqueeze(-1)
    return x, y_target


DATA_FACTORIES = {
    "toy_sin": make_toy_sin,
    "structured": make_structured,
    "random": make_random,
}


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------
class WrappedSTEWithEntropy(nn.Module):
    def __init__(self, input_size, hidden_size, density, entropy_lambda, ste_temperature):
        super().__init__()
        self.cell = STEWithEntropy(
            input_size=input_size,
            hidden_size=hidden_size,
            density=density,
            ste_temperature=ste_temperature,
            entropy_lambda=entropy_lambda,
        )
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
    def __init__(self, encoder: nn.Module, hidden_size: int, entropy_lambda: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.head = ReadoutHead(hidden_size)
        self.entropy_lambda = float(entropy_lambda)

    def forward(self, x):
        out, _ = self.encoder(x)
        return self.head(out)

    def extra_loss(self) -> torch.Tensor:
        if self.entropy_lambda <= 0:
            return torch.tensor(0.0)
        return self.encoder.cell.extra_loss()


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
    n_updates_per_epoch = max(1, N // batch_size)
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        x_tr = x_train[perm]
        y_tr = y_train[perm]
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            xb = x_tr[i : i + batch_size].to(device)
            yb = y_tr[i : i + batch_size].to(device)
            if xb.shape[0] == 0:
                continue
            pred = model(xb)
            mse = (pred - yb).pow(2).mean()
            loss = mse + model.extra_loss()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            epoch_loss += float(mse.item())
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
    model.eval()
    with torch.no_grad():
        pred = model(x_eval.to(device))
        test_mse = float((pred - y_eval.to(device)).pow(2).mean().item())
    return {"test_mse": test_mse, "train_loss_last": losses[-1]}


# ---------------------------------------------------------------------------
# Mode definitions (batch size sweep at h=192)
# ---------------------------------------------------------------------------
MODES = {
    "ste_entropy_b4_h192":  dict(batch_size=4,  hidden_size=192, density=0.3, ste_temperature=1.0, entropy_lambda=0.1),
    "ste_entropy_b8_h192":  dict(batch_size=8,  hidden_size=192, density=0.3, ste_temperature=1.0, entropy_lambda=0.1),
    "ste_entropy_b16_h192": dict(batch_size=16, hidden_size=192, density=0.3, ste_temperature=1.0, entropy_lambda=0.1),
    "ste_entropy_b32_h192": dict(batch_size=32, hidden_size=192, density=0.3, ste_temperature=1.0, entropy_lambda=0.1),
    "ste_entropy_b64_h192": dict(batch_size=64, hidden_size=192, density=0.3, ste_temperature=1.0, entropy_lambda=0.1),
}


def make_model(mode_cfg: dict) -> nn.Module:
    enc = WrappedSTEWithEntropy(
        input_size=1,
        hidden_size=mode_cfg["hidden_size"],
        density=mode_cfg["density"],
        entropy_lambda=mode_cfg["entropy_lambda"],
        ste_temperature=mode_cfg["ste_temperature"],
    )
    return SeqModel(enc, mode_cfg["hidden_size"], entropy_lambda=mode_cfg["entropy_lambda"])


def collect_diagnostics(model: SeqModel) -> dict:
    cell = model.encoder.cell
    nl = cell.neighbor_logits.detach()
    soft = cell.get_ste_soft_mask()
    row_max = soft.max(dim=-1).values
    row_sum = soft.sum(dim=-1) + 1e-8
    top1_frac = float((row_max / row_sum).mean().item())
    return {
        "hidden_size": cell.hidden_size,
        "density": cell.density,
        "entropy_lambda": cell.entropy_lambda,
        "soft_mask_entropy": cell.entropy_value(),
        "max_entropy": cell.max_entropy(),
        "entropy_fraction": float(cell.entropy_value() / cell.max_entropy()),
        "neighbor_logits_mean": float(nl.mean().item()),
        "neighbor_logits_std": float(nl.std().item()),
        "neighbor_logits_min": float(nl.min().item()),
        "neighbor_logits_max": float(nl.max().item()),
        "neighbor_logits_abs_mean": float(nl.abs().mean().item()),
        "fraction_near_zero": float((nl.abs() < 0.01).float().mean().item()),
        "top1_frac": top1_frac,
        "soft_mask_mean": float(soft.mean().item()),
        "soft_mask_std": float(soft.std().item()),
        "n_params": sum(p.numel() for p in model.parameters()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--datasets", nargs="+", default=["toy_sin", "structured", "random"]
    )
    parser.add_argument(
        "--out", type=str, default="analysis/ste_batch_size_bench.json"
    )
    parser.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bench] device={device}")

    results = {
        "config": {
            "epochs": args.epochs, "lr": args.lr,
            "T": args.T, "n_samples": args.n_samples,
            "seeds": args.seeds, "datasets": args.datasets,
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
                    T=args.T, n_samples=args.n_samples, seed=seed
                )
                N = x.shape[0]
                split = int(0.8 * N)
                x_tr, x_ev = x[:split], x[split:]
                y_tr, y_ev = y[:split], y[split:]

                model = make_model(mode_cfg)
                out = train_one(
                    model, x_tr, y_tr, x_ev, y_ev,
                    epochs=args.epochs, lr=args.lr,
                    batch_size=mode_cfg["batch_size"],
                    device=device,
                )
                diag = collect_diagnostics(model)
                cell_result = {
                    "mode": mode_name,
                    "batch_size": mode_cfg["batch_size"],
                    "hidden_size": mode_cfg["hidden_size"],
                    "dataset": ds_name,
                    "seed": seed,
                    "test_mse": out["test_mse"],
                    "train_loss_last": out["train_loss_last"],
                    "elapsed_sec": round(time.time() - t0, 2),
                    "diagnostics": diag,
                }
                results["cells"].append(cell_result)
                print(
                    f"[bench] mode={mode_name:24s} batch={mode_cfg['batch_size']:>3d} "
                    f"ds={ds_name:10s} seed={seed} test_mse={out['test_mse']:.6f} "
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
    header = f"{'mode':24s} | " + " | ".join(f"{ds:>10s}" for ds in args.datasets) + " | mean"
    print(header)
    for mode_name in args.modes:
        cols = []
        for ds in args.datasets:
            vals = summary.get((mode_name, ds), [])
            cols.append(sum(vals) / len(vals) if vals else float("nan"))
        valid = [c for c in cols if not math.isnan(c)]
        mean = sum(valid) / max(len(valid), 1)
        cells_str = " | ".join(f"{c:>10.6f}" for c in cols)
        print(f"{mode_name:24s} | {cells_str} | {mean:.6f}")

    print("\n[bench] Seed variance (std across seeds) by mode × dataset:")
    print(f"{'mode':24s} | " + " | ".join(f"{ds:>10s}" for ds in args.datasets))
    for mode_name in args.modes:
        cols = []
        for ds in args.datasets:
            vals = summary.get((mode_name, ds), [])
            if len(vals) >= 2:
                mean_v = sum(vals) / len(vals)
                var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
                cols.append(math.sqrt(var_v))
            else:
                cols.append(0.0)
        cells_str = " | ".join(f"{c:>10.6f}" for c in cols)
        print(f"{mode_name:24s} | {cells_str}")


if __name__ == "__main__":
    main()