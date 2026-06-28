#!/usr/bin/env python3
"""Benchmark for SoftNeuronAttentionCfCCell (round 264).

Tests whether **soft attention** (row-softmax over per-neuron
neighbor logits, with learned temperature τ_attn) improves over
**hard top-k sparsification** (r263) and **plain CfC** (baseline).

The KEY question: is **learnable structure** better than
**hand-coded structure**?

Modes (5 total):
  * r263_baseline     — r263 NeuronWiseCfCCell density=0.3 (hard top-k)
  * softattn_default  — soft attention, τ_attn init=1.0, L1 λ=0.01
  * softattn_cold     — sharp attention, τ_attn init=0.1, L1 λ=0.001
  * softattn_warm     — soft attention, τ_attn init=5.0, L1 λ=0.1
  * softattn_nopen    — soft attention, no sparsity penalty (L1=0)

Hypotheses (PRD #10-101):

  H1: SoftNeuronAttentionCfCCell beats r263 (hard top-k) on at
      least one dataset because learnable structure outperforms
      hand-coded structure.
  H2: Attention weights become SPARSE naturally (mean attention
      weight < 0.1) after training — soft → sparse without
      explicit top-k.
  H3: Different neurons attend to different sources (per-row
      attention entropy varies, std > 0.5) — evidence of
      specialization.
  H4: SoftNeuronAttentionCfCCell is a strict superset of r263:
      with τ_attn → 0, the soft mask approaches a hard top-k.

Bench config:
  * 3 datasets: toy_sin, structured, random
  * hidden_size = 16
  * 100 epochs, lr=1e-2, batch=16, 2 seeds
  * Loss: MSE + λ × mean(|attention|) for L1 modes
  * Output: JSON with test_mse, attention diagnostics, tau stats.
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
from lnn.core.soft_neuron_attention_cfc import (  # noqa: E402
    SoftNeuronAttentionCfCCell,
)


# ---------------------------------------------------------------------------
# Toy data generators (match r263 bench protocol)
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
class WrappedCfC(nn.Module):
    """Wrap CfCCell so that output is (B, T, hidden_size)."""

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

    def __init__(self, input_size, hidden_size, density=0.3):
        super().__init__()
        self.cell = NeuronWiseCfCCell(
            input_size=input_size, hidden_size=hidden_size, density=density
        )
        self.hidden_size = hidden_size

    def forward(self, x):
        return self.cell(x)


class WrappedSoftAttention(nn.Module):
    """Wrap SoftNeuronAttentionCfCCell with optional L1 penalty in loss."""

    def __init__(
        self,
        input_size,
        hidden_size,
        init_tau_attn: float = 1.0,
        l1_lambda: float = 0.01,
        seed: int = 42,
    ):
        super().__init__()
        self.cell = SoftNeuronAttentionCfCCell(
            input_size=input_size,
            hidden_size=hidden_size,
            init_tau_attn=init_tau_attn,
            l1_lambda=l1_lambda,
            seed=seed,
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
    def __init__(self, encoder: nn.Module, hidden_size: int, sparsity_penalty: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.head = ReadoutHead(hidden_size)
        self.sparsity_penalty = float(sparsity_penalty)

    def forward(self, x):
        out, _ = self.encoder(x)
        return self.head(out)

    def extra_loss(self) -> torch.Tensor:
        """Auxiliary L1 penalty on attention (only for SoftAttn models)."""
        if not isinstance(self.encoder, WrappedSoftAttention):
            return torch.tensor(0.0)
        return self.encoder.cell.sparsity_loss()


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
    use_sparsity_loss: bool = False,
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
            mse = (pred - yb).pow(2).mean()
            loss = mse
            if use_sparsity_loss:
                loss = loss + model.extra_loss()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            epoch_loss += float(mse.item())  # report MSE-only
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
    "r263_baseline": dict(kind="neuronwise", density=0.3, use_sparsity=False),
    "softattn_default": dict(
        kind="softattn", init_tau_attn=1.0, l1_lambda=0.01, use_sparsity=True
    ),
    "softattn_cold": dict(
        kind="softattn", init_tau_attn=0.1, l1_lambda=0.001, use_sparsity=True
    ),
    "softattn_warm": dict(
        kind="softattn", init_tau_attn=5.0, l1_lambda=0.1, use_sparsity=True
    ),
    "softattn_nopen": dict(
        kind="softattn", init_tau_attn=1.0, l1_lambda=0.0, use_sparsity=False
    ),
}


def make_model(mode_cfg: dict, d_in: int, hidden_size: int) -> nn.Module:
    if mode_cfg["kind"] == "baseline":
        raise ValueError("baseline is not a soft-attention mode")
    elif mode_cfg["kind"] == "neuronwise":
        enc = WrappedNeuronWise(d_in, hidden_size, density=mode_cfg["density"])
    else:  # softattn
        enc = WrappedSoftAttention(
            d_in, hidden_size,
            init_tau_attn=mode_cfg["init_tau_attn"],
            l1_lambda=mode_cfg["l1_lambda"],
        )
    return SeqModel(enc, hidden_size)


def collect_diagnostics(model: SeqModel) -> dict:
    enc = model.encoder
    if isinstance(enc, WrappedNeuronWise):
        cell = enc.cell
        tau = cell.get_tau().detach().cpu().tolist()
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
    elif isinstance(enc, WrappedSoftAttention):
        cell = enc.cell
        tau = cell.get_tau().detach().cpu().tolist()
        alpha = cell.get_alpha().detach().cpu().tolist()
        entropy = cell.get_attention_entropy().detach().cpu().tolist()
        return {
            "tau_mean": float(sum(tau) / len(tau)),
            "tau_std": float(torch.tensor(tau).std().item()),
            "tau_min": float(min(tau)),
            "tau_max": float(max(tau)),
            "alpha_mean": float(sum(alpha) / len(alpha)),
            "alpha_std": float(torch.tensor(alpha).std().item()),
            "tau_attn": float(cell.get_tau_attn().item()),
            "attn_entropy_mean": float(sum(entropy) / len(entropy)),
            "attn_entropy_std": float(torch.tensor(entropy).std().item()),
            "attn_max_weight": cell.attention_max_weight(),
            "attn_sparsity_lt_001": cell.attention_sparsity(),
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
    parser.add_argument(
        "--datasets", nargs="+", default=["toy_sin", "structured", "random"]
    )
    parser.add_argument(
        "--out", type=str, default="analysis/soft_neuron_attention_cfc_bench.json"
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
                    use_sparsity_loss=bool(mode_cfg.get("use_sparsity", False)),
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
            f"{mode_name:24s} | {cols[0]:>10.6f} | "
            f"{cols[1]:>10.6f} | {cols[2]:>10.6f} | {mean:.6f}"
        )


if __name__ == "__main__":
    main()
