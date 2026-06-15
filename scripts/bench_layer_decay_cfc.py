"""Round 167 — bench for LayerDecay-CfC (per-layer β schedule) (PRD #10-129).

48 cells: 8 conds × 3 datasets × 2 seeds × 30 epochs:
- ld_constant:       round 165 baseline (control)
- ld_linear_k5:      3-layer, Kx=5, β ∈ [0.5, 0.99] linear
- ld_reverse_k5:     3-layer, Kx=5, β ∈ [0.99, 0.5] reverse
- ld_linear_slow:    3-layer, Kx=5, β ∈ [0.7, 0.99] linear
- ld_linear_fast:    3-layer, Kx=5, β ∈ [0.3, 0.9] linear
- ld_linear_relu:    3-layer, Kx=5, β ∈ [0.7, 0.95] linear
- ld_4layer_linear:  4-layer, Kx=5, β ∈ [0.5, 0.99] linear (depth unlock test)
- ld_4layer_reverse: 4-layer, Kx=5, β ∈ [0.99, 0.5] reverse

Datasets: sin_irr, structured_irr, random_irr (D=2, T=32, missing_rate=0.3).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.layer_decay_cfc import (
    LayerDecayCfCStackedNetwork,
    make_ld_constant,
    make_ld_linear_k5,
    make_ld_reverse_k5,
    make_ld_linear_slow,
    make_ld_linear_fast,
    make_ld_linear_relu,
)


# ---------------------------------------------------------------------------
# Data generators (mirror round 165)
# ---------------------------------------------------------------------------


def make_sin_irr(B=32, T=32, D=2, missing_rate=0.3, seed=0):
    """Sinusoidal data with random missing values (NaN)."""
    torch.manual_seed(seed)
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
    x = torch.zeros(B, T, D)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    mask = torch.rand(B, T, D) > missing_rate
    x = torch.where(mask, x, torch.full_like(x, float("nan")))
    y = torch.sin(t.squeeze(-1)).unsqueeze(-1).expand(B, T, 1)
    return x, y


def make_structured_irr(B=32, T=32, D=2, missing_rate=0.3, seed=0):
    """Structured: sin for first half, sin(2t) for second half."""
    torch.manual_seed(seed)
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
    x = torch.zeros(B, T, D)
    half = T // 2
    t1 = t[:, :half, :].squeeze(-1)
    t2 = t[:, half:, :].squeeze(-1)
    x[:, :half, 0] = torch.sin(t1)
    x[:, :half, 1] = torch.cos(t1)
    x[:, half:, 0] = torch.sin(2 * t2)
    x[:, half:, 1] = torch.cos(2 * t2)
    mask = torch.rand(B, T, D) > missing_rate
    x = torch.where(mask, x, torch.full_like(x, float("nan")))
    y = torch.zeros(B, T, 1)
    y[:, :half, 0] = torch.sin(2 * t1)
    y[:, half:, 0] = torch.sin(t2)
    return x, y


def make_random_irr(B=32, T=32, D=2, missing_rate=0.3, seed=0):
    """Cumulative random walk with NaN masking. Target = true cumsum (no NaN)."""
    torch.manual_seed(seed)
    raw = torch.randn(B, T, D)
    full = torch.cumsum(raw, dim=1) * 0.1
    mask = torch.rand(B, T, D) > missing_rate
    x = torch.where(mask, full, torch.full_like(full, float("nan")))
    y = full.clone()
    return x, y


DATASETS = {
    "sin_irr": make_sin_irr,
    "structured_irr": make_structured_irr,
    "random_irr": make_random_irr,
}


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def make_ld_4layer_linear(input_size, hidden_size, output_size, num_layers=4):
    """4-layer linear β ∈ [0.5, 0.99]."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.5, 0.99], mode="linear",
        return_sequences=True,
    )


def make_ld_4layer_reverse(input_size, hidden_size, output_size, num_layers=4):
    """4-layer reverse β ∈ [0.99, 0.5]."""
    return LayerDecayCfCStackedNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        Kx=5, betas_h=[0.99, 0.5], mode="reverse",
        return_sequences=True,
    )


CONDS = {
    "ld_constant": make_ld_constant,
    "ld_linear_k5": make_ld_linear_k5,
    "ld_reverse_k5": make_ld_reverse_k5,
    "ld_linear_slow": make_ld_linear_slow,
    "ld_linear_fast": make_ld_linear_fast,
    "ld_linear_relu": make_ld_linear_relu,
    "ld_4layer_linear": make_ld_4layer_linear,
    "ld_4layer_reverse": make_ld_4layer_reverse,
}


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


def train_and_eval(model, x_train, y_train, x_eval, y_eval, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        out = model(x_train)
        # Handle output dim: if model output has D=2 instead of 1, select first channel.
        if out.shape[-1] != y_train.shape[-1]:
            out = out[..., :y_train.shape[-1]]
        loss = F.mse_loss(out, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        out = model(x_eval)
        if out.shape[-1] != y_eval.shape[-1]:
            out = out[..., :y_eval.shape[-1]]
        eval_loss = F.mse_loss(out, y_eval).item()
    return eval_loss


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Main bench loop
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--out", type=str, default="results/bench_layer_decay_cfc.json")
    args = parser.parse_args()

    D = 2
    H = args.hidden
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"=== Round 167 LayerDecay-CfC Bench ===")
    print(f"epochs={args.epochs}, seeds={args.seeds}, B={args.batch}, T={args.T}, H={H}, lr={args.lr}")
    print(f"8 conds × 3 datasets × 2 seeds × 30 epochs = 48 cells")
    print(f"Per-layer β schedule for h-side, per-feature β on x-side (Kx=5)")

    results = {}
    for cond_name, cond_fn in CONDS.items():
        # Decide num_layers.
        num_layers = 4 if "4layer" in cond_name else 3
        results[cond_name] = {}
        for ds_name, ds_fn in DATASETS.items():
            results[cond_name][ds_name] = []
            for seed in range(args.seeds):
                x, y = ds_fn(B=args.batch, T=args.T, D=D, seed=seed)
                torch.manual_seed(seed)
                model = cond_fn(input_size=D, hidden_size=H, output_size=1, num_layers=num_layers)
                t0 = time.time()
                eval_loss = train_and_eval(
                    model, x, y, x, y, epochs=args.epochs, lr=args.lr,
                )
                t1 = time.time()
                results[cond_name][ds_name].append({
                    "seed": seed,
                    "loss": eval_loss,
                    "time_sec": round(t1 - t0, 2),
                    "n_params": count_params(model),
                })
                print(f"  {cond_name} | {ds_name} | seed={seed} | loss={eval_loss:.4f} | t={t1-t0:.1f}s")

    # Summary table.
    print("\n=== Summary (mean ± std) ===")
    header = f"{'Cond':<24}"
    for ds_name in DATASETS:
        header += f" | {ds_name:<16}"
    header += " | n_params"
    print(header)
    print("-" * len(header))
    for cond_name in CONDS:
        line = f"{cond_name:<24}"
        n_params = None
        for ds_name in DATASETS:
            losses = [r["loss"] for r in results[cond_name][ds_name]]
            mean = sum(losses) / len(losses)
            std = (sum((l - mean) ** 2 for l in losses) / len(losses)) ** 0.5
            line += f" | {mean:.4f}±{std:.4f}"
        n_params = results[cond_name][list(DATASETS)[0]][0]["n_params"]
        line += f" | {n_params}"
        print(line)

    # Save raw results.
    summary = {}
    for cond_name in CONDS:
        summary[cond_name] = {}
        summary[cond_name]["n_params"] = results[cond_name][list(DATASETS)[0]][0]["n_params"]
        for ds_name in DATASETS:
            losses = [r["loss"] for r in results[cond_name][ds_name]]
            mean = sum(losses) / len(losses)
            std = (sum((l - mean) ** 2 for l in losses) / len(losses)) ** 0.5
            summary[cond_name][ds_name] = {
                "mean": round(mean, 6),
                "std": round(std, 6),
                "raw": [round(l, 6) for l in losses],
            }
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": summary}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
