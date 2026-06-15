"""Round 182 — bench for LearnedBetaPS+LN+Depth-CfC (PRD #10-144).

Tests if depth scaling (num_layers ∈ {2, 3, 4, 5}) helps on
top of round 180 LN + Kh ladder winners.

36 cells: 6 conds × 3 datasets × 2 seeds × 30 epochs:
- lbps_ln_khl_2_5_2_n2: Kh=[2,5,2], num_layers=2 (sin winner shallow)
- lbps_ln_khl_2_5_2_n3: Kh=[2,5,2], num_layers=3 (round 180 control)
- lbps_ln_khl_2_5_2_n4: Kh=[2,5,2], num_layers=4 (deeper)
- lbps_ln_khl_2_5_2_n5: Kh=[2,5,2], num_layers=5 (deepest)
- lbps_ln_khl_5_3_2_n3: Kh=[5,3,2], num_layers=3 (round 180 control)
- lbps_ln_khl_5_3_2_n4: Kh=[5,3,2], num_layers=4 (deeper structured)

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

from lnn.core.learned_beta_ps_ln_khl_cfc import LearnedBetaPSLNKhlCfCStackedNetwork


def make_lbps_ln_khl(Kh_ladder, num_layers):
    """Generic factory with Kh_ladder and num_layers. Pads/truncates to match."""
    # Pad/truncate Kh_ladder to match num_layers
    if len(Kh_ladder) >= num_layers:
        ladder = Kh_ladder[:num_layers]
    else:
        ladder = Kh_ladder + [Kh_ladder[-1]] * (num_layers - len(Kh_ladder))
    return lambda input_size, hidden_size, output_size, num_layers=num_layers: LearnedBetaPSLNKhlCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=ladder, Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_sin_irr(B=32, T=32, D=2, missing_rate=0.3, seed=0):
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


CONDS = {
    "lbps_ln_khl_2_5_2_n2": make_lbps_ln_khl([2, 5, 2], 2),
    "lbps_ln_khl_2_5_2_n3": make_lbps_ln_khl([2, 5, 2], 3),
    "lbps_ln_khl_2_5_2_n4": make_lbps_ln_khl([2, 5, 2], 4),
    "lbps_ln_khl_2_5_2_n5": make_lbps_ln_khl([2, 5, 2], 5),
    "lbps_ln_khl_5_3_2_n3": make_lbps_ln_khl([5, 3, 2], 3),
    "lbps_ln_khl_5_3_2_n4": make_lbps_ln_khl([5, 3, 2], 4),
}


def train_and_eval(model, x_train, y_train, x_eval, y_eval, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        out = model(x_train)
        loss = F.mse_loss(out, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        out = model(x_eval)
        eval_loss = F.mse_loss(out, y_eval).item()
    return eval_loss


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--out", type=str, default="results/bench_learned_beta_ps_ln_depth_cfc.json")
    args = parser.parse_args()

    D = 2
    H = args.hidden
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"=== Round 182 LearnedBetaPS+LN+Depth-CfC Bench ===")
    print(f"epochs={args.epochs}, seeds={args.seeds}, B={args.batch}, T={args.T}, H={H}, lr={args.lr}")
    print(f"6 conds × 3 datasets × 2 seeds × 30 epochs = 36 cells")

    results = {}
    for cond_name, cond_factory in CONDS.items():
        results[cond_name] = {}
        for ds_name, ds_fn in DATASETS.items():
            results[cond_name][ds_name] = []
            for seed in range(args.seeds):
                x, y = ds_fn(B=args.batch, T=args.T, D=D, seed=seed)
                torch.manual_seed(seed)
                # cond_factory takes num_layers=3 default; we override with cond_factory's num_layers
                model = cond_factory(input_size=D, hidden_size=H, output_size=1)
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

    print("\n=== Summary (mean ± std) ===")
    header = f"{'Cond':<28}"
    for ds_name in DATASETS:
        header += f" | {ds_name:<16}"
    header += " | n_params"
    print(header)
    print("-" * len(header))
    for cond_name in CONDS:
        line = f"{cond_name:<28}"
        n_params = None
        for ds_name in DATASETS:
            losses = [r["loss"] for r in results[cond_name][ds_name]]
            mean = sum(losses) / len(losses)
            std = (sum((l - mean) ** 2 for l in losses) / len(losses)) ** 0.5
            line += f" | {mean:.4f}±{std:.4f}"
        n_params = results[cond_name][list(DATASETS)[0]][0]["n_params"]
        line += f" | {n_params}"
        print(line)

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
