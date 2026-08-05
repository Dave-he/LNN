"""N24: MR routing on long-sequence / multi-scale tasks.

N14 found that MR-hybrid-gate-CfC doesn't beat single mfc-hybrid_gate on
AR(2) 3-regime (sl=24, 3 regimes). N14 hypothesised that AR(2) 3-regime is
too simple for MR routing to help (H1: task too narrow).

N24 tests on a richer task:
  - Long sequence (sl=96): 4× longer than AR(2) 3-regime
  - Multi-scale temporal structure: each regime has a distinct *frequency*
    signature, not just AR coefficients (sinusoidal + AR mixed)
  - 8 regimes (vs 3 in N14)

If MR routing has real multi-scale specialisation, it should now help
because the input itself has multiple time scales to specialise on.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lnn.core.multirate_tfp_cfc import MultiRateTfpCfCNetwork


def make_multiscale_task(n_samples, seq_len, n_feat=4, n_regimes=8, seed=0,
                         dt_log_sigma=0.0):
    """Multi-scale non-stationary task with per-regime frequency content.

    Each regime has a distinct *dominant frequency* (sinusoidal carrier) and
    a different AR(2) envelope, so the signal has both fast and slow
    components that benefit from multi-rate decomposition.
    """
    torch.manual_seed(seed)
    x = torch.zeros(n_samples, seq_len, n_feat)
    y = torch.zeros(n_samples, seq_len, 1)
    dt = torch.ones(n_samples, seq_len)
    if dt_log_sigma > 0:
        dt = torch.exp(torch.randn(n_samples, seq_len) * dt_log_sigma - 0.5 * dt_log_sigma**2)
        dt[:, 0] = 1.0
    # Per-regime parameters: (freq_hz, ar1, ar2)
    regimes = [
        (0.05, 0.6, 0.2),     # slow
        (0.10, -0.3, 0.5),    # medium-slow
        (0.20, 0.4, -0.4),    # medium
        (0.30, 0.2, 0.6),     # medium-fast
        (0.40, -0.4, 0.4),    # fast
        (0.50, 0.5, -0.3),    # fast
        (0.60, 0.7, -0.1),    # very-fast
        (0.70, -0.2, 0.5),    # very-very-fast
    ][:n_regimes]
    for s in range(n_samples):
        regime = torch.randint(0, n_regimes, (1,)).item()
        freq, ar1, ar2 = regimes[regime]
        noise = torch.randn(seq_len, n_feat) * 0.1
        t_axis = torch.arange(seq_len).float() * dt[s].mean().item() / seq_len
        # Per-feature carrier (different phases)
        phases = torch.rand(n_feat) * 2 * torch.pi
        carrier = torch.sin(2 * torch.pi * freq * t_axis.unsqueeze(-1) + phases)
        x[s] = carrier.clone()
        for t in range(1, seq_len):
            x[s, t] = ar1 * x[s, t - 1] + ar2 * x[s, max(t - 2, 0)] + noise[t] + 0.3 * carrier[t]
        # Target = sum of next-step features (multi-step ahead needs to track both freq + AR)
        y[s, :-1, 0] = x[s, 1:, :].sum(-1)
    return x, y, dt


def train_eval(model, x_tr, y_tr, dt_tr, x_te, y_te, dt_te,
               epochs=3, batch=8, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x_tr.shape[0]
    t0 = time.perf_counter()
    for _ in range(epochs):
        for b in range(0, n, batch):
            xb = x_tr[b:b + batch]
            yb = y_tr[b:b + batch]
            dtb = dt_tr[b:b + batch]
            opt.zero_grad()
            pred = model(xb, dt=dtb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            opt.step()
    train_s = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        test_mse = nn.functional.mse_loss(model(x_te, dt=dt_te), y_te).item()
    return {"test_mse": test_mse, "train_seconds": train_s}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-tau", type=int, default=4)
    parser.add_argument("--n-regimes", type=int, default=8)
    parser.add_argument("--date", default="2026-08-05")
    args = parser.parse_args()

    n_feat = 4
    out = 1
    hidden = args.hidden
    n_tau = args.n_tau

    sigma_train = 0.5
    torch.manual_seed(0)
    x_irreg, y_irreg, dt_irreg = make_multiscale_task(
        args.n_samples, args.seq_len, n_feat=n_feat, n_regimes=args.n_regimes,
        seed=0, dt_log_sigma=sigma_train,
    )
    n_tr = int(0.8 * x_irreg.shape[0])
    x_tr, y_tr, dt_tr = x_irreg[:n_tr], y_irreg[:n_tr], dt_irreg[:n_tr]
    x_te, y_te = x_irreg[n_tr:], y_irreg[n_tr:]
    torch.manual_seed(1)
    dt_te = torch.exp(torch.randn(*x_te.shape[:2]) * sigma_train - 0.5 * sigma_train**2)
    dt_te[:, 0] = 1.0

    print(f"Task: multi-scale (8 regimes, sinusoidal+AR), sl={args.seq_len}, h={args.hidden}, n_tau={args.n_tau}")
    print(f"Single: n_tau=2 k=1 (degenerate), MR: n_tau={args.n_tau}\n")

    results = {}
    for name in ("cfc", "mfc-hybrid_gate", "mr-hybrid-gate-cfc"):
        msess, trains = [], []
        for r in range(args.repeats):
            torch.manual_seed(42 + r)
            if name == "cfc":
                model = MultiRateTfpCfCNetwork(
                    n_feat, hidden, out, n_tau=2, top_k_active=1,
                    expert_retention_kind="cfc",
                )
            elif name == "mfc-hybrid_gate":
                model = MultiRateTfpCfCNetwork(
                    n_feat, hidden, out, n_tau=2, top_k_active=1,
                    expert_retention_kind="hybrid_gate",
                )
            else:
                model = MultiRateTfpCfCNetwork(
                    n_feat, hidden, out, n_tau=n_tau, top_k_active=n_tau // 2,
                    expert_retention_kind="hybrid_gate",
                )
            res = train_eval(model, x_tr, y_tr, dt_tr, x_te, y_te, dt_te, epochs=args.epochs)
            msess.append(res["test_mse"])
            trains.append(res["train_seconds"])
        n_params = sum(p.numel() for p in model.parameters())
        per_expert = hidden // n_tau if name == "mr-hybrid-gate-cfc" else hidden
        results[name] = {
            "params": n_params,
            "per_expert_hidden": per_expert,
            "mse_mean": statistics.mean(msess),
            "mse_std": statistics.stdev(msess) if len(msess) > 1 else 0.0,
            "train_s_mean": statistics.mean(trains),
        }
        r = results[name]
        print(f"  {name:25s} (per_exp={per_expert:2d}, params={n_params:5d}): "
              f"MSE={r['mse_mean']:.4f}±{r['mse_std']:.4f}  train={r['train_s_mean']:.1f}s")

    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.date}_mr_long_sequence.json"
    md_path = out_dir / f"{args.date}_mr_long_sequence.md"
    json_path.write_text(json.dumps({
        "date": args.date,
        "task": "mr_routing_long_sequence_multiscale",
        "config": vars(args),
        "results": results,
    }, indent=2))

    md = [f"""---
title: MR routing on long-sequence / multi-scale tasks (N24) — {args.date}
date: {args.date}
tags: [LNN, MR-MoE, hybrid_gate, multi-rate, long-sequence, multi-scale, N24]
---

# MR routing on long-sequence / multi-scale tasks (N24) — {args.date}

## Setup
- Task: multi-scale non-stationary (8 regimes, sinusoidal + AR mixed)
- sl={args.seq_len}, h={args.hidden}, n_tau={args.n_tau}
- 8 regimes with distinct *frequency content* (0.05–0.70 Hz carriers)

## Results

| model | per_expert | params | test MSE | train s |
|---|---:|---:|---:|---:|
"""]
    for name in ("cfc", "mfc-hybrid_gate", "mr-hybrid-gate-cfc"):
        r = results[name]
        md.append(f"| {name} | {r['per_expert_hidden']} | {r['params']} | "
                  f"{r['mse_mean']:.4f} ± {r['mse_std']:.4f} | "
                  f"{r['train_s_mean']:.1f} |\n")
    md_path.write_text("".join(md))
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
