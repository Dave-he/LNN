"""N2 PDE domain validation: L-RFM (frozen LTC features) on 1D heat equation.

The L-RFM paper (arXiv 2606.15571) targets stiff/dispersive PDE solving.
This benchmark validates that the in-code implementation is competitive
on the paper's actual domain (1D heat equation u_t = alpha * u_xx).

Setup:
    1D heat equation: u_t = alpha * u_xx,  x in [0, 1], periodic BC
    Initial: u(x, 0) = sin(2*pi*x) (one mode)
    Analytical solution: u(x, t) = sin(2*pi*x) * exp(-alpha * (2*pi)^2 * t)

Models:
    L-RFM n_features=32/64/128  (frozen LTC features, only readout trained)
    CfC h=8/16/24 (trained, for comparison on a domain they were not designed for)
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

from lnn.core.cfc import CfCCell
from lnn.core.lrfm import LRFMSequenceRegressor


def make_heat_equation_data(n_samples, n_space, alpha=0.01, dt=0.001, n_steps=50, seed=0):
    """Generate heat equation trajectories: u_t = alpha * u_xx.

    Initial condition: u(x, 0) = sin(2*pi*x)
    Analytical solution: u(x, t) = sin(2*pi*x) * exp(-alpha*(2*pi)^2 * t)
    """
    torch.manual_seed(seed)
    # Spatial grid in [0, 1], periodic
    x = torch.linspace(0, 1, n_space + 1)[:-1]  # periodic so drop last point
    # Initial condition
    u0 = torch.sin(2 * torch.pi * x)  # (n_space,)
    # Analytical solution at each time step
    t_grid = torch.arange(n_steps + 1) * dt
    # u(x, t) = sin(2*pi*x) * exp(-alpha*(2*pi)^2 * t)
    # Shape: (n_steps+1, n_space)
    u_analytical = torch.sin(2 * torch.pi * x).unsqueeze(0) * torch.exp(
        -alpha * (2 * torch.pi) ** 2 * t_grid
    ).unsqueeze(-1)
    # Replicate for n_samples
    u_data = u_analytical.unsqueeze(0).expand(n_samples, -1, -1).clone()
    # Add small noise
    u_data = u_data + torch.randn_like(u_data) * 1e-4
    return u_data, x  # (n_samples, n_steps+1, n_space)


class _CfCSpatialWrap(nn.Module):
    """Wrap CfC for PDE solving: per-step (u_t-1) -> predict u_t."""

    def __init__(self, cell, n_space):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(cell.hidden_size, n_space)

    def forward(self, x, dt=1.0):
        # x: (B, T, n_space)
        b, t, _ = x.shape
        h = torch.zeros(b, self.cell.hidden_size)
        outs = []
        for i in range(t):
            h = self.cell(x[:, i, :], h, dt=dt)
            outs.append(self.head(h))
        return torch.stack(outs, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--n-space", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--date", default="2026-08-05")
    args = parser.parse_args()

    n_space = args.n_space
    out = 1  # predict full field, but we'll use a single output head per space point

    torch.manual_seed(0)
    u_data, x_grid = make_heat_equation_data(args.n_samples, n_space, n_steps=args.n_steps, seed=0)
    n_tr = int(0.8 * u_data.shape[0])
    u_tr, u_te = u_data[:n_tr], u_data[n_tr:]
    print(f"Heat equation: u_t = alpha * u_xx, x in [0,1], {n_space} points, t in [0, {args.n_steps}]")
    print(f"Train: {n_tr} samples, Test: {u_te.shape[0]} samples\n")

    # Models
    rows = []
    # L-RFM variants (paper's actual domain)
    for n_feat in [32, 64, 128]:
        msess, trains = [], []
        for r in range(args.repeats):
            torch.manual_seed(42 + r)
            model = LRFMSequenceRegressor(
                input_size=n_space, output_size=n_space, n_features=n_feat,
            )
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
            t0 = time.perf_counter()
            for _ in range(args.epochs):
                for b in range(0, n_tr, 8):
                    xb = u_tr[b:b+8]
                    yb = u_te[b:b+8] if False else u_tr[b+1:b+9]  # next step
                    if yb.shape[0] != xb.shape[0]:
                        continue
                    opt.zero_grad()
                    pred = model(xb, dt=0.001)  # dt matches data generation
                    loss = nn.functional.mse_loss(pred, yb)
                    loss.backward()
                    opt.step()
            train_s = time.perf_counter() - t0
            with torch.no_grad():
                mse = nn.functional.mse_loss(model(u_te[:, :-1], dt=0.001), u_te[:, 1:]).item()
            msess.append(mse)
            trains.append(train_s)
        rows.append({
            "model": f"L-RFM n_features={n_feat}",
            "params": sum(p.numel() for p in LRFMSequenceRegressor(input_size=n_space, output_size=n_space, n_features=n_feat).parameters()),
            "test_mse_mean": statistics.mean(msess),
            "test_mse_std": statistics.stdev(msess) if len(msess) > 1 else 0.0,
            "train_s_mean": statistics.mean(trains),
        })

    # Trained CfC (out of domain — for comparison)
    for h in [8, 16, 24]:
        msess, trains = [], []
        for r in range(args.repeats):
            torch.manual_seed(42 + r)
            cell = CfCCell(n_space, h)
            model = _CfCSpatialWrap(cell, n_space)
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            t0 = time.perf_counter()
            for _ in range(args.epochs):
                for b in range(0, n_tr, 8):
                    xb = u_tr[b:b+8]
                    yb = u_tr[b+1:b+9] if b+9 <= n_tr else u_tr[b:b+8]
                    if yb.shape[0] != xb.shape[0]:
                        continue
                    opt.zero_grad()
                    pred = model(xb, dt=0.001)
                    loss = nn.functional.mse_loss(pred, yb)
                    loss.backward()
                    opt.step()
            train_s = time.perf_counter() - t0
            with torch.no_grad():
                mse = nn.functional.mse_loss(model(u_te[:, :-1], dt=0.001), u_te[:, 1:]).item()
            msess.append(mse)
            trains.append(train_s)
        rows.append({
            "model": f"CfC h={h} (out-of-domain)",
            "params": sum(p.numel() for p in _CfCSpatialWrap(CfCCell(n_space, h), n_space).parameters()),
            "test_mse_mean": statistics.mean(msess),
            "test_mse_std": statistics.stdev(msess) if len(msess) > 1 else 0.0,
            "train_s_mean": statistics.mean(trains),
        })

    for r in rows:
        print(f"  {r['model']:30s}: params={r['params']:5d}, "
              f"MSE={r['test_mse_mean']:.6f}±{r['test_mse_std']:.6f}, train_s={r['train_s_mean']:.2f}")

    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.date}_lrfm_heat.json"
    md_path = out_dir / f"{args.date}_lrfm_heat.md"
    json_path.write_text(json.dumps({
        "date": args.date,
        "task": "lrfm_heat_equation_validation",
        "config": vars(args),
        "rows": rows,
    }, indent=2))
    md = [f"""---
title: L-RFM on heat equation (N2 PDE domain validation) — {args.date}
date: {args.date}
tags: [LNN, L-RFM, heat-equation, PDE, N2, domain-validation]
arxiv_refs: [2606.15571]
---

# L-RFM on heat equation (N2 PDE domain validation) — {args.date}

## Setup
- Task: 1D heat equation `u_t = alpha * u_xx` on periodic domain
- Initial: `u(x, 0) = sin(2*pi*x)`, solution: `u(x, t) = sin(2*pi*x) * exp(-alpha * (2*pi)^2 * t)`
- This is L-RFM paper's actual domain (PDE solving)
- L-RFM (frozen LTC features + linear readout) vs trained CfC (out of domain)

## Results

| model | params | test MSE |
|---|---:|---:|
"""]
    for r in rows:
        md.append(f"| {r['model']} | {r['params']} | "
                  f"{r['test_mse_mean']:.6f} ± {r['test_mse_std']:.6f} |\n")
    md_path.write_text("".join(md))
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
