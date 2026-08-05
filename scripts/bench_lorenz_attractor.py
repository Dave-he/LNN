"""N18: Validate retention findings on Lorenz attractor (nonlinear ODE system).

AR(2) 3-regime is too simple. Lorenz attractor is a real nonlinear ODE
system used in many LNN papers as a real-world test case:

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

With sigma=10, rho=28, beta=8/3, the system is chaotic.

If the 21-round retention design space findings (CfC = structural-generic,
TFP regresses on OOD dt, hybrid_gate better in-dist, MR routing better
on multi-scale) transfer to this realistic nonlinear ODE, N18 is positive.
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
from lnn.core.memory_fusion_cfc import MemoryFusionCfCCell
from lnn.core.multirate_tfp_cfc import MultiRateTfpCfCNetwork


def make_lorenz_dataset(n_samples, seq_len, sigma=10.0, rho=28.0, beta=8.0/3.0,
                        dt=0.01, dt_jitter=0.0, seed=0):
    """Generate Lorenz attractor sequences with optional irregular dt.

    The system is integrated using a simple Euler scheme. Each sample
    starts from a random initial condition near the attractor.
    """
    torch.manual_seed(seed)
    X = torch.zeros(n_samples, seq_len, 3)
    Y = torch.zeros(n_samples, seq_len, 1)  # target = next-step x
    Dt = torch.ones(n_samples, seq_len)
    if dt_jitter > 0:
        # Per-sample irregular dt
        Dt = torch.exp(torch.randn(n_samples, seq_len) * dt_jitter - 0.5 * dt_jitter**2)
        Dt[:, 0] = 1.0
    for s in range(n_samples):
        # Random initial condition near attractor
        x = torch.randn(3) * 5.0
        for t in range(seq_len):
            dti = Dt[s, t].item() * dt
            dx = sigma * (x[1] - x[0]) * dti
            dy = (x[0] * (rho - x[2]) - x[1]) * dti
            dz = (x[0] * x[1] - beta * x[2]) * dti
            x_new = x + torch.stack([dx, dy, dz])
            X[s, t] = x_new
            Y[s, t, 0] = x_new[0]  # predict x(t)
            x = x_new
    return X, Y, Dt


def train_eval(model, x_tr, y_tr, dt_tr, x_te_reg, y_te_reg, x_te_irreg, y_te_irreg, dt_te_irreg,
               epochs=4, batch=8, lr=1e-2):
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
        test_mse_reg = nn.functional.mse_loss(model(x_te_reg, dt=1.0), y_te_reg).item()
        test_mse_irreg = nn.functional.mse_loss(model(x_te_irreg, dt=dt_te_irreg), y_te_irreg).item()
    return {"test_mse_regular": test_mse_reg, "test_mse_irregular": test_mse_irreg, "train_seconds": train_s}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=192)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--date", default="2026-08-05")
    args = parser.parse_args()

    n_feat = 3  # (x, y, z) of Lorenz
    out = 1
    hidden = args.hidden

    # Train on regular dt
    torch.manual_seed(0)
    x, y, _ = make_lorenz_dataset(args.n_samples, args.seq_len, seed=0, dt_jitter=0.0)
    n_tr = int(0.8 * x.shape[0])
    x_tr, y_tr = x[:n_tr], y[:n_tr]
    x_te_reg, y_te_reg = x[n_tr:], y[n_tr:]

    # Test on regular AND irregular dt
    torch.manual_seed(1)
    _, _, dt_te_irreg_reg = make_lorenz_dataset(x_te_reg.shape[0], args.seq_len, seed=1, dt_jitter=0.5)
    torch.manual_seed(2)
    _, _, dt_te_irreg_ood = make_lorenz_dataset(x_te_reg.shape[0], args.seq_len, seed=2, dt_jitter=1.0)
    # Use the same X (Lorenz trajectory) but with different dt sequences
    # The trained model will see the same X with perturbed dt.

    print(f"Task: Lorenz attractor (chaotic, x(t+dt) prediction)")
    print(f"sl={args.seq_len}, h={hidden}, n_tau=4 (for MR)")
    print(f"Test dt: regular (sigma=0), in-dist irregular (sigma=0.5), OOD irregular (sigma=1.0)\n")

    models = {
        "cfc-baseline":  lambda: _wrap(CfCCell(n_feat, hidden), hidden, out),
        "mfc-hybrid_gate": lambda: _wrap(
            MemoryFusionCfCCell(n_feat, hidden, retention_kind="hybrid_gate"), hidden, out
        ),
        "mr-hybrid-gate-cfc (n_tau=4)": lambda: MultiRateTfpCfCNetwork(
            n_feat, hidden, out, n_tau=4, top_k_active=2,
            expert_retention_kind="hybrid_gate",
        ),
    }
    results = {}
    # dt_tr must match x_tr's batch size (n_tr = int(0.8 * n_samples))
    dt_tr = torch.ones(x_tr.shape[0], args.seq_len)
    for name, factory in models.items():
        msess_reg, msess_irreg, msess_ood = [], [], []
        for r in range(args.repeats):
            torch.manual_seed(42 + r)
            model = factory()
            # Eval regular + in-dist irregular (sigma=0.5)
            torch.manual_seed(42 + r + 100)
            res_reg = train_eval(model, x_tr, y_tr, dt_tr,
                                 x_te_reg, y_te_reg, x_te_reg, y_te_reg, dt_te_irreg_reg,
                                 epochs=args.epochs)
            msess_reg.append(res_reg["test_mse_regular"])
            msess_irreg.append(res_reg["test_mse_irregular"])
            # Eval OOD dt (sigma=1.0)
            torch.manual_seed(42 + r + 100)
            res_ood = train_eval(model, x_tr, y_tr, dt_tr,
                                 x_te_reg, y_te_reg, x_te_reg, y_te_reg, dt_te_irreg_ood,
                                 epochs=args.epochs)
            msess_ood.append(res_ood["test_mse_irregular"])
        results[name] = {
            "regular_mse": statistics.mean(msess_reg),
            "in_dist_irreg_mse": statistics.mean(msess_irreg),
            "ood_irreg_mse": statistics.mean(msess_ood),
        }
        # Compute degradation ratios
        in_dist_degrad = results[name]["in_dist_irreg_mse"] / max(results[name]["regular_mse"], 1e-9)
        ood_degrad = results[name]["ood_irreg_mse"] / max(results[name]["regular_mse"], 1e-9)
        print(f"  {name:35s}: reg={results[name]['regular_mse']:.4f}, "
              f"in_dist={results[name]['in_dist_irreg_mse']:.4f} ({in_dist_degrad:.2f}x), "
              f"OOD={results[name]['ood_irreg_mse']:.4f} ({ood_degrad:.2f}x)")

    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.date}_lorenz_attractor.json"
    md_path = out_dir / f"{args.date}_lorenz_attractor.md"
    json_path.write_text(json.dumps({
        "date": args.date,
        "task": "lorenz_attractor_retention_validation",
        "config": vars(args),
        "results": results,
    }, indent=2))

    md = [f"""---
title: Lorenz attractor retention validation (N18) — {args.date}
date: {args.date}
tags: [LNN, CfC, TFP, hybrid_gate, lorenz, nonlinear-ODE, N18, real-world-validation]
---

# Lorenz attractor retention validation (N18) — {args.date}

## Setup
- Task: Lorenz attractor next-step prediction
  - dx/dt = sigma * (y - x)
  - dy/dt = x * (rho - z) - y
  - dz/dt = x * y - beta * z
  - sigma=10, rho=28, beta=8/3 → chaotic
- sl={args.seq_len}, h={hidden}
- Train dt: regular (sigma=0)
- Test dt: regular, in-dist irregular (sigma=0.5), OOD irregular (sigma=1.0)

## Results

| model | regular MSE | in-dist irregular | OOD irregular |
|---|---:|---:|---:|
"""]
    for name in models:
        r = results[name]
        in_dist = r["in_dist_irreg_mse"] / max(r["regular_mse"], 1e-9)
        ood = r["ood_irreg_mse"] / max(r["regular_mse"], 1e-9)
        md.append(f"| {name} | {r['regular_mse']:.4f} | "
                  f"{r['in_dist_irreg_mse']:.4f} ({in_dist:.2f}x) | "
                  f"{r['ood_irreg_mse']:.4f} ({ood:.2f}x) |\n")
    md_path.write_text("".join(md))
    print(f"\nWrote {json_path} and {md_path}")


def _wrap(cell, hidden, out):
    """Wrap a cell in a Sequence-to-Sequence net for training."""
    return _SeqWrap(cell, out)


class _SeqWrap(nn.Module):
    def __init__(self, cell, out_dim):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(cell.hidden_size, out_dim)

    def forward(self, x, dt=1.0):
        b, t, _ = x.shape
        h = torch.zeros(b, self.cell.hidden_size)
        outs = []
        for i in range(t):
            if isinstance(dt, torch.Tensor):
                # Defensive: align dt batch size with x batch size
                if dt.shape[0] != b:
                    # Take first b rows (shouldn't happen if dt is per-sample)
                    if dt.shape[0] > b:
                        dt = dt[:b]
                    else:
                        # Pad by repeating last row
                        pad = dt[-1:].expand(b - dt.shape[0], *dt.shape[1:])
                        dt = torch.cat([dt, pad], dim=0)
                dt_i = dt[:, i]
                # Ensure dt is (B, 1) for cell broadcast
                if dt_i.dim() == 1:
                    dt_i = dt_i.unsqueeze(-1)
            else:
                dt_i = dt
            h = self.cell(x[:, i, :], h, dt=dt_i)
            outs.append(self.head(h))
        return torch.stack(outs, dim=1)


if __name__ == "__main__":
    main()
