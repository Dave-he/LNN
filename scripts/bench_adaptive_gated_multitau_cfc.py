#!/usr/bin/env python3
"""Benchmark for AdaptiveGatedMultiTauCfCCell (arXiv:2606.22801 response, round 243).

Trains three configs side-by-side on three toy datasets:

  * ``baseline``     — single-τ CfC (n_tau=1)
  * ``n_tau``        — round 76 multi-τ CfC (n_tau=3, static τ, equal fuse)
  * ``adaptive_gated``— round 243 adaptive τ + gated fusion

For each cell we record:
  - ``test_mse``           — final test MSE (B×T average)
  - ``final_gate_entropy`` — branch-fusion gate entropy (H ∈ [0, log n_tau])
  - ``final_tau_eff_mean`` — mean τ per branch averaged over T
  - ``final_tau_eff_std``  — std of τ across inputs (input-conditioned spread)

Hypotheses (round 243 PRD):
  H1 (task safe): ``adaptive_gated`` does not regress task loss by more
      than 10% relative to ``baseline`` on any dataset.
  H2 (gate entropy healthy): ``final_gate_entropy >= log(n_tau) * 0.5``
      — gate is not collapsed to a single branch.
  H3 (τ input-conditioned): ``final_tau_eff_std / final_tau_eff_mean >= 0.1``
      — τ actually varies across inputs (not a constant).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lnn.core.adaptive_gated_multitau_cfc import (  # noqa: E402
    AdaptiveGatedMultiTauCfCCell,
    gated_fusion_entropy,
)
from lnn.core.cfc import CfCCell  # noqa: E402

DEVICE = "cpu"


def make_dataset(name: str, T: int = 64, d_in: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(-1)
    if name == "toy_sin":
        x = torch.sin(t)
        y = torch.cos(t)
    elif name == "structured":
        x = torch.sin(t) + 0.3 * torch.sin(3 * t)
        y = torch.roll(x, shifts=1, dims=0)
    elif name == "random":
        x = torch.randn(T, d_in, generator=g)
        y = torch.roll(x, shifts=1, dims=0)
    else:
        raise ValueError(f"unknown dataset {name}")
    return x, y


class SeqModel(nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int, mode: str):
        super().__init__()
        self.mode = mode
        self.d_h = d_h
        if mode == "baseline":
            self.cell = CfCCell(d_in, d_h, n_tau=1)
            self._uses_aux = False
        elif mode == "n_tau":
            self.cell = CfCCell(d_in, d_h, n_tau=3, tau_scales=(0.1, 1.0, 10.0))
            self._uses_aux = False
        elif mode == "adaptive_gated":
            self.cell = AdaptiveGatedMultiTauCfCCell(
                d_in, d_h, n_tau=3, tau_base=(0.1, 1.0, 10.0),
            )
            self._uses_aux = True
        else:
            raise ValueError(f"unknown mode {mode}")
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor
                 ) -> tuple[torch.Tensor, list[dict]]:
        B = x.shape[1]
        h = torch.zeros(B, self.cell.hidden_size, device=x.device)
        outs, aux_per_step = [], []
        for t in range(x.shape[0]):
            if self._uses_aux:
                h, aux = self.cell.forward_with_aux(x[t], h)
                aux_per_step.append(aux)
            else:
                h = self.cell(x[t], h)
                # Compute gate-entropy diagnostic = 0 for non-gated cells.
                aux_per_step.append({
                    "gate_entropy": torch.tensor(0.0),
                    "tau_eff": torch.zeros(B, 1),
                })
            outs.append(self.head(h))
        return torch.stack(outs), aux_per_step


def train_one(name: str, seed: int, mode: str, epochs: int = 100, d_h: int = 9):
    torch.manual_seed(seed)
    x, y = make_dataset(name, seed=seed)
    x = x.unsqueeze(1).to(DEVICE)
    y = y.unsqueeze(1).to(DEVICE)

    model = SeqModel(d_in=x.shape[-1], d_h=d_h, d_out=y.shape[-1],
                     mode=mode).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(epochs):
        opt.zero_grad()
        pred, _ = model(x)
        task = ((pred - y) ** 2).mean()
        task.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        pred, aux = model(x)
        test_mse = ((pred - y) ** 2).mean().item()
        gate_ent = float(
            torch.stack([a["gate_entropy"] for a in aux]).mean().item()
        )
        tau_stack = torch.stack([a["tau_eff"] for a in aux])  # (T, B, n_tau)
        tau_mean = tau_stack.mean(dim=(0, 1))  # (n_tau,)
        tau_std = tau_stack.std(dim=(0, 1))    # (n_tau,)

    return {
        "dataset": name,
        "seed": seed,
        "mode": mode,
        "test_mse": test_mse,
        "final_gate_entropy": gate_ent,
        "final_tau_eff_mean": float(tau_mean.mean().item()),
        "final_tau_eff_std": float(tau_std.mean().item()),
        "final_tau_eff_per_branch": tau_mean.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["toy_sin", "structured", "random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--d_h", type=int, default=9)
    parser.add_argument("--modes", nargs="+",
                        default=["baseline", "n_tau", "adaptive_gated"])
    parser.add_argument("--out", type=str,
                        default="analysis/adaptive_gated_multitau_cfc_bench.json")
    args = parser.parse_args()

    rows = []
    for mode in args.modes:
        for ds in args.datasets:
            for seed in args.seeds:
                r = train_one(ds, seed, mode,
                              epochs=args.epochs, d_h=args.d_h)
                rows.append(r)
                print(f"[{mode:>14s}] ds={ds:>9s} seed={seed} "
                      f"test_mse={r['test_mse']:.4f} "
                      f"gate_H={r['final_gate_entropy']:.3f} "
                      f"tau={r['final_tau_eff_mean']:.3f}±"
                      f"{r['final_tau_eff_std']:.3f}")

    # Verdict
    summary: dict = {"rows": rows, "verdict": {}}
    for ds in args.datasets:
        baseline = [r for r in rows if r["dataset"] == ds
                    and r["mode"] == "baseline"]
        if not baseline:
            continue
        base_mse = sum(r["test_mse"] for r in baseline) / len(baseline)
        adaptive = [r for r in rows if r["dataset"] == ds
                    and r["mode"] == "adaptive_gated"]
        if not adaptive:
            continue
        ad_mse = sum(r["test_mse"] for r in adaptive) / len(adaptive)
        delta = (ad_mse - base_mse) / max(base_mse, 1e-9) * 100.0
        gate_h = sum(r["final_gate_entropy"] for r in adaptive) / len(adaptive)
        tau_var = sum(
            r["final_tau_eff_std"] / max(r["final_tau_eff_mean"], 1e-9)
            for r in adaptive
        ) / len(adaptive)
        n_tau = adaptive[0]["final_tau_eff_per_branch"]
        summary["verdict"][ds] = {
            "baseline_mse": base_mse,
            "adaptive_mse": ad_mse,
            "delta_pct": delta,
            "mean_gate_entropy": gate_h,
            "log_n_tau": math.log(len(n_tau)),
            "tau_cv": tau_var,
            "h1_task_safe": abs(delta) <= 10.0,
            "h2_gate_entropy_healthy": gate_h >= math.log(len(n_tau)) * 0.5,
            "h3_tau_input_conditioned": tau_var >= 0.1,
        }

    print("\n=== Verdict (round 243 hypotheses) ===")
    for ds, v in summary["verdict"].items():
        print(f"[{ds:>9s}] Δ%={v['delta_pct']:+.1f} "
              f"gate_H={v['mean_gate_entropy']:.3f}/{v['log_n_tau']:.3f} "
              f"tau_cv={v['tau_cv']:.3f}  "
              f"H1={'✓' if v['h1_task_safe'] else '✗'} "
              f"H2={'✓' if v['h2_gate_entropy_healthy'] else '✗'} "
              f"H3={'✓' if v['h3_tau_input_conditioned'] else '✗'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()