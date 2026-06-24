#!/usr/bin/env python3
"""Benchmark for PerBranchMultiBasinLyapunovCfCCell (round 248).

Trains four configs side-by-side on three toy datasets:

  * ``baseline``           — single-τ CfC
  * ``frozen_sampled``     — round 246 FrozenSampledMultiTauCfCCell
  * ``frozen_multibasin``  — round 247 (global basins)
  * ``per_branch``         — round 248 (per-branch basins, this round)

Hypotheses (round 248 PRD):
  H1 (composition safe): ``per_branch`` does not regress task loss by
      more than 10% relative to ``baseline`` on any dataset.
  H2 (basins used): mean per-branch basin entropy >= log(n_basin) * 0.5.
  H3 (compositional win over r246): at least one dataset has
      ``per_branch`` strictly better than ``frozen_sampled``.
  H4 (per-branch beats global): ``per_branch`` strictly better than
      ``frozen_multibasin`` on at least one dataset.
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

from lnn.core.cfc import CfCCell  # noqa: E402
from lnn.core.frozen_sampled_multitau_cfc import FrozenSampledMultiTauCfCCell  # noqa: E402
from lnn.core.frozen_multibasin_lyapunov_cfc import (  # noqa: E402
    FrozenMultiBasinLyapunovCfCCell,
)
from lnn.core.per_branch_multibasin_lyapunov_cfc import (  # noqa: E402
    PerBranchMultiBasinLyapunovCfCCell,
)

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
            self.cell = CfCCell(d_in, d_h)
        elif mode == "frozen_sampled":
            self.cell = FrozenSampledMultiTauCfCCell(
                d_in, d_h, n_branches=4, tau_min=0.05, tau_max=20.0, seed=42,
            )
        elif mode == "frozen_multibasin":
            self.cell = FrozenMultiBasinLyapunovCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
            )
        elif mode == "per_branch":
            self.cell = PerBranchMultiBasinLyapunovCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
            )
        else:
            raise ValueError(f"unknown mode {mode}")
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor
                 ) -> tuple[torch.Tensor, list[dict]]:
        B = x.shape[1]
        outs, aux_per_step = [], []
        if self.mode in ("frozen_sampled", "frozen_multibasin", "per_branch"):
            h_list = self.cell.init_state(B, device=x.device)
            for t in range(x.shape[0]):
                if self.mode == "per_branch":
                    h, h_list, aux = self.cell.forward_with_aux(
                        x[t], h_list, lyap_lambda=0.0, sep_lambda=0.0,
                    )
                elif self.mode == "frozen_multibasin":
                    h, h_list, aux = self.cell.forward_with_aux(
                        x[t], h_list, lyap_lambda=0.0, sep_lambda=0.0,
                    )
                else:
                    h, h_list, aux = self.cell.forward_with_aux(x[t], h_list)
                outs.append(self.head(h))
                aux_per_step.append(aux)
        else:
            h = torch.zeros(B, self.cell.hidden_size, device=x.device)
            for t in range(x.shape[0]):
                h = self.cell(x[t], h)
                outs.append(self.head(h))
                aux_per_step.append({})
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
        pred, aux = model(x)
        task = ((pred - y) ** 2).mean()
        loss = task
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        pred, aux = model(x)
        test_mse = ((pred - y) ** 2).mean().item()
        mean_H = 0.0
        if mode == "per_branch":
            Hs = [a.get("mean_basin_H", torch.tensor(0.0)).item()
                  for a in aux]
            mean_H = sum(Hs) / len(Hs) if Hs else 0.0
        elif mode == "frozen_multibasin":
            Hs = [a.get("basin_entropy", torch.tensor(0.0)).item()
                  for a in aux]
            mean_H = sum(Hs) / len(Hs) if Hs else 0.0

    return {
        "dataset": name,
        "seed": seed,
        "mode": mode,
        "test_mse": test_mse,
        "mean_basin_H": mean_H,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["toy_sin", "structured", "random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--d_h", type=int, default=9)
    parser.add_argument("--modes", nargs="+",
                        default=["baseline", "frozen_sampled",
                                 "frozen_multibasin", "per_branch"])
    parser.add_argument("--out", type=str,
                        default="analysis/per_branch_multibasin_lyapunov_cfc_bench.json")
    args = parser.parse_args()

    rows = []
    for mode in args.modes:
        for ds in args.datasets:
            for seed in args.seeds:
                r = train_one(ds, seed, mode,
                              epochs=args.epochs, d_h=args.d_h)
                rows.append(r)
                print(f"[{mode:>18s}] ds={ds:>9s} seed={seed} "
                      f"test_mse={r['test_mse']:.4f} "
                      f"mean_H={r['mean_basin_H']:.3f}")

    summary: dict = {"rows": rows, "verdict": {}}
    for ds in args.datasets:
        ds_rows = {m: [r for r in rows if r["dataset"] == ds
                       and r["mode"] == m] for m in args.modes}
        if not all(ds_rows.values()):
            continue
        means = {m: sum(r["test_mse"] for r in ds_rows[m]) / len(ds_rows[m])
                 for m in args.modes}
        per_branch_H = sum(r["mean_basin_H"] for r in ds_rows["per_branch"]
                           ) / len(ds_rows["per_branch"])
        delta_base = (means["per_branch"] - means["baseline"]) / max(
            means["baseline"], 1e-9) * 100.0
        delta_r246 = (means["per_branch"] - means["frozen_sampled"]) / max(
            means["frozen_sampled"], 1e-9) * 100.0
        delta_r247 = (means["per_branch"] - means["frozen_multibasin"]) / max(
            means["frozen_multibasin"], 1e-9) * 100.0
        summary["verdict"][ds] = {
            **{f"{m}_mse": means[m] for m in args.modes},
            "delta_pct_vs_baseline": delta_base,
            "delta_pct_vs_r246": delta_r246,
            "delta_pct_vs_r247": delta_r247,
            "per_branch_H_mean": per_branch_H,
            "log_n_basin": math.log(3),
            "h1_task_safe_vs_baseline": delta_base < 10.0,
            "h2_basin_used": per_branch_H >= math.log(3) * 0.5,
            "h3_compositional_win_vs_r246": delta_r246 < 0.0,
            "h4_per_branch_beats_global": delta_r247 < 0.0,
        }

    print("\n=== Verdict (round 248 hypotheses) ===")
    for ds, v in summary["verdict"].items():
        print(f"[{ds:>9s}] "
              f"per_branch={v['per_branch_mse']:.4f} "
              f"baseline={v['baseline_mse']:.4f} "
              f"r246={v['frozen_sampled_mse']:.4f} "
              f"r247={v['frozen_multibasin_mse']:.4f}  "
              f"Δ%_baseline={v['delta_pct_vs_baseline']:+.1f} "
              f"Δ%_r246={v['delta_pct_vs_r246']:+.1f} "
              f"Δ%_r247={v['delta_pct_vs_r247']:+.1f}  "
              f"H_per_branch={v['per_branch_H_mean']:.3f}/{v['log_n_basin']:.3f}  "
              f"H1={'✓' if v['h1_task_safe_vs_baseline'] else '✗'} "
              f"H2={'✓' if v['h2_basin_used'] else '✗'} "
              f"H3={'✓' if v['h3_compositional_win_vs_r246'] else '✗'} "
              f"H4={'✓' if v['h4_per_branch_beats_global'] else '✗'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()