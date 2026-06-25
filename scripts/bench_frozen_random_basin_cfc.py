#!/usr/bin/env python3
"""Benchmark for FrozenRandomBasinCfCCell (round 250).

Tests whether FROZEN random basin centers (not learned) can match
round 248's learned-basin performance. Compares:

  * ``baseline``           — single-τ CfC
  * ``per_branch``         — round 248 (LEARNED basin centers)
  * ``frozen_random_basin`` — round 250 (FROZEN random basin centers)

Hypotheses (round 250 PRD):
  H1 (safe): frozen_random_basin does not regress baseline by >10%.
  H2 (parity with r248): within ±10% of r248 on every dataset.
  H3 (random basins used): mean_basin_H >= log(n_basin) * 0.5.
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
from lnn.core.per_branch_multibasin_lyapunov_cfc import (  # noqa: E402
    PerBranchMultiBasinLyapunovCfCCell,
)
from lnn.core.frozen_random_basin_cfc import (  # noqa: E402
    FrozenRandomBasinCfCCell,
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
        elif mode == "per_branch":
            self.cell = PerBranchMultiBasinLyapunovCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
            )
        elif mode == "frozen_random_basin":
            self.cell = FrozenRandomBasinCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, tau_seed=42, basin_seed=137,
            )
        else:
            raise ValueError(f"unknown mode {mode}")
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor
                 ) -> tuple[torch.Tensor, list[dict]]:
        B = x.shape[1]
        outs, aux_per_step = [], []
        if self.mode in ("per_branch", "frozen_random_basin"):
            h_list = self.cell.init_state(B, device=x.device)
            for t in range(x.shape[0]):
                h, h_list, aux = self.cell.forward_with_aux(
                    x[t], h_list, lyap_lambda=0.0, sep_lambda=0.0,
                )
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
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        pred, aux = model(x)
        test_mse = ((pred - y) ** 2).mean().item()
        mean_H = 0.0
        Hs = [a.get("mean_basin_H", torch.tensor(0.0)).item() for a in aux]
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
                        default=["baseline", "per_branch",
                                 "frozen_random_basin"])
    parser.add_argument("--out", type=str,
                        default="analysis/frozen_random_basin_cfc_bench.json")
    args = parser.parse_args()

    rows = []
    for mode in args.modes:
        for ds in args.datasets:
            for seed in args.seeds:
                r = train_one(ds, seed, mode,
                              epochs=args.epochs, d_h=args.d_h)
                rows.append(r)
                print(f"[{mode:>20s}] ds={ds:>9s} seed={seed} "
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
        mean_H = sum(r["mean_basin_H"] for r in ds_rows["frozen_random_basin"]
                     ) / len(ds_rows["frozen_random_basin"])
        delta_base = (means["frozen_random_basin"] - means["baseline"]) / max(
            means["baseline"], 1e-9) * 100.0
        delta_r248 = (means["frozen_random_basin"] - means["per_branch"]) / max(
            means["per_branch"], 1e-9) * 100.0
        summary["verdict"][ds] = {
            **{f"{m}_mse": means[m] for m in args.modes},
            "delta_pct_vs_baseline": delta_base,
            "delta_pct_vs_r248": delta_r248,
            "mean_basin_H": mean_H,
            "log_n_basin": math.log(3),
            "h1_safe_vs_baseline": delta_base < 10.0,
            "h2_parity_with_r248": abs(delta_r248) <= 10.0,
            "h3_basins_used": mean_H >= math.log(3) * 0.5,
        }

    print("\n=== Verdict (round 250 hypotheses) ===")
    for ds, v in summary["verdict"].items():
        print(f"[{ds:>9s}] "
              f"frozen_random_basin={v['frozen_random_basin_mse']:.4f} "
              f"per_branch={v['per_branch_mse']:.4f} "
              f"baseline={v['baseline_mse']:.4f}  "
              f"Δ%_baseline={v['delta_pct_vs_baseline']:+.1f} "
              f"Δ%_r248={v['delta_pct_vs_r248']:+.1f}  "
              f"mean_H={v['mean_basin_H']:.3f}/{v['log_n_basin']:.3f}  "
              f"H1={'✓' if v['h1_safe_vs_baseline'] else '✗'} "
              f"H2={'✓' if v['h2_parity_with_r248'] else '✗'} "
              f"H3={'✓' if v['h3_basins_used'] else '✗'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()