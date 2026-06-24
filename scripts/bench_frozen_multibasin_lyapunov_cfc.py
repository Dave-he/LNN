#!/usr/bin/env python3
"""Benchmark for FrozenMultiBasinLyapunovCfCCell (round 247, composition 246 × 244).

Trains three configs side-by-side on three toy datasets:

  * ``baseline``           — single-τ CfC
  * ``frozen_sampled``     — round 246 FrozenSampledMultiTauCfCCell
  * ``frozen_multibasin``  — round 247 composition (this round)

For each cell we record:
  - ``test_mse``            — final test MSE
  - ``final_basin_entropy``— mean basin-assignment entropy (combined only)
  - ``final_V_mean``        — mean multi-basin Lyapunov value (combined only)

Hypotheses (round 247 PRD):
  H1 (composition safe): ``frozen_multibasin`` does not regress task
      loss by more than 10% relative to ``baseline`` on any dataset.
  H2 (basins used): ``final_basin_entropy >= log(n_basin) * 0.5`` —
      multi-basin structure is meaningful on top of frozen τ.
  H3 (compositional win): at least one dataset has frozen_multibasin
      Δ% strictly better than frozen_sampled alone.
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
        else:
            raise ValueError(f"unknown mode {mode}")
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor
                 ) -> tuple[torch.Tensor, list[dict]]:
        B = x.shape[1]
        outs, aux_per_step = [], []
        if self.mode in ("frozen_sampled", "frozen_multibasin"):
            h_list = self.cell.init_state(B, device=x.device)
            for t in range(x.shape[0]):
                if self.mode == "frozen_multibasin":
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
        if mode == "frozen_multibasin":
            for a in aux:
                if "lyap_loss_total" in a:
                    loss = loss + a["lyap_loss_total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        pred, aux = model(x)
        test_mse = ((pred - y) ** 2).mean().item()
        basin_ent = 0.0
        V_mean = 0.0
        if mode == "frozen_multibasin":
            ents = [a.get("basin_entropy", torch.tensor(0.0)).item()
                    for a in aux]
            basin_ent = sum(ents) / len(ents) if ents else 0.0
            vs = [a.get("V_next", torch.tensor(0.0)) for a in aux]
            V_mean = (sum(vs) / len(vs)).item() if vs else 0.0

    return {
        "dataset": name,
        "seed": seed,
        "mode": mode,
        "test_mse": test_mse,
        "final_basin_entropy": basin_ent,
        "final_V_mean": V_mean,
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
                                 "frozen_multibasin"])
    parser.add_argument("--out", type=str,
                        default="analysis/frozen_multibasin_lyapunov_cfc_bench.json")
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
                      f"basin_H={r['final_basin_entropy']:.3f} "
                      f"V_mean={r['final_V_mean']:.3f}")

    summary: dict = {"rows": rows, "verdict": {}}
    for ds in args.datasets:
        baseline = [r for r in rows if r["dataset"] == ds
                    and r["mode"] == "baseline"]
        fs = [r for r in rows if r["dataset"] == ds
              and r["mode"] == "frozen_sampled"]
        fmb = [r for r in rows if r["dataset"] == ds
               and r["mode"] == "frozen_multibasin"]
        if not baseline or not fmb:
            continue
        base_mse = sum(r["test_mse"] for r in baseline) / len(baseline)
        fs_mse = sum(r["test_mse"] for r in fs) / max(len(fs), 1)
        fmb_mse = sum(r["test_mse"] for r in fmb) / len(fmb)
        delta_base = (fmb_mse - base_mse) / max(base_mse, 1e-9) * 100.0
        delta_fs = (fmb_mse - fs_mse) / max(fs_mse, 1e-9) * 100.0
        basin_ent = sum(r["final_basin_entropy"] for r in fmb) / len(fmb)
        V_mean = sum(r["final_V_mean"] for r in fmb) / len(fmb)
        summary["verdict"][ds] = {
            "baseline_mse": base_mse,
            "frozen_sampled_mse": fs_mse,
            "frozen_multibasin_mse": fmb_mse,
            "delta_pct_vs_baseline": delta_base,
            "delta_pct_vs_frozen_sampled": delta_fs,
            "basin_entropy_mean": basin_ent,
            "V_mean": V_mean,
            "log_n_basin": math.log(3),
            "h1_task_safe": abs(delta_base) <= 10.0,
            "h2_basin_used": basin_ent >= math.log(3) * 0.5,
            "h3_compositional_win": delta_fs < 0.0,  # negative = better
        }

    print("\n=== Verdict (round 247 hypotheses) ===")
    for ds, v in summary["verdict"].items():
        print(f"[{ds:>9s}] Δ%_vs_baseline={v['delta_pct_vs_baseline']:+.1f} "
              f"Δ%_vs_frozen_sampled={v['delta_pct_vs_frozen_sampled']:+.1f} "
              f"basin_H={v['basin_entropy_mean']:.3f}/{v['log_n_basin']:.3f} "
              f"V_mean={v['V_mean']:.3f}  "
              f"H1={'✓' if v['h1_task_safe'] else '✗'} "
              f"H2={'✓' if v['h2_basin_used'] else '✗'} "
              f"H3={'✓' if v['h3_compositional_win'] else '✗'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()