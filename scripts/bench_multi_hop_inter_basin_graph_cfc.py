#!/usr/bin/env python3
"""Benchmark for MultiHopInterBasinGraphCfCCell (round 259).

Tests whether **graph depth** (n_hops iterations of inter-basin message
passing) matters. Inherits r258's adjacency A; iterates the
graph-mix operation n_hops times.

Modes (6 total):
  * baseline                 (CfC, no branches)
  * r248_per_branch          (no repulsion, no graph)
  * r257_d2                  (current best — geometric repulsion only)
  * r258_graph_only          (r258 = n_hops=1)
  * r259_hop2                (NEW, n_hops=2 — 2-hop message passing)
  * r259_hop3                (NEW, n_hops=3 — 3-hop)

Hypotheses (PRD #10-96):
  H1: n_hops=2 marginally beats n_hops=1 on structured (more propagation
      helps on multi-modal data).
  H2: n_hops=3+ over-smooths — H → log K, losing basin selectivity.
  H3: r259_hop2 is the new best on structured (extends r258's win).
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
from lnn.core.inter_basin_distance_cfc import (  # noqa: E402
    InterBasinDistanceCfCCell,
)
from lnn.core.inter_basin_graph_cfc import (  # noqa: E402
    InterBasinGraphCfCCell,
)
from lnn.core.multi_hop_inter_basin_graph_cfc import (  # noqa: E402
    MultiHopInterBasinGraphCfCCell,
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
        elif mode == "r248_per_branch":
            self.cell = PerBranchMultiBasinLyapunovCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
            )
        elif mode == "r257_d2":
            self.cell = InterBasinDistanceCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                d_min=2.0,
            )
        elif mode == "r258_graph_only":
            self.cell = InterBasinGraphCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                d_min=2.0, sym_lambda=0.0, sparse_lambda=0.0,
            )
        elif mode == "r259_hop2":
            self.cell = MultiHopInterBasinGraphCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                d_min=2.0, n_hops=2,
                sym_lambda=0.0, sparse_lambda=0.0,
            )
        elif mode == "r259_hop3":
            self.cell = MultiHopInterBasinGraphCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                d_min=2.0, n_hops=3,
                sym_lambda=0.0, sparse_lambda=0.0,
            )
        else:
            raise ValueError(f"unknown mode {mode}")
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor
                 ) -> tuple[torch.Tensor, list[dict]]:
        B = x.shape[1]
        outs, aux_per_step = [], []
        if self.mode == "r248_per_branch":
            h_list = self.cell.init_state(B, device=x.device)
            for t in range(x.shape[0]):
                h, h_list, aux = self.cell.forward_with_aux(
                    x[t], h_list, lyap_lambda=0.0, sep_lambda=0.0,
                )
                outs.append(self.head(h))
                aux_per_step.append(aux)
        elif self.mode in ("r257_d2", "r258_graph_only",
                           "r259_hop2", "r259_hop3"):
            h_list = self.cell.init_state(B, device=x.device)
            for t in range(x.shape[0]):
                if self.mode == "r257_d2":
                    h, h_list, aux = self.cell.forward_with_aux(
                        x[t], h_list, lyap_lambda=0.0, sep_lambda=0.0,
                        dist_lambda=1.0,
                    )
                else:
                    h, h_list, aux = self.cell.forward_with_aux(
                        x[t], h_list, lyap_lambda=0.0, sep_lambda=0.0,
                        dist_lambda=1.0, graph_lambda=0.0,
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

    aux_modes = (
        "r257_d2", "r258_graph_only", "r259_hop2", "r259_hop3",
    )

    H_first = 0.0
    H_last = 0.0
    H_raw_last = 0.0

    for ep in range(epochs):
        opt.zero_grad()
        pred, aux = model(x)
        task = ((pred - y) ** 2).mean()
        loss = task
        if mode in aux_modes:
            ival = [a.get("inter_basin_loss", torch.tensor(0.0))
                    for a in aux]
            dist_loss = sum(ival) / max(len(ival), 1)
            loss = loss + dist_loss
            if ep == 0:
                H_first = sum(
                    a.get("mean_basin_H", torch.tensor(0.0)).item()
                    for a in aux
                ) / max(len(aux), 1)
            if ep == epochs - 1:
                H_last = sum(
                    a.get("mean_basin_H", torch.tensor(0.0)).item()
                    for a in aux
                ) / max(len(aux), 1)
                H_raw_last = sum(
                    a.get("mean_basin_H_raw", torch.tensor(0.0)).item()
                    for a in aux
                ) / max(len(aux), 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        pred, aux = model(x)
        test_mse = ((pred - y) ** 2).mean().item()

    return {
        "dataset": name,
        "seed": seed,
        "mode": mode,
        "test_mse": test_mse,
        "H_first": H_first,
        "H_last": H_last,
        "H_raw_last": H_raw_last,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["toy_sin", "structured", "random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--d_h", type=int, default=9)
    parser.add_argument("--modes", nargs="+",
                        default=["baseline", "r248_per_branch", "r257_d2",
                                 "r258_graph_only", "r259_hop2", "r259_hop3"])
    parser.add_argument("--out", type=str,
                        default="analysis/multi_hop_inter_basin_graph_cfc_bench.json")
    args = parser.parse_args()

    rows = []
    for mode in args.modes:
        for ds in args.datasets:
            for seed in args.seeds:
                r = train_one(ds, seed, mode,
                              epochs=args.epochs, d_h=args.d_h)
                rows.append(r)
                print(f"[{mode:>16s}] ds={ds:>9s} seed={seed} "
                      f"test_mse={r['test_mse']:.4f} "
                      f"H_first={r['H_first']:.3f} "
                      f"H_last={r['H_last']:.3f} "
                      f"H_raw_last={r['H_raw_last']:.3f}")

    summary: dict = {"rows": rows, "verdict": {}}
    for ds in args.datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if not ds_rows:
            continue
        means = {m: sum(r["test_mse"] for r in ds_rows
                        if r["mode"] == m) / max(
            len([r for r in ds_rows if r["mode"] == m]), 1)
                 for m in args.modes}
        H_means = {m: sum(r["H_last"] for r in ds_rows
                          if r["mode"] == m) / max(
            len([r for r in ds_rows if r["mode"] == m]), 1)
                   for m in args.modes}
        H_raw_means = {m: sum(r["H_raw_last"] for r in ds_rows
                              if r["mode"] == m) / max(
            len([r for r in ds_rows if r["mode"] == m]), 1)
                       for m in args.modes}
        summary["verdict"][ds] = {
            **{f"{m}_mse": means[m] for m in args.modes},
            **{f"{m}_H": H_means[m] for m in args.modes},
            **{f"{m}_Hraw": H_raw_means[m] for m in args.modes},
        }

    print("\n=== Verdict (round 259 — MultiHopInterBasinGraph) ===")
    mode_order = ["baseline", "r248_per_branch", "r257_d2",
                  "r258_graph_only", "r259_hop2", "r259_hop3"]
    for ds, v in summary["verdict"].items():
        mse_parts = [f"{m}={v.get(f'{m}_mse', 0):.4f}" for m in mode_order]
        H_parts = [f"H_{m}={v.get(f'{m}_H', 0):.3f}" for m in mode_order]
        Hraw_parts = [f"Hraw_{m}={v.get(f'{m}_Hraw', 0):.3f}"
                      for m in mode_order]
        print(f"[{ds:>9s}] " + " ".join(mse_parts))
        print(f"           " + " ".join(H_parts))
        print(f"           " + " ".join(Hraw_parts))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()