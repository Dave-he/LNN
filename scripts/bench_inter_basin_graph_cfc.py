#!/usr/bin/env python3
"""Benchmark for InterBasinGraphCfCCell (round 258).

Closes the structural gap from
``docs/research/2026-06-25_round257_bridge_to_neuronwise_research.md``:
after r257 separated basins geometrically, round 258 adds a **learned
sparse basin adjacency** A ∈ R^{K×K} that mediates inter-basin message
passing within each branch.

Modes (6 total):
  * baseline                 (CfC, no branches)
  * r248_per_branch          (no repulsion, no graph)
  * r257_d2                  (current best — geometric repulsion only)
  * r258_graph_only          (graph mix, no aux regularizers, graph_lambda=0.0
                              in loss but the adjacency is still trained via
                              the cross-entropy against basin probabilities)
  * r258_sym                 (graph mix + sym_lambda=1.0)
  * r258_symsp               (graph mix + sym_lambda=1.0, sparse_lambda=0.5)

Hypotheses (PRD #10-95):
  H1: graph mix INCREASES basin selectivity (lower H_per_branch final
      vs r257) — directed graph biases routing toward a subset.
  H2: r258 with regularizers matches or beats r257_d2 on toy_sin /
      random while preserving structured gains.
  H3: the learned adjacency becomes ASYMMETRIC (||A - A^T||_F > 0.1)
      and SPARSE (avg off-diag |A| < 0.1) after training.

Note on graph_lambda: we always compute the graph_loss_total in aux but
only add it to the training loss when graph_lambda > 0. The "graph_only"
mode uses graph_lambda=0 (no extra loss), letting the adjacency learn
purely through basin-assignment gradients; the regularized modes use
graph_lambda=1.
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
            # Graph mix active but no sym/sparse regularizers.
            self.cell = InterBasinGraphCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                d_min=2.0, sym_lambda=0.0, sparse_lambda=0.0,
            )
        elif mode == "r258_sym":
            self.cell = InterBasinGraphCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                d_min=2.0, sym_lambda=1.0, sparse_lambda=0.0,
            )
        elif mode == "r258_symsp":
            self.cell = InterBasinGraphCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                d_min=2.0, sym_lambda=1.0, sparse_lambda=0.5,
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
        elif self.mode == "r257_d2":
            h_list = self.cell.init_state(B, device=x.device)
            for t in range(x.shape[0]):
                h, h_list, aux = self.cell.forward_with_aux(
                    x[t], h_list, lyap_lambda=0.0, sep_lambda=0.0,
                    dist_lambda=1.0,
                )
                outs.append(self.head(h))
                aux_per_step.append(aux)
        elif self.mode in ("r258_graph_only", "r258_sym", "r258_symsp"):
            h_list = self.cell.init_state(B, device=x.device)
            for t in range(x.shape[0]):
                h, h_list, aux = self.cell.forward_with_aux(
                    x[t], h_list, lyap_lambda=0.0, sep_lambda=0.0,
                    dist_lambda=1.0,
                    graph_lambda=1.0 if self.mode != "r258_graph_only" else 0.0,
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
        "r257_d2", "r258_graph_only", "r258_sym", "r258_symsp",
    )

    H_first = 0.0
    H_last = 0.0
    sym_last = 0.0
    sparse_last = 0.0
    H_raw_last = 0.0
    dist_first = 0.0
    dist_last = 0.0

    for ep in range(epochs):
        opt.zero_grad()
        pred, aux = model(x)
        task = ((pred - y) ** 2).mean()
        loss = task
        if mode in aux_modes:
            if mode == "r257_d2":
                ival = [a.get("inter_basin_loss", torch.tensor(0.0))
                        for a in aux]
                dist_loss = sum(ival) / max(len(ival), 1)
                loss = loss + dist_loss
                if ep == 0:
                    dist_first = dist_loss.item()
                    H_first = sum(
                        a.get("mean_basin_H", torch.tensor(0.0)).item()
                        for a in aux
                    ) / max(len(aux), 1)
                if ep == epochs - 1:
                    dist_last = dist_loss.item()
                    H_last = sum(
                        a.get("mean_basin_H", torch.tensor(0.0)).item()
                        for a in aux
                    ) / max(len(aux), 1)
            else:
                ival = [a.get("inter_basin_loss", torch.tensor(0.0))
                        for a in aux]
                dist_loss = sum(ival) / max(len(ival), 1)
                gval = [a.get("graph_loss_total", torch.tensor(0.0))
                        for a in aux]
                g_loss = sum(gval) / max(len(gval), 1)
                loss = loss + dist_loss + g_loss
                if ep == 0:
                    dist_first = dist_loss.item()
                    H_first = sum(
                        a.get("mean_basin_H", torch.tensor(0.0)).item()
                        for a in aux
                    ) / max(len(aux), 1)
                if ep == epochs - 1:
                    dist_last = dist_loss.item()
                    H_last = sum(
                        a.get("mean_basin_H", torch.tensor(0.0)).item()
                        for a in aux
                    ) / max(len(aux), 1)
                    H_raw_last = sum(
                        a.get("mean_basin_H_raw", torch.tensor(0.0)).item()
                        for a in aux
                    ) / max(len(aux), 1)
                    sym_last = aux[0]["graph_symmetry"].item()
                    sparse_last = aux[0]["graph_sparsity"].item()
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
        "dist_first": dist_first,
        "dist_last": dist_last,
        "H_first": H_first,
        "H_last": H_last,
        "H_raw_last": H_raw_last,
        "sym_last": sym_last,
        "sparse_last": sparse_last,
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
                                 "r258_graph_only", "r258_sym",
                                 "r258_symsp"])
    parser.add_argument("--out", type=str,
                        default="analysis/inter_basin_graph_cfc_bench.json")
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
                      f"dist_first={r['dist_first']:.3f} "
                      f"dist_last={r['dist_last']:.3f} "
                      f"H_first={r['H_first']:.3f} "
                      f"H_last={r['H_last']:.3f} "
                      f"H_raw_last={r['H_raw_last']:.3f} "
                      f"sym={r['sym_last']:.3f} "
                      f"sp={r['sparse_last']:.3f}")

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
        sym_means = {m: sum(r["sym_last"] for r in ds_rows
                            if r["mode"] == m) / max(
            len([r for r in ds_rows if r["mode"] == m]), 1)
                     for m in args.modes}
        sp_means = {m: sum(r["sparse_last"] for r in ds_rows
                           if r["mode"] == m) / max(
            len([r for r in ds_rows if r["mode"] == m]), 1)
                    for m in args.modes}
        summary["verdict"][ds] = {
            **{f"{m}_mse": means[m] for m in args.modes},
            **{f"{m}_H": H_means[m] for m in args.modes},
            **{f"{m}_Hraw": H_raw_means[m] for m in args.modes},
            **{f"{m}_sym": sym_means[m] for m in args.modes},
            **{f"{m}_sparse": sp_means[m] for m in args.modes},
        }

    print("\n=== Verdict (round 258 — InterBasinGraph) ===")
    mode_order = ["baseline", "r248_per_branch", "r257_d2",
                  "r258_graph_only", "r258_sym", "r258_symsp"]
    for ds, v in summary["verdict"].items():
        mse_parts = [f"{m}={v.get(f'{m}_mse', 0):.4f}" for m in mode_order]
        H_parts = [f"H_{m}={v.get(f'{m}_H', 0):.3f}" for m in mode_order]
        Hraw_parts = [f"Hraw_{m}={v.get(f'{m}_Hraw', 0):.3f}"
                      for m in mode_order]
        sym_parts = [f"sym_{m}={v.get(f'{m}_sym', 0):.3f}"
                     for m in mode_order
                     if m.startswith("r258")]
        print(f"[{ds:>9s}] " + " ".join(mse_parts))
        print(f"           " + " ".join(H_parts))
        print(f"           " + " ".join(Hraw_parts))
        if sym_parts:
            print(f"           " + " ".join(sym_parts))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()