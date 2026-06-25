#!/usr/bin/env python3
"""Benchmark for PerStepInterBasinGraphCfCCell (round 260).

Tests whether **input-dependent adjacency** A_t = MLP(x_t) improves over
r258's static learned adjacency.

Modes (7 total):
  * baseline                 (CfC, no branches)
  * r248_per_branch          (no repulsion, no graph)
  * r257_d2                  (geometric repulsion only)
  * r258_graph_only          (r258 = static A)
  * r260_perstep             (NEW, A_t = softmax(MLP(x_t)))
  * r260_perstep_static_bias (NEW, A_t = softmax(static_logits + MLP(x_t)))
  * r260_perstep_h4          (NEW, with mlp_hidden=4)

Hypotheses (PRD #10-97):
  H1: r260_perstep beats r258 on structured (input carries basin info).
  H2: r260_perstep H_per_branch is MORE VARIABLE across timesteps than r258.
  H3: r260_perstep is at least as robust as r258 on random (no overfitting).
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
from lnn.core.per_step_inter_basin_graph_cfc import (  # noqa: E402
    PerStepInterBasinGraphCfCCell,
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
        elif mode == "r260_perstep":
            self.cell = PerStepInterBasinGraphCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                d_min=2.0, sym_lambda=0.0, sparse_lambda=0.0,
            )
        elif mode == "r260_perstep_h4":
            self.cell = PerStepInterBasinGraphCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                d_min=2.0, sym_lambda=0.0, sparse_lambda=0.0,
                mlp_hidden=4,
            )
        else:
            raise ValueError(f"unknown mode {mode}")
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor
                 ) -> tuple[torch.Tensor, list[dict]]:
        B = x.shape[1]
        outs, aux_per_step = [], []
        if self.mode in ("r248_per_branch", "r257_d2", "r258_graph_only",
                          "r260_perstep", "r260_perstep_h4"):
            h_list = self.cell.init_state(B, device=x.device)
            for t in range(x.shape[0]):
                if self.mode == "r248_per_branch":
                    h, h_list, aux = self.cell.forward_with_aux(
                        x[t], h_list, lyap_lambda=0.0, sep_lambda=0.0,
                    )
                elif self.mode == "r257_d2":
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
        "r248_per_branch", "r257_d2", "r258_graph_only",
        "r260_perstep", "r260_perstep_h4",
    )

    H_first = 0.0
    H_last = 0.0
    H_raw_last = 0.0
    H_std = 0.0
    A_diversity_last = 0.0

    for ep in range(epochs):
        opt.zero_grad()
        pred, aux = model(x)
        task = ((pred - y) ** 2).mean()
        loss = task
        if mode in ("r257_d2", "r258_graph_only",
                    "r260_perstep", "r260_perstep_h4"):
            ival = [a.get("inter_basin_loss", torch.tensor(0.0))
                    for a in aux]
            dist_loss = sum(ival) / max(len(ival), 1)
            loss = loss + dist_loss
        if mode in aux_modes and ep == 0:
            H_first = sum(
                a.get("mean_basin_H", torch.tensor(0.0)).item()
                for a in aux
            ) / max(len(aux), 1)
        if mode in aux_modes and ep == epochs - 1:
            H_last = sum(
                a.get("mean_basin_H", torch.tensor(0.0)).item()
                for a in aux
            ) / max(len(aux), 1)
            H_raw_last = sum(
                a.get("mean_basin_H_raw", torch.tensor(0.0)).item()
                for a in aux
            ) / max(len(aux), 1)
            # H_per_timestep std (NEW, tests H2).
            H_series = torch.tensor([
                a.get("mean_basin_H", torch.tensor(0.0)).item()
                for a in aux
            ])
            H_std = H_series.std().item()
            if mode in ("r260_perstep", "r260_perstep_h4"):
                A_diversity_last = sum(
                    a.get("A_diversity", torch.tensor(0.0)).item()
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
        "H_std": H_std,
        "A_diversity_last": A_diversity_last,
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
                                 "r258_graph_only",
                                 "r260_perstep", "r260_perstep_h4"])
    parser.add_argument("--out", type=str,
                        default="analysis/per_step_inter_basin_graph_cfc_bench.json")
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
                      f"H_last={r['H_last']:.3f} "
                      f"H_std={r['H_std']:.4f} "
                      f"A_div={r['A_diversity_last']:.4f}")

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
        H_std_means = {m: sum(r["H_std"] for r in ds_rows
                              if r["mode"] == m) / max(
            len([r for r in ds_rows if r["mode"] == m]), 1)
                       for m in args.modes}
        A_div_means = {m: sum(r["A_diversity_last"] for r in ds_rows
                              if r["mode"] == m) / max(
            len([r for r in ds_rows if r["mode"] == m]), 1)
                       for m in args.modes}
        summary["verdict"][ds] = {
            **{f"{m}_mse": means[m] for m in args.modes},
            **{f"{m}_H": H_means[m] for m in args.modes},
            **{f"{m}_Hstd": H_std_means[m] for m in args.modes},
            **{f"{m}_Adiv": A_div_means[m] for m in args.modes},
        }

    print("\n=== Verdict (round 260 — PerStepInterBasinGraph) ===")
    for ds, v in summary["verdict"].items():
        mse_parts = [f"{m}={v.get(f'{m}_mse', 0):.4f}"
                     for m in args.modes]
        H_parts = [f"Hstd_{m}={v.get(f'{m}_Hstd', 0):.4f}"
                   for m in args.modes]
        Adiv_parts = [f"Adiv_{m}={v.get(f'{m}_Adiv', 0):.4f}"
                      for m in args.modes if "perstep" in m]
        print(f"[{ds:>9s}] " + " ".join(mse_parts))
        print(f"           " + " ".join(H_parts))
        if Adiv_parts:
            print(f"           " + " ".join(Adiv_parts))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()