#!/usr/bin/env python3
"""Benchmark for ChannelProjectionCfCCell (round 262).

Tests whether **multi-channel input (d_in=4)** with a learned channel
projection beats r260 (raw-input MLP) on a richer input signal.

Modes (5 total, all bench on d_in=4 multi-channel data):
  * r248_per_branch       (no graph)
  * r257_d2               (geometric only)
  * r258_graph_only       (static A)
  * r260_perstep          (raw-input MLP)
  * r262_channel_proj     (NEW, learnable channel projection)

Datasets (4 total, all d_in=4):
  * multi_ch_sin          (sin + cos + 2nd-mode + lag1)
  * multi_ch_struct       (sin + cos + harmonics + lag1)
  * multi_ch_random       (4-channel noise)
  * multi_ch_mixed        (sin first half, random second half)

Hypotheses (PRD #10-99):
  H1: r262 beats r260 on at least one dataset (projection extracts signal).
  H2: routing_context_var > x_t.var (projection amplifies signal).
  H3: r262 ties or beats r260 on all datasets (safe superset of r260).
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
from lnn.core.channel_projection_cfc import (  # noqa: E402
    ChannelProjectionCfCCell,
)

DEVICE = "cpu"


def make_dataset(name: str, T: int = 64, d_in: int = 4, seed: int = 0):
    """Multi-channel datasets with d_in=4."""
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 4 * math.pi, T)
    if name == "multi_ch_sin":
        # [t_norm, sin(t), cos(t), sin(2t)]
        t_norm = t / (4 * math.pi)
        ch1 = t_norm
        ch2 = torch.sin(t)
        ch3 = torch.cos(t)
        ch4 = torch.sin(2 * t)
        x = torch.stack([ch1, ch2, ch3, ch4], dim=-1)
        y = torch.cos(t).unsqueeze(-1)
    elif name == "multi_ch_struct":
        ch1 = torch.sin(t)
        ch2 = torch.cos(t)
        ch3 = torch.sin(2 * t) + 0.3 * torch.sin(3 * t)
        ch4 = torch.roll(ch1, shifts=1, dims=0)  # lag-1
        x = torch.stack([ch1, ch2, ch3, ch4], dim=-1)
        y = torch.roll(ch1, shifts=1, dims=0).unsqueeze(-1)
    elif name == "multi_ch_random":
        x = torch.randn(T, d_in, generator=g)
        y = torch.roll(x[:, :1], shifts=1, dims=0)
    elif name == "multi_ch_mixed":
        # First half: sin; second half: random.
        ch1 = torch.sin(t)
        ch2 = torch.cos(t)
        # Replace second half with noise.
        half = T // 2
        ch1_noise = torch.randn(T, generator=g) * 0.1
        ch1[half:] = ch1_noise[half:]
        ch3 = torch.sin(2 * t)
        ch4 = torch.roll(ch1, shifts=1, dims=0)
        x = torch.stack([ch1, ch2, ch3, ch4], dim=-1)
        y = torch.roll(ch1, shifts=1, dims=0).unsqueeze(-1)
    else:
        raise ValueError(f"unknown dataset {name}")
    return x, y


class SeqModel(nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int, mode: str):
        super().__init__()
        self.mode = mode
        self.d_h = d_h
        if mode == "r248_per_branch":
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
        elif mode == "r262_channel_proj":
            self.cell = ChannelProjectionCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                d_min=2.0, sym_lambda=0.0, sparse_lambda=0.0,
                d_ctx=8,
            )
        else:
            raise ValueError(f"unknown mode {mode}")
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor
                 ) -> tuple[torch.Tensor, list[dict]]:
        B = x.shape[1]
        outs, aux_per_step = [], []
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
        return torch.stack(outs), aux_per_step


def train_one(name: str, seed: int, mode: str, epochs: int = 100, d_h: int = 9):
    torch.manual_seed(seed)
    x, y = make_dataset(name, seed=seed)
    x = x.unsqueeze(1).to(DEVICE)  # (T, 1, d_in)
    y = y.unsqueeze(1).to(DEVICE)  # (T, 1, 1)

    model = SeqModel(d_in=x.shape[-1], d_h=d_h, d_out=y.shape[-1],
                     mode=mode).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    H_first = 0.0
    H_last = 0.0
    ctx_var = 0.0

    for ep in range(epochs):
        opt.zero_grad()
        pred, aux = model(x)
        task = ((pred - y) ** 2).mean()
        loss = task
        if mode != "r248_per_branch":
            ival = [a.get("inter_basin_loss", torch.tensor(0.0))
                    for a in aux]
            dist_loss = sum(ival) / max(len(ival), 1)
            loss = loss + dist_loss
            # Entropy reg so A_t has gradient (round 261 fix).
            h_vals = [a.get("mean_basin_H", torch.tensor(0.0))
                      for a in aux]
            mean_h = sum(h_vals) / max(len(h_vals), 1)
            loss = loss + 0.01 * mean_h
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
            if mode == "r262_channel_proj":
                ctx_var = sum(
                    a.get("routing_context_var", torch.tensor(0.0)).item()
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
        "ctx_var": ctx_var,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["multi_ch_sin", "multi_ch_struct",
                                 "multi_ch_random", "multi_ch_mixed"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--d_h", type=int, default=9)
    parser.add_argument("--modes", nargs="+",
                        default=["r248_per_branch", "r257_d2",
                                 "r258_graph_only", "r260_perstep",
                                 "r262_channel_proj"])
    parser.add_argument("--out", type=str,
                        default="analysis/channel_projection_cfc_bench.json")
    args = parser.parse_args()

    rows = []
    for mode in args.modes:
        for ds in args.datasets:
            for seed in args.seeds:
                r = train_one(ds, seed, mode,
                              epochs=args.epochs, d_h=args.d_h)
                rows.append(r)
                print(f"[{mode:>20s}] ds={ds:>15s} seed={seed} "
                      f"test_mse={r['test_mse']:.4f} "
                      f"H_last={r['H_last']:.3f} "
                      f"ctx_var={r['ctx_var']:.4f}")

    summary: dict = {"rows": rows, "verdict": {}}
    for ds in args.datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if not ds_rows:
            continue
        means = {m: sum(r["test_mse"] for r in ds_rows
                        if r["mode"] == m) / max(
            len([r for r in ds_rows if r["mode"] == m]), 1)
                 for m in args.modes}
        ctx_var_means = {m: sum(r["ctx_var"] for r in ds_rows
                                 if r["mode"] == m) / max(
            len([r for r in ds_rows if r["mode"] == m]), 1)
                         for m in args.modes}
        summary["verdict"][ds] = {
            **{f"{m}_mse": means[m] for m in args.modes},
            **{f"{m}_ctx_var": ctx_var_means[m] for m in args.modes},
        }

    print("\n=== Verdict (round 262 — ChannelProjection) ===")
    for ds, v in summary["verdict"].items():
        mse_parts = [f"{m}={v.get(f'{m}_mse', 0):.4f}"
                     for m in args.modes]
        print(f"[{ds:>15s}] " + " ".join(mse_parts))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()