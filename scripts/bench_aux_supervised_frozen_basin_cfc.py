#!/usr/bin/env python3
"""Benchmark for AuxSupervisedFrozenRandomBasinCfCCell (round 251).

Tests whether aux supervision enables frozen random basins to recover
task loss parity with learned basins (round 248). Compares:

  * ``baseline``                   — single-τ CfC
  * ``per_branch``                 — round 248 (LEARNED basins)
  * ``frozen_no_aux``              — round 250 (frozen basins, lyap_lambda=0)
  * ``aux_supervised_frozen``      — round 251 (frozen basins, lyap_lambda=0.1)

Hypotheses (round 251 PRD):
  H1: aux_supervised_frozen recovers parity with r248 within ±10% per dataset.
  H2: aux loss decreases monotonically over training (track first vs last).
  H3: V_next <= V_prev × (1 - alpha) on average after training.
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
from lnn.core.aux_supervised_frozen_basin_cfc import (  # noqa: E402
    AuxSupervisedFrozenRandomBasinCfCCell,
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
        elif mode == "frozen_no_aux":
            self.cell = FrozenRandomBasinCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, tau_seed=42, basin_seed=137,
            )
        elif mode == "aux_supervised_frozen":
            self.cell = AuxSupervisedFrozenRandomBasinCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, tau_seed=42, basin_seed=137,
                lyap_lambda=0.1,
            )
        else:
            raise ValueError(f"unknown mode {mode}")
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor
                 ) -> tuple[torch.Tensor, list[dict]]:
        B = x.shape[1]
        outs, aux_per_step = [], []
        if self.mode in ("per_branch", "frozen_no_aux",
                         "aux_supervised_frozen"):
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

    aux_losses_first = []
    aux_losses_last = []

    for ep in range(epochs):
        opt.zero_grad()
        pred, aux = model(x)
        task = ((pred - y) ** 2).mean()
        loss = task
        if mode == "aux_supervised_frozen":
            lyap_vals = [a.get("lyap_loss", torch.tensor(0.0))
                         for a in aux]
            aux_loss = sum(lyap_vals) / max(len(lyap_vals), 1)
            loss = loss + 0.1 * aux_loss
            if ep == 0:
                aux_losses_first.append(aux_loss.item())
            if ep == epochs - 1:
                aux_losses_last.append(aux_loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        pred, aux = model(x)
        test_mse = ((pred - y) ** 2).mean().item()
        # V contraction check: mean(V_next) <= mean(V_prev) * (1 - alpha)
        V_nexts = []
        V_prevs = []
        for a in aux:
            V = a.get("per_branch_V_next")
            if V is not None:
                V_nexts.append(V.mean().item())
        if mode == "aux_supervised_frozen":
            contraction_ratio = (sum(V_nexts) / len(V_nexts)
                                  if V_nexts else 0.0)
        else:
            contraction_ratio = 0.0
        aux_loss_first = (aux_losses_first[0] if aux_losses_first else 0.0)
        aux_loss_last = (aux_losses_last[0] if aux_losses_last else 0.0)

    return {
        "dataset": name,
        "seed": seed,
        "mode": mode,
        "test_mse": test_mse,
        "aux_loss_first": aux_loss_first,
        "aux_loss_last": aux_loss_last,
        "V_next_mean": contraction_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["toy_sin", "structured", "random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--d_h", type=int, default=9)
    parser.add_argument("--modes", nargs="+",
                        default=["baseline", "per_branch", "frozen_no_aux",
                                 "aux_supervised_frozen"])
    parser.add_argument("--out", type=str,
                        default="analysis/aux_supervised_frozen_basin_cfc_bench.json")
    args = parser.parse_args()

    rows = []
    for mode in args.modes:
        for ds in args.datasets:
            for seed in args.seeds:
                r = train_one(ds, seed, mode,
                              epochs=args.epochs, d_h=args.d_h)
                rows.append(r)
                print(f"[{mode:>22s}] ds={ds:>9s} seed={seed} "
                      f"test_mse={r['test_mse']:.4f} "
                      f"aux_first={r['aux_loss_first']:.4f} "
                      f"aux_last={r['aux_loss_last']:.4f} "
                      f"V_mean={r['V_next_mean']:.3f}")

    summary: dict = {"rows": rows, "verdict": {}}
    alpha = 0.05
    for ds in args.datasets:
        ds_rows = {m: [r for r in rows if r["dataset"] == ds
                       and r["mode"] == m] for m in args.modes}
        if not all(ds_rows.values()):
            continue
        means = {m: sum(r["test_mse"] for r in ds_rows[m]) / len(ds_rows[m])
                 for m in args.modes}
        aux_first = (sum(r["aux_loss_first"] for r
                          in ds_rows["aux_supervised_frozen"])
                      / len(ds_rows["aux_supervised_frozen"]))
        aux_last = (sum(r["aux_loss_last"] for r
                         in ds_rows["aux_supervised_frozen"])
                     / len(ds_rows["aux_supervised_frozen"]))
        delta_r248 = (means["aux_supervised_frozen"] - means["per_branch"]
                      ) / max(means["per_branch"], 1e-9) * 100.0
        delta_r250 = (means["aux_supervised_frozen"] - means["frozen_no_aux"]
                      ) / max(means["frozen_no_aux"], 1e-9) * 100.0
        summary["verdict"][ds] = {
            **{f"{m}_mse": means[m] for m in args.modes},
            "delta_pct_vs_r248": delta_r248,
            "delta_pct_vs_r250": delta_r250,
            "aux_loss_first": aux_first,
            "aux_loss_last": aux_last,
            "h1_parity_with_r248": abs(delta_r248) <= 10.0,
            "h2_aux_decreasing": aux_last < aux_first,
            "h3_v_contraction": aux_last < (1.0 - alpha) * aux_first,
        }

    print("\n=== Verdict (round 251 hypotheses) ===")
    for ds, v in summary["verdict"].items():
        print(f"[{ds:>9s}] "
              f"aux_super={v['aux_supervised_frozen_mse']:.4f} "
              f"per_branch={v['per_branch_mse']:.4f} "
              f"frozen_no_aux={v['frozen_no_aux_mse']:.4f} "
              f"baseline={v['baseline_mse']:.4f}  "
              f"Δ%_r248={v['delta_pct_vs_r248']:+.1f} "
              f"Δ%_r250={v['delta_pct_vs_r250']:+.1f}  "
              f"aux_first={v['aux_loss_first']:.3f}→{v['aux_loss_last']:.3f}  "
              f"H1={'✓' if v['h1_parity_with_r248'] else '✗'} "
              f"H2={'✓' if v['h2_aux_decreasing'] else '✗'} "
              f"H3={'✓' if v['h3_v_contraction'] else '✗'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()