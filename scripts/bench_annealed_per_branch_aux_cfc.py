#!/usr/bin/env python3
"""Benchmark for AnnealedPerBranchAux (round 256).

Pivots the 10-round arc to a NEW axis: training-epoch annealing of λ.
Tests hypothesis: aux is most useful EARLY (initial regularizer) and
should be REDUCED LATE (task freedom).

Modes:
  * baseline
  * r248_per_branch
  * r249_input_geom
  * r252_lyap_aux           (constant λ = 0.1)
  * r253_adaptive_aux       (per-branch H-gated)
  * r254_per_step_aux       (per-step H-gated)
  * r255_combined_product   (2D product)
  * r256_anneal_linear      ⭐ (NEW, linear anneal λ_max → 0)
  * r256_anneal_cosine      ⭐ (NEW, cosine anneal λ_max → 0)
  * r256_anneal_exp         ⭐ (NEW, exponential anneal λ_max → 0)

Hypotheses:
  H1: anneal improves early convergence (lower loss at ep=25).
  H2: anneal matches r248 final at ep=100 (no regression).
  H3: anneal beats constant r252 on toy_sin/random where r252 hurts.
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
from lnn.core.input_geometry_gated_per_branch_cfc import (  # noqa: E402
    InputGeometryGatedPerBranchCfCCell,
)
from lnn.core.lyap_aux_per_branch_multibasin_cfc import (  # noqa: E402
    LyapAuxPerBranchMultiBasinLyapunovCfCCell,
)
from lnn.core.adaptive_aux_per_branch_cfc import (  # noqa: E402
    AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell,
)
from lnn.core.per_step_adaptive_aux_cfc import (  # noqa: E402
    PerStepAdaptiveAuxMultiBasinLyapunovCfCCell,
)
from lnn.core.combined_per_branch_per_step_aux_cfc import (  # noqa: E402
    CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell,
)
from lnn.core.annealed_per_branch_aux_cfc import (  # noqa: E402
    AnnealedPerBranchMultiBasinLyapunovCfCCell,
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
        elif mode == "r249_input_geom":
            self.cell = InputGeometryGatedPerBranchCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
            )
        elif mode == "r252_lyap_aux":
            self.cell = LyapAuxPerBranchMultiBasinLyapunovCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                lyap_lambda=0.1,
            )
        elif mode == "r253_adaptive_aux":
            self.cell = AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                lyap_lambda_max=0.1, adaptive_aux=True,
            )
        elif mode == "r254_per_step_aux":
            self.cell = PerStepAdaptiveAuxMultiBasinLyapunovCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                lyap_lambda_max=0.1, per_step_aux=True,
            )
        elif mode == "r255_combined_product":
            self.cell = CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                lyap_lambda_max=0.1, combination="product",
            )
        elif mode == "r256_anneal_linear":
            self.cell = AnnealedPerBranchMultiBasinLyapunovCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                lyap_lambda_max=0.1, anneal_epochs=50,
                anneal_schedule="linear",
            )
        elif mode == "r256_anneal_cosine":
            self.cell = AnnealedPerBranchMultiBasinLyapunovCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                lyap_lambda_max=0.1, anneal_epochs=50,
                anneal_schedule="cosine",
            )
        elif mode == "r256_anneal_exp":
            self.cell = AnnealedPerBranchMultiBasinLyapunovCfCCell(
                d_in, d_h, n_branches=4, n_basin=3,
                tau_min=0.05, tau_max=20.0, seed=42,
                lyap_lambda_max=0.1, anneal_epochs=50,
                anneal_schedule="exp",
            )
        else:
            raise ValueError(f"unknown mode {mode}")
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor
                 ) -> tuple[torch.Tensor, list[dict]]:
        B = x.shape[1]
        outs, aux_per_step = [], []
        if self.mode in ("r248_per_branch", "r249_input_geom",
                         "r252_lyap_aux", "r253_adaptive_aux",
                         "r254_per_step_aux", "r255_combined_product",
                         "r256_anneal_linear", "r256_anneal_cosine",
                         "r256_anneal_exp"):
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

    aux_first = 0.0
    aux_last = 0.0
    aux_mid = 0.0
    lambda_first = 0.0
    lambda_last = 0.0
    lambda_mid = 0.0
    task_first = 0.0
    task_mid25 = 0.0
    task_mid50 = 0.0
    task_last = 0.0

    aux_modes = ("r252_lyap_aux", "r253_adaptive_aux", "r254_per_step_aux",
                 "r255_combined_product",
                 "r256_anneal_linear", "r256_anneal_cosine", "r256_anneal_exp")

    for ep in range(epochs):
        # Annealing cells need epoch updates.
        if mode.startswith("r256_"):
            model.cell.set_epoch(ep)

        opt.zero_grad()
        pred, aux = model(x)
        task = ((pred - y) ** 2).mean()
        loss = task
        if mode in aux_modes:
            lyap_vals = [a.get("lyap_loss", torch.tensor(0.0))
                         for a in aux]
            aux_loss = sum(lyap_vals) / max(len(lyap_vals), 1)
            loss = loss + aux_loss
            if ep == 0:
                aux_first = aux_loss.item()
                task_first = task.item()
                if mode.startswith("r256_"):
                    lambda_first = aux[0].get(
                        "current_lambda", torch.tensor(0.0)
                    ).item()
            if ep == 25:
                aux_mid = aux_loss.item()
                task_mid25 = task.item()
                if mode.startswith("r256_"):
                    lambda_mid = aux[0].get(
                        "current_lambda", torch.tensor(0.0)
                    ).item()
            if ep == 50:
                task_mid50 = task.item()
            if ep == epochs - 1:
                aux_last = aux_loss.item()
                task_last = task.item()
                if mode.startswith("r256_"):
                    lambda_last = aux[0].get(
                        "current_lambda", torch.tensor(0.0)
                    ).item()
        else:
            if ep == 0:
                task_first = task.item()
            if ep == 25:
                task_mid25 = task.item()
            if ep == 50:
                task_mid50 = task.item()
            if ep == epochs - 1:
                task_last = task.item()
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
        "task_first": task_first,
        "task_mid25": task_mid25,
        "task_mid50": task_mid50,
        "task_last": task_last,
        "aux_first": aux_first,
        "aux_mid": aux_mid,
        "aux_last": aux_last,
        "lambda_first": lambda_first,
        "lambda_mid": lambda_mid,
        "lambda_last": lambda_last,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["toy_sin", "structured", "random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--d_h", type=int, default=9)
    parser.add_argument("--modes", nargs="+",
                        default=["baseline", "r248_per_branch", "r249_input_geom",
                                 "r252_lyap_aux", "r253_adaptive_aux",
                                 "r254_per_step_aux", "r255_combined_product",
                                 "r256_anneal_linear", "r256_anneal_cosine",
                                 "r256_anneal_exp"])
    parser.add_argument("--out", type=str,
                        default="analysis/annealed_per_branch_aux_cfc_bench.json")
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
                      f"task_ep25={r['task_mid25']:.4f} "
                      f"task_ep100={r['task_last']:.4f} "
                      f"λ_first={r['lambda_first']:.3f} "
                      f"λ_mid={r['lambda_mid']:.3f} "
                      f"λ_last={r['lambda_last']:.3f}")

    summary: dict = {"rows": rows, "verdict": {}}
    for ds in args.datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if not ds_rows:
            continue
        means = {m: sum(r["test_mse"] for r in ds_rows
                        if r["mode"] == m) / max(
            len([r for r in ds_rows if r["mode"] == m]), 1)
                 for m in args.modes}
        # early-convergence (ep=25) means
        means_e25 = {m: sum(r["task_mid25"] for r in ds_rows
                            if r["mode"] == m) / max(
            len([r for r in ds_rows if r["mode"] == m]), 1)
                     for m in args.modes}
        # lambda r256 linear
        lambda_r256 = (sum(r["lambda_last"] for r
                            in ds_rows if r["mode"] == "r256_anneal_linear")
                       / max(len([r for r in ds_rows
                                  if r["mode"] == "r256_anneal_linear"]), 1))
        lambda_r256_mid = (sum(r["lambda_mid"] for r
                                in ds_rows if r["mode"] == "r256_anneal_linear")
                            / max(len([r for r in ds_rows
                                       if r["mode"] == "r256_anneal_linear"]),
                                  1))
        summary["verdict"][ds] = {
            **{f"{m}_mse": means[m] for m in args.modes},
            **{f"{m}_e25": means_e25[m] for m in args.modes},
            "lambda_r256_linear_last": lambda_r256,
            "lambda_r256_linear_mid": lambda_r256_mid,
        }

    print("\n=== Verdict (round 256 — AnnealedPerBranchAux) ===")
    for ds, v in summary["verdict"].items():
        print(f"[{ds:>9s}] "
              f"r256_lin={v['r256_anneal_linear_mse']:.4f} "
              f"r256_cos={v['r256_anneal_cosine_mse']:.4f} "
              f"r256_exp={v['r256_anneal_exp_mse']:.4f} "
              f"r255={v['r255_combined_product_mse']:.4f} "
              f"r252={v['r252_lyap_aux_mse']:.4f} "
              f"r249={v['r249_input_geom_mse']:.4f} "
              f"r248={v['r248_per_branch_mse']:.4f} "
              f"baseline={v['baseline_mse']:.4f}  "
              f"λ_lin_mid={v['lambda_r256_linear_mid']:.3f} "
              f"λ_lin_last={v['lambda_r256_linear_last']:.3f}")
        print(f"           "
              f"e25 r256_lin={v['r256_anneal_linear_e25']:.4f} "
              f"e25 r252={v['r252_lyap_aux_e25']:.4f} "
              f"e25 r248={v['r248_per_branch_e25']:.4f} "
              f"e25 baseline={v['baseline_e25']:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
