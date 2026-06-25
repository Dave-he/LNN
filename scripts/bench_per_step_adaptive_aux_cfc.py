#!/usr/bin/env python3
"""Benchmark for PerStepAdaptiveAuxMultiBasinLyapunovCfCCell (round 254).

Tests the temporal axis of the per-branch basin mechanism: aux weight
is per-STEP (mean H across branches), complementary to r253 per-BRANCH.

Modes:
  * baseline
  * r248_per_branch
  * r249_input_geom
  * r252_lyap_aux       (constant aux)
  * r253_adaptive_aux   (per-branch H gating)
  * r254_per_step_aux   (per-step mean-H gating) ⭐

Hypotheses:
  H1: r254 matches r253 toy_sin/random within ±5%.
  H2: r254 improves r253 on structured (complementary gating helps).
  H3: mean λ_t low on toy_sin (stable), high on random (transitions).
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
        else:
            raise ValueError(f"unknown mode {mode}")
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor
                 ) -> tuple[torch.Tensor, list[dict]]:
        B = x.shape[1]
        outs, aux_per_step = [], []
        if self.mode in ("r248_per_branch", "r249_input_geom",
                         "r252_lyap_aux", "r253_adaptive_aux",
                         "r254_per_step_aux"):
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
    lambda_first = 0.0
    lambda_last = 0.0

    for ep in range(epochs):
        opt.zero_grad()
        pred, aux = model(x)
        task = ((pred - y) ** 2).mean()
        loss = task
        if mode in ("r252_lyap_aux", "r253_adaptive_aux", "r254_per_step_aux"):
            lyap_vals = [a.get("lyap_loss", torch.tensor(0.0))
                         for a in aux]
            aux_loss = sum(lyap_vals) / max(len(lyap_vals), 1)
            loss = loss + aux_loss
            if ep == 0:
                aux_first = aux_loss.item()
                if mode == "r254_per_step_aux":
                    lambda_first = sum(
                        a.get("lambda_step", torch.tensor(0.0)).item()
                        for a in aux
                    ) / len(aux)
            if ep == epochs - 1:
                aux_last = aux_loss.item()
                if mode == "r254_per_step_aux":
                    lambda_last = sum(
                        a.get("lambda_step", torch.tensor(0.0)).item()
                        for a in aux
                    ) / len(aux)
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
        "aux_first": aux_first,
        "aux_last": aux_last,
        "lambda_first": lambda_first,
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
                                 "r254_per_step_aux"])
    parser.add_argument("--out", type=str,
                        default="analysis/per_step_adaptive_aux_cfc_bench.json")
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
                      f"aux_first={r['aux_first']:.4f} "
                      f"aux_last={r['aux_last']:.4f} "
                      f"λ_first={r['lambda_first']:.3f} "
                      f"λ_last={r['lambda_last']:.3f}")

    summary: dict = {"rows": rows, "verdict": {}}
    for ds in args.datasets:
        ds_rows = {m: [r for r in rows if r["dataset"] == ds
                       and r["mode"] == m] for m in args.modes}
        if not all(ds_rows.values()):
            continue
        means = {m: sum(r["test_mse"] for r in ds_rows[m]) / len(ds_rows[m])
                 for m in args.modes}
        # r254 lambda_last (mean over seeds)
        lambda_r254 = (sum(r["lambda_last"] for r
                            in ds_rows["r254_per_step_aux"])
                       / max(len(ds_rows["r254_per_step_aux"]), 1))
        summary["verdict"][ds] = {
            **{f"{m}_mse": means[m] for m in args.modes},
            "lambda_r254_last": lambda_r254,
        }

    print("\n=== Verdict (round 254) ===")
    for ds, v in summary["verdict"].items():
        r254 = v["r254_per_step_aux_mse"]
        r253 = v["r253_adaptive_aux_mse"]
        r252 = v["r252_lyap_aux_mse"]
        r249 = v["r249_input_geom_mse"]
        r248 = v["r248_per_branch_mse"]
        bl = v["baseline_mse"]
        lam = v["lambda_r254_last"]
        print(f"[{ds:>9s}] "
              f"r254={r254:.4f} r253={r253:.4f} r252={r252:.4f} "
              f"r249={r249:.4f} r248={r248:.4f} baseline={bl:.4f}  "
              f"λ_last={lam:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
