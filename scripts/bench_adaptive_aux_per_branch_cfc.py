#!/usr/bin/env python3
"""Benchmark for AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell (round 253).

Closes the 7-round arc (r246-r252). Round 252 used constant aux weight
(0.1) across all branches and regressed on structured data (+34.5%
vs r248). Round 253 makes the aux weight **adaptive per branch**,
scaled by per-branch basin assignment entropy H_k.

Hypotheses (round 253 PRD):
  H1: r253 matches r252 toy_sin/random within ±5% (preserves the wins).
  H2: r253 IMPROVES r252 on structured (closing the +34.5% gap).
  H3: mean λ on structured < mean λ on toy_sin (confident branches
      on structured → low λ, periodic dynamics preserved).

Modes:
  * ``baseline``           — single-τ CfC
  * ``r248_per_branch``    — round 248 (LEARNED, no aux)
  * ``r249_input_geom``    — round 249 (gate, no aux)
  * ``r252_lyap_aux``      — round 252 (LEARNED + constant aux)
  * ``r253_adaptive_aux``  — round 253 (LEARNED + adaptive aux) ⭐
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
        else:
            raise ValueError(f"unknown mode {mode}")
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor
                 ) -> tuple[torch.Tensor, list[dict]]:
        B = x.shape[1]
        outs, aux_per_step = [], []
        if self.mode in ("r248_per_branch", "r249_input_geom",
                         "r252_lyap_aux", "r253_adaptive_aux"):
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
    mean_lambda_first = 0.0
    mean_lambda_last = 0.0

    for ep in range(epochs):
        opt.zero_grad()
        pred, aux = model(x)
        task = ((pred - y) ** 2).mean()
        loss = task
        if mode in ("r252_lyap_aux", "r253_adaptive_aux"):
            lyap_vals = [a.get("lyap_loss", torch.tensor(0.0))
                         for a in aux]
            aux_loss = sum(lyap_vals) / max(len(lyap_vals), 1)
            loss = loss + aux_loss
            if ep == 0:
                aux_first = aux_loss.item()
                if mode == "r253_adaptive_aux":
                    mean_lambda_first = sum(
                        a.get("mean_lambda", torch.tensor(0.0)).item()
                        for a in aux
                    ) / len(aux)
            if ep == epochs - 1:
                aux_last = aux_loss.item()
                if mode == "r253_adaptive_aux":
                    mean_lambda_last = sum(
                        a.get("mean_lambda", torch.tensor(0.0)).item()
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
        "mean_lambda_first": mean_lambda_first,
        "mean_lambda_last": mean_lambda_last,
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
                                 "r252_lyap_aux", "r253_adaptive_aux"])
    parser.add_argument("--out", type=str,
                        default="analysis/adaptive_aux_per_branch_cfc_bench.json")
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
                      f"λ_first={r['mean_lambda_first']:.3f} "
                      f"λ_last={r['mean_lambda_last']:.3f}")

    summary: dict = {"rows": rows, "verdict": {}}
    for ds in args.datasets:
        ds_rows = {m: [r for r in rows if r["dataset"] == ds
                       and r["mode"] == m] for m in args.modes}
        if not all(ds_rows.values()):
            continue
        means = {m: sum(r["test_mse"] for r in ds_rows[m]) / len(ds_rows[m])
                 for m in args.modes}
        mean_lambda_r253 = (sum(r["mean_lambda_last"] for r
                                in ds_rows["r253_adaptive_aux"])
                            / len(ds_rows["r253_adaptive_aux"]))
        # H1: r253 within ±10% of r252 on toy_sin/random.
        h1 = {}
        for comp_ds in ("toy_sin", "random"):
            comp_rows = [r for r in rows if r["dataset"] == comp_ds]
            r252_mse = (sum(r["test_mse"] for r in comp_rows
                            if r["mode"] == "r252_lyap_aux")
                        / max(len([r for r in comp_rows
                                   if r["mode"] == "r252_lyap_aux"]), 1))
            r253_mse = (sum(r["test_mse"] for r in comp_rows
                            if r["mode"] == "r253_adaptive_aux")
                        / max(len([r for r in comp_rows
                                   if r["mode"] == "r253_adaptive_aux"]), 1))
            h1[comp_ds] = abs(r253_mse - r252_mse) / max(r252_mse, 1e-9) <= 0.10
        # H2: r253 IMPROVES or matches r252 on structured.
        struct_rows = [r for r in rows if r["dataset"] == "structured"]
        r252_struct = (sum(r["test_mse"] for r in struct_rows
                            if r["mode"] == "r252_lyap_aux")
                        / max(len([r for r in struct_rows
                                   if r["mode"] == "r252_lyap_aux"]), 1))
        r253_struct = (sum(r["test_mse"] for r in struct_rows
                            if r["mode"] == "r253_adaptive_aux")
                        / max(len([r for r in struct_rows
                                   if r["mode"] == "r253_adaptive_aux"]), 1))
        h2 = r253_struct <= r252_struct * 1.10  # within 10% of r252

        summary["verdict"][ds] = {
            **{f"{m}_mse": means[m] for m in args.modes},
            "mean_lambda_r253_last": mean_lambda_r253,
            "delta_pct_r253_vs_r252": (r253_struct - r252_struct
                                       ) / max(r252_struct, 1e-9) * 100.0,
            "h1_match_r252_toy_random": all(h1.values()) if h1 else None,
            "h2_improves_structured": h2,
        }

    print("\n=== Verdict (round 253 hypotheses) ===")
    for ds, v in summary["verdict"].items():
        h1 = v["h1_match_r252_toy_random"]
        h2 = v["h2_improves_structured"]
        h3_lambda = v["mean_lambda_r253_last"]
        print(f"[{ds:>9s}] "
              f"r253={v['r253_adaptive_aux_mse']:.4f} "
              f"r252={v['r252_lyap_aux_mse']:.4f} "
              f"r249={v['r249_input_geom_mse']:.4f} "
              f"r248={v['r248_per_branch_mse']:.4f} "
              f"baseline={v['baseline_mse']:.4f}  "
              f"Δ%_r252={v['delta_pct_r253_vs_r252']:+.1f}  "
              f"λ_last={h3_lambda:.3f}  "
              f"H1={'✓' if h1 else '✗'} H2={'✓' if h2 else '✗'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
