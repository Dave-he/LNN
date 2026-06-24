#!/usr/bin/env python3
"""Benchmark for HierarchicalMultiTauCfCCell (arXiv:2606.19579 FlowFake response, round 245).

Trains three configs side-by-side on three toy datasets:

  * ``baseline``        — single-τ CfC (no multi-τ)
  * ``multi_tau``       — round 76 CfC(n_tau=3, geometric)
  * ``hierarchical``    — round 245 HierarchicalMultiTauCfCCell (2-band)

For each cell we record:
  - ``test_mse``       — final test MSE
  - ``final_alpha``    — learned mixing weight (hierarchical only)
  - ``final_fast_std`` — fast-band hidden-state std (variability)
  - ``final_slow_std`` — slow-band hidden-state std (variability)

Hypotheses (round 245 PRD):
  H1 (task safe): ``hierarchical`` does not regress task loss by more
      than 10% relative to ``baseline`` on any dataset.
  H2 (mix learnt): on smooth data the learned α moves toward fast
      (responds to local detail); on noisy data α moves toward slow.
  H3 (multi-band effect): the two bands carry *operationally different*
      information — final_fast_std ≠ final_slow_std.
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
from lnn.core.hierarchical_multitau_cfc import HierarchicalMultiTauCfCCell  # noqa: E402

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
        elif mode == "multi_tau":
            self.cell = CfCCell(d_in, d_h, n_tau=3, tau_scales=(0.1, 1.0, 10.0))
        elif mode == "hierarchical":
            self.cell = HierarchicalMultiTauCfCCell(
                d_in, d_h, tau_fast=0.1, tau_slow=5.0, learn_mix=True,
            )
        else:
            raise ValueError(f"unknown mode {mode}")
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor
                 ) -> tuple[torch.Tensor, list[dict]]:
        B = x.shape[1]
        outs, aux_per_step = [], []
        if self.mode == "hierarchical":
            h_f, h_s = self.cell.init_state(B, device=x.device)
            for t in range(x.shape[0]):
                h, h_f, h_s, aux = self.cell.forward_with_aux(x[t], h_f, h_s)
                outs.append(self.head(h))
                aux_per_step.append({
                    "alpha": aux["alpha"],
                    "fast_std": h_f.std().item(),
                    "slow_std": h_s.std().item(),
                })
        else:
            h = torch.zeros(B, self.cell.hidden_size, device=x.device)
            for t in range(x.shape[0]):
                h = self.cell(x[t], h)
                outs.append(self.head(h))
                aux_per_step.append({"alpha": torch.tensor(0.5)})
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
        pred, _ = model(x)
        task = ((pred - y) ** 2).mean()
        task.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        pred, aux = model(x)
        test_mse = ((pred - y) ** 2).mean().item()
        if mode == "hierarchical":
            alpha_final = aux[-1]["alpha"].item()
            fast_std = sum(a["fast_std"] for a in aux) / len(aux)
            slow_std = sum(a["slow_std"] for a in aux) / len(aux)
        else:
            alpha_final = 0.5
            fast_std = 0.0
            slow_std = 0.0

    return {
        "dataset": name,
        "seed": seed,
        "mode": mode,
        "test_mse": test_mse,
        "final_alpha": alpha_final,
        "final_fast_std": fast_std,
        "final_slow_std": slow_std,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["toy_sin", "structured", "random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--d_h", type=int, default=9)
    parser.add_argument("--modes", nargs="+",
                        default=["baseline", "multi_tau", "hierarchical"])
    parser.add_argument("--out", type=str,
                        default="analysis/hierarchical_multitau_cfc_bench.json")
    args = parser.parse_args()

    rows = []
    for mode in args.modes:
        for ds in args.datasets:
            for seed in args.seeds:
                r = train_one(ds, seed, mode,
                              epochs=args.epochs, d_h=args.d_h)
                rows.append(r)
                print(f"[{mode:>13s}] ds={ds:>9s} seed={seed} "
                      f"test_mse={r['test_mse']:.4f} "
                      f"alpha={r['final_alpha']:.3f} "
                      f"fast={r['final_fast_std']:.3f} "
                      f"slow={r['final_slow_std']:.3f}")

    summary: dict = {"rows": rows, "verdict": {}}
    for ds in args.datasets:
        baseline = [r for r in rows if r["dataset"] == ds
                    and r["mode"] == "baseline"]
        if not baseline:
            continue
        base_mse = sum(r["test_mse"] for r in baseline) / len(baseline)
        mt = [r for r in rows if r["dataset"] == ds
              and r["mode"] == "multi_tau"]
        hier = [r for r in rows if r["dataset"] == ds
                and r["mode"] == "hierarchical"]
        if not hier:
            continue
        hier_mse = sum(r["test_mse"] for r in hier) / len(hier)
        mt_mse = sum(r["test_mse"] for r in mt) / max(len(mt), 1)
        delta = (hier_mse - base_mse) / max(base_mse, 1e-9) * 100.0
        alpha_mean = sum(r["final_alpha"] for r in hier) / len(hier)
        fast_std_mean = sum(r["final_fast_std"] for r in hier) / len(hier)
        slow_std_mean = sum(r["final_slow_std"] for r in hier) / len(hier)
        summary["verdict"][ds] = {
            "baseline_mse": base_mse,
            "multi_tau_mse": mt_mse,
            "hierarchical_mse": hier_mse,
            "delta_pct_vs_baseline": delta,
            "delta_pct_vs_multi_tau": (hier_mse - mt_mse) / max(mt_mse, 1e-9) * 100,
            "alpha_learned": alpha_mean,
            "fast_std": fast_std_mean,
            "slow_std": slow_std_mean,
            "h1_task_safe": abs(delta) <= 10.0,
            "h2_alpha_learned": abs(alpha_mean - 0.5) > 0.05,
            "h3_bands_distinct": abs(fast_std_mean - slow_std_mean) > 1e-3,
        }

    print("\n=== Verdict (round 245 hypotheses) ===")
    for ds, v in summary["verdict"].items():
        print(f"[{ds:>9s}] Δ%_vs_baseline={v['delta_pct_vs_baseline']:+.1f} "
              f"Δ%_vs_multi_tau={v['delta_pct_vs_multi_tau']:+.1f} "
              f"α={v['alpha_learned']:.3f} "
              f"fast={v['fast_std']:.3f} slow={v['slow_std']:.3f}  "
              f"H1={'✓' if v['h1_task_safe'] else '✗'} "
              f"H2={'✓' if v['h2_alpha_learned'] else '✗'} "
              f"H3={'✓' if v['h3_bands_distinct'] else '✗'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()