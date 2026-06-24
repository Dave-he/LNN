#!/usr/bin/env python3
"""Benchmark for FrozenSampledMultiTauCfCCell (arXiv:2606.15571 L-RFM response, round 246).

Trains three configs side-by-side on three toy datasets:

  * ``baseline``        — single-τ CfC (no multi-τ)
  * ``hierarchical``    — round 245 HierarchicalMultiTauCfCCell (hand-picked)
  * ``frozen_sampled``  — round 246 FrozenSampledMultiTauCfCCell (L-RFM style)

For each cell we record:
  - ``test_mse``           — final test MSE
  - ``final_alpha_entropy``— mean mix-weight entropy (frozen_sampled only)
  - ``final_log_coverage`` — frozen τ log-coverage (frozen_sampled only)
  - ``final_learned_alpha_max`` — max learned α (frozen_sampled only)

Hypotheses (round 246 PRD):
  H1 (task safe): ``frozen_sampled`` does not regress task loss by more
      than 10% relative to ``baseline`` on any dataset.
  H2 (mix entropy healthy): final_alpha_entropy >= log(4) * 0.5 — the
      frozen τ values are *actually used* (no collapse to one branch).
  H3 (frozen ≠ hand-picked): frozen_sampled Δ% vs baseline is within
      ±20% of hierarchical Δ% vs baseline — random coverage is
      competitive with hand-picked.
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
from lnn.core.frozen_sampled_multitau_cfc import FrozenSampledMultiTauCfCCell  # noqa: E402

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
        elif mode == "hierarchical":
            self.cell = HierarchicalMultiTauCfCCell(
                d_in, d_h, tau_fast=0.1, tau_slow=5.0, learn_mix=True,
            )
        elif mode == "frozen_sampled":
            self.cell = FrozenSampledMultiTauCfCCell(
                d_in, d_h, n_branches=4, tau_min=0.05, tau_max=20.0, seed=42,
            )
        else:
            raise ValueError(f"unknown mode {mode}")
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor
                 ) -> tuple[torch.Tensor, list[dict]]:
        B = x.shape[1]
        outs, aux_per_step = [], []
        if self.mode == "frozen_sampled":
            h_list = self.cell.init_state(B, device=x.device)
            for t in range(x.shape[0]):
                h, h_list, aux = self.cell.forward_with_aux(x[t], h_list)
                outs.append(self.head(h))
                aux_per_step.append(aux)
        elif self.mode == "hierarchical":
            h_f, h_s = self.cell.init_state(B, device=x.device)
            for t in range(x.shape[0]):
                h, h_f, h_s, aux = self.cell.forward_with_aux(x[t], h_f, h_s)
                outs.append(self.head(h))
                aux_per_step.append({"alpha": aux["alpha"]})
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
        pred, _ = model(x)
        task = ((pred - y) ** 2).mean()
        task.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        pred, aux = model(x)
        test_mse = ((pred - y) ** 2).mean().item()
        alpha_ent = 0.0
        alpha_max = 0.0
        log_cov = 0.0
        if mode == "frozen_sampled":
            ents = [a.get("alpha_entropy", torch.tensor(0.0)).item()
                    for a in aux]
            alpha_ent = sum(ents) / len(ents) if ents else 0.0
            alpha = aux[-1].get("alpha", torch.tensor([0.25, 0.25, 0.25, 0.25]))
            alpha_max = float(alpha.max().item())
            log_cov = aux[-1].get("log_coverage", torch.tensor(0.0)).item()

    return {
        "dataset": name,
        "seed": seed,
        "mode": mode,
        "test_mse": test_mse,
        "final_alpha_entropy": alpha_ent,
        "final_alpha_max": alpha_max,
        "final_log_coverage": log_cov,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["toy_sin", "structured", "random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--d_h", type=int, default=9)
    parser.add_argument("--modes", nargs="+",
                        default=["baseline", "hierarchical", "frozen_sampled"])
    parser.add_argument("--out", type=str,
                        default="analysis/frozen_sampled_multitau_cfc_bench.json")
    args = parser.parse_args()

    rows = []
    for mode in args.modes:
        for ds in args.datasets:
            for seed in args.seeds:
                r = train_one(ds, seed, mode,
                              epochs=args.epochs, d_h=args.d_h)
                rows.append(r)
                print(f"[{mode:>15s}] ds={ds:>9s} seed={seed} "
                      f"test_mse={r['test_mse']:.4f} "
                      f"α_H={r['final_alpha_entropy']:.3f} "
                      f"α_max={r['final_alpha_max']:.3f} "
                      f"log_cov={r['final_log_coverage']:.2f}")

    summary: dict = {"rows": rows, "verdict": {}}
    for ds in args.datasets:
        baseline = [r for r in rows if r["dataset"] == ds
                    and r["mode"] == "baseline"]
        hier = [r for r in rows if r["dataset"] == ds
                and r["mode"] == "hierarchical"]
        frozen = [r for r in rows if r["dataset"] == ds
                  and r["mode"] == "frozen_sampled"]
        if not baseline or not frozen:
            continue
        base_mse = sum(r["test_mse"] for r in baseline) / len(baseline)
        hier_mse = sum(r["test_mse"] for r in hier) / max(len(hier), 1)
        frozen_mse = sum(r["test_mse"] for r in frozen) / len(frozen)
        delta_base = (frozen_mse - base_mse) / max(base_mse, 1e-9) * 100.0
        delta_hier = (frozen_mse - hier_mse) / max(hier_mse, 1e-9) * 100.0
        alpha_ent = sum(r["final_alpha_entropy"] for r in frozen) / len(frozen)
        alpha_max = sum(r["final_alpha_max"] for r in frozen) / len(frozen)
        log_cov = sum(r["final_log_coverage"] for r in frozen) / len(frozen)
        summary["verdict"][ds] = {
            "baseline_mse": base_mse,
            "hierarchical_mse": hier_mse,
            "frozen_sampled_mse": frozen_mse,
            "delta_pct_vs_baseline": delta_base,
            "delta_pct_vs_hierarchical": delta_hier,
            "alpha_entropy_mean": alpha_ent,
            "alpha_max_mean": alpha_max,
            "log_coverage": log_cov,
            "log_n_branches": math.log(4),
            "h1_task_safe": abs(delta_base) <= 10.0,
            "h2_mix_entropy_healthy": alpha_ent >= math.log(4) * 0.5,
            "h3_frozen_competitive": abs(delta_base - (hier_mse - base_mse)
                                          / max(base_mse, 1e-9) * 100) <= 20.0,
        }

    print("\n=== Verdict (round 246 hypotheses) ===")
    for ds, v in summary["verdict"].items():
        print(f"[{ds:>9s}] Δ%_vs_baseline={v['delta_pct_vs_baseline']:+.1f} "
              f"Δ%_vs_hier={v['delta_pct_vs_hierarchical']:+.1f} "
              f"α_H={v['alpha_entropy_mean']:.3f}/{v['log_n_branches']:.3f} "
              f"α_max={v['alpha_max_mean']:.3f} "
              f"log_cov={v['log_coverage']:.2f}  "
              f"H1={'✓' if v['h1_task_safe'] else '✗'} "
              f"H2={'✓' if v['h2_mix_entropy_healthy'] else '✗'} "
              f"H3={'✓' if v['h3_frozen_competitive'] else '✗'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()