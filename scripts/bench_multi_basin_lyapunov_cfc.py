#!/usr/bin/env python3
"""Benchmark for MultiBasinLyapunovStableCfCCell (arXiv:2606.18315 response, round 244).

Trains three configs side-by-side on three toy datasets:

  * ``baseline``   — single-τ CfC (no Lyapunov)
  * ``lyap``       — round 240 LyapunovStableCfCCell (single basin at origin)
  * ``multi_basin``— round 244 MultiBasinLyapunovStableCfCCell (K=3 basins)

For each cell we record:
  - ``test_mse``            — final test MSE
  - ``final_V_mean``        — average Lyapunov value over T
  - ``final_basin_entropy`` — mean per-sample basin-assignment entropy
  - ``final_basin_top1_frac`` — fraction of samples assigned to basin 0

Hypotheses (round 244 PRD):
  H1 (task safe): ``multi_basin`` does not regress task loss by more than
      10% relative to ``baseline`` on any dataset.
  H2 (basin utilisation): ``final_basin_entropy >= log(n_basin) * 0.5``
      — multiple basins are actually being used (no collapse to one).
  H3 (basin contraction): ``final_V_mean_multi_basin <= final_V_mean_lyap``
      — multi-basin Lyapunov is at least as informative as single-basin
      (basins are easier to reach than the origin).
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
from lnn.core.lyapunov_stable_cfc import (  # noqa: E402
    LyapunovStableCfCCell,
    lyapunov_value,
)
from lnn.core.multi_basin_lyapunov_cfc import (  # noqa: E402
    MultiBasinLyapunovStableCfCCell,
    basin_assignment_entropy,
    multi_basin_lyapunov_value,
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
            self._aux = False
        elif mode == "lyap":
            self.cell = LyapunovStableCfCCell(d_in, d_h, alpha=0.05)
            self._aux = True
        elif mode == "multi_basin":
            self.cell = MultiBasinLyapunovStableCfCCell(
                d_in, d_h, n_basin=3, alpha=0.05, beta_v=2.0,
            )
            self._aux = True
        else:
            raise ValueError(f"unknown mode {mode}")
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor
                 ) -> tuple[torch.Tensor, list[dict]]:
        B = x.shape[1]
        h = torch.zeros(B, self.cell.hidden_size, device=x.device)
        outs, aux_per_step = [], []
        for t in range(x.shape[0]):
            if self._aux:
                h, aux = self.cell.forward_with_aux(
                    x[t], h, lyap_lambda=0.1,
                )
                aux_per_step.append(aux)
            else:
                h = self.cell(x[t], h)
                aux_per_step.append({
                    "V_next": torch.tensor(0.0),
                    "basin_entropy": torch.tensor(0.0),
                    "basin_assign": None,
                })
            outs.append(self.head(h))
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
        pred, aux = model(x)
        task = ((pred - y) ** 2).mean()
        loss = task
        if mode == "lyap":
            for a in aux:
                if "lyap_loss_total" in a:
                    loss = loss + a["lyap_loss_total"]
                if "pd_loss_total" in a:
                    loss = loss + a["pd_loss_total"]
        elif mode == "multi_basin":
            for a in aux:
                if "lyap_loss_total" in a:
                    loss = loss + a["lyap_loss_total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        pred, aux = model(x)
        test_mse = ((pred - y) ** 2).mean().item()
        V_seq = torch.stack([a.get("V_next", torch.tensor(0.0)) for a in aux])
        V_mean = V_seq.mean().item() if V_seq.numel() > 0 else 0.0
        ent_seq = torch.stack(
            [a.get("basin_entropy", torch.tensor(0.0)) for a in aux]
        )
        basin_ent = ent_seq.mean().item() if ent_seq.numel() > 0 else 0.0
        # Compute top-1 fraction if available (multi_basin).
        top1 = 0.0
        if mode == "multi_basin":
            top1_per_step = []
            for a in aux:
                assign = a.get("basin_assign", None)
                if assign is not None:
                    top1_per_step.append(assign.argmax(dim=-1).eq(0).float().mean().item())
            top1 = sum(top1_per_step) / max(len(top1_per_step), 1)

    return {
        "dataset": name,
        "seed": seed,
        "mode": mode,
        "test_mse": test_mse,
        "final_V_mean": V_mean,
        "final_basin_entropy": basin_ent,
        "final_basin_top1_frac": top1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["toy_sin", "structured", "random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--d_h", type=int, default=9)
    parser.add_argument("--modes", nargs="+",
                        default=["baseline", "lyap", "multi_basin"])
    parser.add_argument("--out", type=str,
                        default="analysis/multi_basin_lyapunov_cfc_bench.json")
    args = parser.parse_args()

    rows = []
    for mode in args.modes:
        for ds in args.datasets:
            for seed in args.seeds:
                r = train_one(ds, seed, mode,
                              epochs=args.epochs, d_h=args.d_h)
                rows.append(r)
                print(f"[{mode:>11s}] ds={ds:>9s} seed={seed} "
                      f"test_mse={r['test_mse']:.4f} "
                      f"V_mean={r['final_V_mean']:.3f} "
                      f"basin_H={r['final_basin_entropy']:.3f} "
                      f"top1={r['final_basin_top1_frac']:.2f}")

    summary: dict = {"rows": rows, "verdict": {}}
    for ds in args.datasets:
        baseline = [r for r in rows if r["dataset"] == ds
                    and r["mode"] == "baseline"]
        if not baseline:
            continue
        base_mse = sum(r["test_mse"] for r in baseline) / len(baseline)
        multi = [r for r in rows if r["dataset"] == ds
                 and r["mode"] == "multi_basin"]
        lyap = [r for r in rows if r["dataset"] == ds
                and r["mode"] == "lyap"]
        if not multi:
            continue
        ad_mse = sum(r["test_mse"] for r in multi) / len(multi)
        delta = (ad_mse - base_mse) / max(base_mse, 1e-9) * 100.0
        basin_ent = sum(r["final_basin_entropy"] for r in multi) / len(multi)
        lyap_v = sum(r["final_V_mean"] for r in lyap) / max(len(lyap), 1)
        multi_v = sum(r["final_V_mean"] for r in multi) / len(multi)
        summary["verdict"][ds] = {
            "baseline_mse": base_mse,
            "lyap_mse": sum(r["test_mse"] for r in lyap) / max(len(lyap), 1),
            "multi_basin_mse": ad_mse,
            "delta_pct_vs_baseline": delta,
            "mean_basin_entropy": basin_ent,
            "log_n_basin": math.log(3),
            "lyap_V_mean": lyap_v,
            "multi_basin_V_mean": multi_v,
            "h1_task_safe": abs(delta) <= 10.0,
            "h2_basin_diverse": basin_ent >= math.log(3) * 0.5,
            "h3_multi_basin_V_le_lyap": multi_v <= lyap_v + 1e-3,
        }

    print("\n=== Verdict (round 244 hypotheses) ===")
    for ds, v in summary["verdict"].items():
        print(f"[{ds:>9s}] Δ%_vs_baseline={v['delta_pct_vs_baseline']:+.1f} "
              f"basin_H={v['mean_basin_entropy']:.3f}/{v['log_n_basin']:.3f} "
              f"V_multi={v['multi_basin_V_mean']:.3f} V_lyap={v['lyap_V_mean']:.3f}  "
              f"H1={'✓' if v['h1_task_safe'] else '✗'} "
              f"H2={'✓' if v['h2_basin_diverse'] else '✗'} "
              f"H3={'✓' if v['h3_multi_basin_V_le_lyap'] else '✗'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()