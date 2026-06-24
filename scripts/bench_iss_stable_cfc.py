#!/usr/bin/env python3
"""Benchmark for ISSStableCfCCell (arXiv:2606.14136 response, round 242).

Trains CfC with and without an ISS (Input-to-State Stability) certificate
on three toy datasets (toy_sin, structured, random) and reports:

  - task loss (test MSE)
  - final ISS loss (mean over T)
  - final bound_ratio (V_next / bound, <1 means ISS satisfied)
  - final V_mean and x_norm_sq_mean (sanity check on coupling)

Hypotheses:
  H1 (task safety): with ``iss_lambda=0.1``, ``pd_lambda=0.1``, task loss
      changes by <= 5% relative to baseline. (Round 240/241 both regressed
      on random; ISS should be more robust because beta allows input
      drive without exploding V.)
  H2 (ISS holds): final bound_ratio < 1.0 — the model actually satisfies
      the ISS contraction.
  H3 (input-V coupling): V(h) positively correlates with ||x||^2 (R^2 > 0.5
      across the test trajectory).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lnn.core.iss_stable_cfc import ISSStableCfCCell  # noqa: E402

DEVICE = "cpu"


def make_dataset(name: str, T: int = 64, d_in: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 4 * 3.141592653589793, T).unsqueeze(-1)
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
    def __init__(self, d_in: int, d_h: int, d_out: int, use_iss: bool,
                 alpha: float = 0.05, beta: float = 0.01):
        super().__init__()
        self.use_iss = use_iss
        self.cell = ISSStableCfCCell(d_in, d_h, alpha=alpha, beta=beta)
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor, iss_lambda: float = 0.0,
                pd_lambda: float = 0.0) -> tuple[torch.Tensor, list[dict]]:
        B = x.shape[1]
        h = torch.zeros(B, self.cell.hidden_size, device=x.device)
        outs, aux_per_step = [], []
        for t in range(x.shape[0]):
            if self.use_iss:
                h, aux = self.cell.forward_with_aux(
                    x[t], h, iss_lambda=iss_lambda, pd_lambda=pd_lambda,
                )
                aux_per_step.append(aux)
            else:
                h = self.cell(x[t], h)
                # Compute diagnostics even for baseline.
                from lnn.core.iss_stable_cfc import (
                    iss_decay_loss, input_bound_ratio,
                )
                from lnn.core.lyapunov_stable_cfc import lyapunov_value
                with torch.no_grad():
                    V_h = lyapunov_value(h, self.cell.lyapunov_P)
                    V_next = lyapunov_value(h, self.cell.lyapunov_P)
                    x_norm_sq = (x[t] * x[t]).sum(dim=-1)
                    loss = iss_decay_loss(h, h, x[t], self.cell.lyapunov_P,
                                          alpha=self.cell.alpha, beta=self.cell.beta)
                    ratio = input_bound_ratio(h, h, x[t], self.cell.lyapunov_P,
                                              alpha=self.cell.alpha, beta=self.cell.beta)
                    aux_per_step.append({
                        "V_h": V_h, "V_next": V_next, "x_norm_sq": x_norm_sq,
                        "iss_loss": loss, "bound_ratio": ratio,
                    })
            outs.append(self.head(h))
        return torch.stack(outs), aux_per_step


def train_one(name: str, seed: int, use_iss: bool, epochs: int = 100,
              iss_lambda: float = 0.1, pd_lambda: float = 0.1,
              alpha: float = 0.05, beta: float = 0.01, d_h: int = 8):
    torch.manual_seed(seed)
    x, y = make_dataset(name, seed=seed)
    x = x.unsqueeze(1).to(DEVICE)
    y = y.unsqueeze(1).to(DEVICE)

    model = SeqModel(d_in=x.shape[-1], d_h=d_h, d_out=y.shape[-1],
                     use_iss=use_iss, alpha=alpha, beta=beta).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(epochs):
        opt.zero_grad()
        pred, aux = model(x, iss_lambda=iss_lambda, pd_lambda=pd_lambda)
        task = ((pred - y) ** 2).mean()
        loss = task
        if use_iss:
            for a in aux:
                if "iss_loss_total" in a:
                    loss = loss + a["iss_loss_total"]
                if "pd_loss_total" in a:
                    loss = loss + a["pd_loss_total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        pred, aux = model(x, iss_lambda=0.0)
        test_mse = ((pred - y) ** 2).mean().item()
        iss_loss = torch.stack([a["iss_loss"] for a in aux]).mean().item()
        bound_ratio = torch.stack([a["bound_ratio"] for a in aux]).mean().item()
        V_seq = torch.stack([a["V_next"] for a in aux]).mean(dim=-1)
        x_norm_sq_seq = torch.stack([a["x_norm_sq"] for a in aux]).mean(dim=-1)

        # Compute Pearson correlation between V and ||x||^2 across T.
        v_centered = V_seq - V_seq.mean()
        x_centered = x_norm_sq_seq - x_norm_sq_seq.mean()
        denom = (v_centered.norm() * x_centered.norm()).clamp_min(1e-9)
        corr = (v_centered * x_centered).sum() / denom

    return {
        "dataset": name,
        "seed": seed,
        "use_iss": use_iss,
        "test_mse": test_mse,
        "final_iss_loss": iss_loss,
        "final_bound_ratio": bound_ratio,
        "final_V_mean": V_seq.mean().item(),
        "final_x_norm_sq_mean": x_norm_sq_seq.mean().item(),
        "v_x_corr": corr.item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["toy_sin", "structured", "random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--iss_lambda", type=float, default=0.1)
    parser.add_argument("--pd_lambda", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--d_h", type=int, default=8)
    parser.add_argument("--out", type=str,
                        default="analysis/iss_stable_cfc_bench.json")
    args = parser.parse_args()

    results = []
    for ds in args.datasets:
        for seed in args.seeds:
            base = train_one(ds, seed, use_iss=False, epochs=args.epochs,
                             alpha=args.alpha, beta=args.beta, d_h=args.d_h)
            iss = train_one(ds, seed, use_iss=True, epochs=args.epochs,
                            iss_lambda=args.iss_lambda, pd_lambda=args.pd_lambda,
                            alpha=args.alpha, beta=args.beta, d_h=args.d_h)
            results.append({"condition": "baseline", **base})
            results.append({"condition": "iss", **iss})

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[bench] wrote {len(results)} rows -> {out_path}")

    print("\n=== Summary (mean over seeds) ===")
    print(f"{'dataset':<12} {'cond':<10} {'test_mse':<10} {'iss_loss':<10} "
          f"{'bnd_ratio':<10} {'V_mean':<10} {'x_n2':<10} {'corr':<8}")
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        by_key.setdefault((r["dataset"], r["condition"]), []).append(r)
    for ds in args.datasets:
        for cond in ("baseline", "iss"):
            rows = by_key[(ds, cond)]
            mse = sum(r["test_mse"] for r in rows) / len(rows)
            il = sum(r["final_iss_loss"] for r in rows) / len(rows)
            br = sum(r["final_bound_ratio"] for r in rows) / len(rows)
            vm = sum(r["final_V_mean"] for r in rows) / len(rows)
            xn = sum(r["final_x_norm_sq_mean"] for r in rows) / len(rows)
            co = sum(r["v_x_corr"] for r in rows) / len(rows)
            print(f"{ds:<12} {cond:<10} {mse:<10.4f} {il:<10.4f} "
                  f"{br:<10.4f} {vm:<10.4f} {xn:<10.4f} {co:<8.3f}")

    print("\n=== Verdicts ===")
    for ds in args.datasets:
        b = by_key[(ds, "baseline")]
        i = by_key[(ds, "iss")]
        b_mse = sum(r["test_mse"] for r in b) / len(b)
        i_mse = sum(r["test_mse"] for r in i) / len(i)
        i_br = sum(r["final_bound_ratio"] for r in i) / len(i)
        i_corr = sum(r["v_x_corr"] for r in i) / len(i)

        delta = (i_mse - b_mse) / max(b_mse, 1e-9) * 100.0
        h1 = abs(delta) <= 5.0
        h2 = i_br < 1.0
        h3 = i_corr > 0.5
        print(f"{ds:<12} H1(task ±5%) {'OK' if h1 else 'FAIL'} "
              f"(Δ={delta:+.1f}%)  H2(bnd<1) {'OK' if h2 else 'FAIL'} "
              f"(ratio={i_br:.3f})  H3(corr>0.5) {'OK' if h3 else 'FAIL'} "
              f"(corr={i_corr:.3f})")


if __name__ == "__main__":
    main()