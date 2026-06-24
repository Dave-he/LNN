#!/usr/bin/env python3
"""Benchmark for ControllabilityCfCCell (arXiv:2606.08431 response, round 241).

Trains CfC with and without a controllability regularizer on three toy
datasets (toy_sin, structured, random) and reports:

  - task loss (test MSE)
  - mean input sensitivity c_t (over T, over seeds)
  - final controllability loss
  - input Jacobian norm (diagnostic)

Hypotheses:
  H1 (task safety): with ``ctrl_lambda=0.1`` and ``margin=0.05``, task
      loss on clean input changes by <= 5% relative to baseline.
  H2 (input sensitivity up): mean c_t in the controllability condition
      is >= 1.2x baseline (model is more input-driven).
  H3 (Jacobian non-degenerate): the mean Jacobian norm stays above 0
      throughout training (no input-saturating collapse).
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

from lnn.core.controllability_cfc import ControllabilityCfCCell  # noqa: E402

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
    def __init__(self, d_in: int, d_h: int, d_out: int, use_ctrl: bool,
                 margin: float = 0.05):
        super().__init__()
        self.use_ctrl = use_ctrl
        self.cell = ControllabilityCfCCell(d_in, d_h, margin=margin)
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor, ctrl_lambda: float = 0.0
                ) -> tuple[torch.Tensor, list[dict]]:
        B = x.shape[1]
        h = torch.zeros(B, self.cell.hidden_size, device=x.device)
        outs, aux_per_step = [], []
        for t in range(x.shape[0]):
            if self.use_ctrl:
                h, aux = self.cell.forward_with_aux(x[t], h, ctrl_lambda=ctrl_lambda)
                aux_per_step.append(aux)
            else:
                h = self.cell(x[t], h)
                # Still compute c_t for the diagnostic baseline.
                from lnn.core.controllability_cfc import input_sensitivity
                c_t = input_sensitivity(x[t], h, self.cell.cell).detach()
                aux_per_step.append({"c_t": c_t})
            outs.append(self.head(h))
        return torch.stack(outs), aux_per_step


def train_one(name: str, seed: int, use_ctrl: bool, epochs: int = 100,
              ctrl_lambda: float = 0.1, margin: float = 0.05, d_h: int = 8):
    torch.manual_seed(seed)
    x, y = make_dataset(name, seed=seed)
    x = x.unsqueeze(1).to(DEVICE)
    y = y.unsqueeze(1).to(DEVICE)

    model = SeqModel(d_in=x.shape[-1], d_h=d_h, d_out=y.shape[-1],
                     use_ctrl=use_ctrl, margin=margin).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(epochs):
        opt.zero_grad()
        pred, aux = model(x, ctrl_lambda=ctrl_lambda)
        task = ((pred - y) ** 2).mean()
        loss = task
        if use_ctrl:
            for a in aux:
                if "ctrl_loss_total" in a:
                    loss = loss + a["ctrl_loss_total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Final eval (always compute jac_norm for both conditions, regardless of
    # ctrl_lambda at eval time).
    from lnn.core.controllability_cfc import input_jacobian_norm as _jac
    with torch.no_grad():
        pred, aux = model(x, ctrl_lambda=0.0)
        test_mse = ((pred - y) ** 2).mean().item()
        c_t_seq = torch.stack([a["c_t"] for a in aux]).mean(dim=-1)
        c_t_mean = c_t_seq.mean().item()
        if use_ctrl:
            ctrl_loss = torch.stack([a["ctrl_loss"] for a in aux]).mean().item()
        else:
            ctrl_loss = None

    # Compute Jacobian diagnostic on the trained cell (separately so both
    # baseline and ctrl report it). Use enable_grad so autograd works inside
    # the eval block (input_jacobian_norm requires grad-tracking inputs).
    h_eval = torch.zeros(x.shape[1], model.cell.hidden_size)
    jac_vals = []
    with torch.enable_grad():
        for t in range(x.shape[0]):
            jn = _jac(x[t], h_eval, model.cell.cell).mean().item()
            jac_vals.append(jn)
            h_eval = model.cell.cell(x[t], h_eval).detach()
    jacobian_norm = sum(jac_vals) / len(jac_vals)

    return {
        "dataset": name,
        "seed": seed,
        "use_ctrl": use_ctrl,
        "test_mse": test_mse,
        "c_t_mean": c_t_mean,
        "final_ctrl_loss": ctrl_loss,
        "final_jacobian_norm": jacobian_norm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["toy_sin", "structured", "random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ctrl_lambda", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--d_h", type=int, default=8)
    parser.add_argument("--out", type=str,
                        default="analysis/controllability_cfc_bench.json")
    args = parser.parse_args()

    results = []
    for ds in args.datasets:
        for seed in args.seeds:
            base = train_one(ds, seed, use_ctrl=False, epochs=args.epochs,
                             margin=args.margin, d_h=args.d_h)
            ctrl = train_one(ds, seed, use_ctrl=True, epochs=args.epochs,
                             ctrl_lambda=args.ctrl_lambda, margin=args.margin,
                             d_h=args.d_h)
            results.append({"condition": "baseline", **base})
            results.append({"condition": "ctrl", **ctrl})

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[bench] wrote {len(results)} rows -> {out_path}")

    print("\n=== Summary (mean over seeds) ===")
    print(f"{'dataset':<12} {'cond':<10} {'test_mse':<10} {'c_t':<10} "
          f"{'ctrl_loss':<12} {'jac_norm':<10}")
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        by_key.setdefault((r["dataset"], r["condition"]), []).append(r)
    for ds in args.datasets:
        for cond in ("baseline", "ctrl"):
            rows = by_key[(ds, cond)]
            mse = sum(r["test_mse"] for r in rows) / len(rows)
            ct = sum(r["c_t_mean"] for r in rows) / len(rows)
            cl = sum((r["final_ctrl_loss"] or 0.0) for r in rows) / len(rows)
            jn = sum((r["final_jacobian_norm"] or 0.0) for r in rows) / len(rows)
            print(f"{ds:<12} {cond:<10} {mse:<10.4f} {ct:<10.4f} "
                  f"{cl:<12.4f} {jn:<10.4f}")

    print("\n=== Verdicts ===")
    for ds in args.datasets:
        b = by_key[(ds, "baseline")]
        c = by_key[(ds, "ctrl")]
        b_mse = sum(r["test_mse"] for r in b) / len(b)
        c_mse = sum(r["test_mse"] for r in c) / len(c)
        b_ct = sum(r["c_t_mean"] for r in b) / len(b)
        c_ct = sum(r["c_t_mean"] for r in c) / len(c)
        c_jn = sum((r["final_jacobian_norm"] or 0.0) for r in c) / len(c)

        delta = (c_mse - b_mse) / max(b_mse, 1e-9) * 100.0
        ct_ratio = c_ct / max(b_ct, 1e-9)

        h1 = abs(delta) <= 5.0
        h2 = ct_ratio >= 1.20
        h3 = c_jn > 0
        print(f"{ds:<12} H1(task ±5%) {'OK' if h1 else 'FAIL'} "
              f"(Δ={delta:+.1f}%)  H2(c_t ≥1.2x) {'OK' if h2 else 'FAIL'} "
              f"(ratio={ct_ratio:.2f}x)  H3(jac>0) {'OK' if h3 else 'FAIL'} "
              f"(jac={c_jn:.4f})")


if __name__ == "__main__":
    main()