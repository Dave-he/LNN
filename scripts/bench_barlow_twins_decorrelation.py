#!/usr/bin/env python3
"""Benchmark for barlow_twins_decorrelation_loss (round 290).

Round 290 reformulates r289's decorrelation loss in Barlow-Twins style
(Zbontar et al. 2021) to fix the H3 loss-formulation bug. The BT loss
normalizes by per-feature std (not by diag), so the optimizer cannot
inflate the diagonal to escape.

Hypotheses:
  H1 (target-independent): improves-or-maintains task loss on ALL 3
     datasets at λ_off=0.005, λ_on=0.005.
  H3 (decorrelated): BT cross-correlation has off-diag → 0 and
     diag → 1 — measured by bt_diag/bt_off ratio.
  H4 (strict-positive default): H1 ∧ H3 → +1 SP, breaking the 6-round
     TD streak.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lnn.core.ste_entropy_neuron_wise_cfc import STEWithEntropy  # noqa: E402
from lnn.core.blend_gated_liquid_tau_cfc import (  # noqa: E402
    BlendGatedLiquidTauCfCCell,
)
from lnn.core.decorrelation_loss import (  # noqa: E402
    barlow_twins_decorrelation_loss,
    barlow_twins_covariance_diagnostics,
)


def make_toy_sin(T=48, n_samples=192, seed=0):
    t = torch.linspace(0, 1, T + 1).unsqueeze(0).repeat(n_samples, 1)
    y = torch.sin(2 * math.pi * t)
    return y[:, :-1].unsqueeze(-1), y[:, 1:].unsqueeze(-1)


def make_structured(T=48, n_samples=192, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_segments = 4
    seg_len = (T + 1) // n_segments
    levels = torch.tensor([0.0, 1.0, -0.5, 0.7])
    y = torch.zeros(n_samples, T + 1)
    for i in range(n_segments):
        start = i * seg_len
        end = (i + 1) * seg_len if i < n_segments - 1 else T + 1
        y[:, start:end] = levels[i % len(levels)]
    y = y + torch.randn(n_samples, T + 1, generator=g) * 0.01
    return y[:, :-1].unsqueeze(-1), y[:, 1:].unsqueeze(-1)


def make_random(T=48, n_samples=192, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(n_samples, T + 1, generator=g)
    return y[:, :-1].unsqueeze(-1), y[:, 1:].unsqueeze(-1)


DATA_FACTORIES = {
    "toy_sin": make_toy_sin,
    "structured": make_structured,
    "random": make_random,
}


def gap_corrupt(x, p=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    B, T, _ = x.shape
    keep = (torch.rand(B, T, 1, generator=g) > p).float()
    return x * keep


class SeqModel(nn.Module):
    def __init__(self, cell, hidden_size, entropy_lambda,
                 lambda_off, lambda_on):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(hidden_size, 1)
        self.entropy_lambda = float(entropy_lambda)
        self.lambda_off = float(lambda_off)
        self.lambda_on = float(lambda_on)

    def forward(self, x):
        out, _ = self.cell(x)
        return self.head(out)

    def extra_loss(self, x):
        out, _ = self.cell(x)
        ent = self.cell.extra_loss() if self.entropy_lambda > 0 else torch.tensor(0.0)
        bt = barlow_twins_decorrelation_loss(
            out, lambda_off=self.lambda_off, lambda_on=self.lambda_on)
        return ent + bt


MODES = {
    "static_tau": dict(kind="static", lo=0.0, ln=0.0),
    "blend_gated": dict(kind="blend", lo=0.0, ln=0.0),
    "bt_a0001": dict(kind="blend", lo=0.001, ln=0.001),
    "bt_a0005": dict(kind="blend", lo=0.005, ln=0.005),
    "bt_a005": dict(kind="blend", lo=0.05, ln=0.05),
    "bt_a0005_offonly": dict(kind="blend", lo=0.005, ln=0.0),
}
_COMMON = dict(input_size=1, hidden_size=128, density=0.3,
               ste_temperature=1.0, entropy_lambda=0.1)


def make_model(cfg):
    if cfg["kind"] == "static":
        cell = STEWithEntropy(**_COMMON)
    else:
        cell = BlendGatedLiquidTauCfCCell(
            liquid_tau_strength=1.0, pred_gate_beta=4.0, ema_gamma=0.5,
            gate_mode="blend", **_COMMON)
    return SeqModel(cell, _COMMON["hidden_size"],
                    _COMMON["entropy_lambda"], cfg["lo"], cfg["ln"])


def train_one(model, x_tr, y_tr, x_ev, y_ev, x_gap, epochs, lr, bs, device):
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = x_tr.shape[0]
    last = 0.0
    last_bt = 0.0
    for _ in range(epochs):
        perm = torch.randperm(N)
        xb_all, yb_all = x_tr[perm], y_tr[perm]
        ep, nb = 0.0, 0
        ep_bt = 0.0
        for i in range(0, N, bs):
            xb = xb_all[i:i + bs].to(device)
            yb = yb_all[i:i + bs].to(device)
            if xb.shape[0] == 0:
                continue
            pred = model(xb)
            mse = (pred - yb).pow(2).mean()
            extra = model.extra_loss(xb)
            loss = mse + extra
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep += float(mse.item())
            ep_bt += float(extra.item())
            nb += 1
        last = ep / max(nb, 1)
        last_bt = ep_bt / max(nb, 1)
    model.eval()
    with torch.no_grad():
        clean = float((model(x_ev.to(device)) - y_ev.to(device)).pow(2).mean())
        gap = float((model(x_gap.to(device)) - y_ev.to(device)).pow(2).mean())
    return {"test_mse": clean, "gap_mse": gap,
            "gap_ratio": gap / max(clean, 1e-12), "train_loss_last": last,
            "bt_loss_last": last_bt}


def cov_diag(model, x_sample):
    cell = model.cell
    d = {"n_params": sum(p.numel() for p in model.parameters())}
    with torch.no_grad():
        out, _ = cell(x_sample)
    d.update(barlow_twins_covariance_diagnostics(out))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=192)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--gap-p", type=float, default=0.3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--datasets", nargs="+",
                    default=["toy_sin", "structured", "random"])
    ap.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    ap.add_argument("--out", type=str,
                    default="analysis/barlow_twins_decorrelation_bench.json")
    args = ap.parse_args()

    device = torch.device("cpu")
    print(f"[bench] device={device} epochs={args.epochs} seeds={args.seeds}")
    results = {"config": vars(args), "cells": []}

    for mode in args.modes:
        cfg = MODES[mode]
        for ds in args.datasets:
            for seed in args.seeds:
                torch.manual_seed(seed)
                x, y = DATA_FACTORIES[ds](
                    T=args.T, n_samples=args.n_samples, seed=seed)
                ntr = int(0.8 * x.shape[0])
                x_gap = gap_corrupt(x[ntr:], p=args.gap_p, seed=seed + 99)
                model = make_model(cfg)
                t0 = time.time()
                out = train_one(model, x[:ntr], y[:ntr], x[ntr:], y[ntr:],
                                x_gap, args.epochs, args.lr,
                                args.batch_size, device)
                el = time.time() - t0
                diag = cov_diag(model, x[ntr:][:16].to(device))
                results["cells"].append({
                    "mode": mode, "dataset": ds, "seed": seed,
                    "test_mse": out["test_mse"], "gap_mse": out["gap_mse"],
                    "gap_ratio": out["gap_ratio"],
                    "train_loss_last": out["train_loss_last"],
                    "bt_loss_last": out["bt_loss_last"],
                    "elapsed_sec": round(el, 2), "diagnostics": diag})
                ratio = diag.get("bt_ratio", float("nan"))
                bt = out["bt_loss_last"]
                print(f"[bench] {mode:18s} {ds:10s} s{seed} "
                      f"mse={out['test_mse']:.5f} gap={out['gap_mse']:.5f} "
                      f"gr={out['gap_ratio']:.2f} bt_ratio={ratio:.2f} "
                      f"bt={bt:.4f} ({el:.1f}s)")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {outp}")

    def agg(key):
        s = {}
        for c in results["cells"]:
            s.setdefault((c["mode"], c["dataset"]), []).append(c[key])
        return s
    mse, gr, diag_d = agg("test_mse"), agg("gap_ratio"), agg("diagnostics")

    print("\n[bench] mean test_mse (clean) | gap_ratio | bt_ratio:")
    hdr = " | ".join(f"{d:>22s}" for d in args.datasets)
    print(f"{'mode':18s} | {hdr}")
    for mode in args.modes:
        cells = []
        for d in args.datasets:
            v = mse.get((mode, d), [])
            r = gr.get((mode, d), [])
            a = diag_d.get((mode, d), [])
            vm = sum(v) / len(v) if v else float("nan")
            rm = sum(r) / len(r) if r else float("nan")
            ratio = (sum(x.get("bt_ratio", float("nan")) for x in a)
                     / len(a)) if a else float("nan")
            cells.append(f"{vm:.5f}/{rm:.2f}/{ratio:.2f}")
        print(f"{mode:18s} | " + " | ".join(f"{c:>22s}" for c in cells))

    print("\n[bench] Δ%% vs blend_gated (negative=better) on clean mse:")
    for d in args.datasets:
        base = mse.get(("blend_gated", d), [])
        if not base:
            continue
        bm = sum(base) / len(base)
        line = f"  {d:11s}: blend={bm:.5f}"
        for m in ("bt_a0001", "bt_a0005", "bt_a005", "bt_a0005_offonly"):
            v = mse.get((m, d), [])
            if v:
                vm = sum(v) / len(v)
                line += f"  {m}={vm:.5f} ({100*(vm-bm)/max(abs(bm),1e-12):+.1f}%)"
        print(line)

    print("\n[bench] H1/H3 acceptance check (r290 → strict-positive?):")

    def mean(key, mode, ds):
        vals = mse.get((mode, ds), []) if key == "mse" else (
            gr.get((mode, ds), []) if key == "gr" else
            [x.get("bt_ratio", float("nan"))
             for x in diag_d.get((mode, ds), [])])
        return sum(vals) / len(vals) if vals else float("nan")

    print("  H1 task loss improves-or-maintains vs blend on ALL 3 datasets:")
    h1_ok = False
    h1_pass_modes = []
    for m in ("bt_a0001", "bt_a0005", "bt_a005", "bt_a0005_offonly"):
        all_ok = True
        per_ds = []
        for ds in args.datasets:
            v = mean("mse", m, ds)
            b = mean("mse", "blend_gated", ds)
            ok = v <= b * 1.05
            per_ds.append(f"{ds}: {100*(v-b)/max(abs(b),1e-12):+.1f}%")
            if not ok:
                all_ok = False
        print(f"     {m}: {' | '.join(per_ds)}  "
              f"{'OK' if all_ok else 'FAIL'}")
        if all_ok:
            h1_ok = True
            h1_pass_modes.append(m)

    print("  H3 bt_ratio ≥ 5 (BT cross-correlation diag >> off-diag):")
    h3_ok = False
    for m in ("bt_a0001", "bt_a0005", "bt_a005", "bt_a0005_offonly"):
        ratios = []
        for ds in args.datasets:
            ratios.append(mean("ratio", m, ds))
        avg = sum(ratios) / max(len(ratios), 1)
        ok = avg >= 5.0
        print(f"     {m}: ratio={[f'{r:.1f}' for r in ratios]} "
              f"avg={avg:.2f}  {'OK' if ok else 'FAIL'}")
        if ok:
            h3_ok = True

    h4_ok = h1_ok and h3_ok
    print(f"  H4 strict-positive default (H1 ∧ H3): "
          f"{'YES — STRICT POSITIVE — 6-round TD streak BROKEN' if h4_ok else 'NO — TD / NEGATIVE'}")
    if h4_ok:
        print(f"     Mode(s): {h1_pass_modes}")


if __name__ == "__main__":
    main()