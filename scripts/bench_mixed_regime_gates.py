#!/usr/bin/env python3
"""Mixed-regime benchmark for the liquid-τ gate line (round 281).

Round 280 proved the homogeneous toy benchmark is SATURATED: toy_sin is
solved to <1e-4 by every mode, so init noise dominates and the four gate
variants cannot be discriminated. But the gates were designed for a
setting NONE of the r277-280 datasets contain: WITHIN-sequence regime
shifts (natural-gas LNN arXiv:2604.24788 — "limit responsiveness when
regimes shift rapidly").

This bench builds that setting. Each sequence transitions between three
regimes:

    seg 0 [0:S]    : smooth sine   (predictable, low |Δ²x|)
    seg 1 [S:2S]   : i.i.d. noise  (unpredictable, high |Δx| AND |Δ²x|)
    seg 2 [2S:3S]  : structured    (piecewise-constant levels)

A good gate keeps the liquid τ ACTIVE in segs 0/2 (predictable) and
COLLAPSES it in seg 1 (noise). We report MSE per-segment AND overall.

Five modes (the full gate line):
  * static_tau    — r267 (static per-neuron τ, no gate)
  * liquid_tau    — r277 (ungated input-dependent τ)
  * gated_vel     — r278 (velocity gate |Δ¹x|)
  * gated_accel   — r279 (acceleration gate |Δ²x|)
  * gated_blend   — r280 (max(velocity, acceleration))

Shared params match r267-r280 production:
  input_size=1, hidden=192, density=0.3, ste_temperature=1.0,
  entropy_lambda=0.1, 100 epochs, lr=1e-2, batch=16.
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
from lnn.core.liquid_tau_ste_cfc import LiquidTauSTECfCCell  # noqa: E402
from lnn.core.pred_gated_liquid_tau_cfc import (  # noqa: E402
    PredictabilityGatedLiquidTauCfCCell,
)
from lnn.core.accel_gated_liquid_tau_cfc import (  # noqa: E402
    AccelGatedLiquidTauCfCCell,
)
from lnn.core.blend_gated_liquid_tau_cfc import (  # noqa: E402
    BlendGatedLiquidTauCfCCell,
)


# ---------------------------------------------------------------------------
# Mixed-regime data generator (within-sequence regime shifts)
# ---------------------------------------------------------------------------
def make_mixed_regime(seg_len=32, n_samples=256, seed=0):
    """Three concatenated regimes: smooth sine | noise | structured.

    Returns (x, y) for one-step-ahead prediction, each (n, 3*seg_len, 1),
    plus segment boundary indices for per-segment loss attribution.
    """
    g = torch.Generator().manual_seed(seed)
    T = 3 * seg_len
    y = torch.zeros(n_samples, T + 1)

    # seg 0: smooth sine (a few cycles across the segment)
    t0 = torch.linspace(0, 1, seg_len).unsqueeze(0)
    y[:, :seg_len] = torch.sin(2 * math.pi * 2.0 * t0)

    # seg 1: i.i.d. noise
    y[:, seg_len:2 * seg_len] = torch.randn(
        n_samples, seg_len, generator=g) * 0.5

    # seg 2: piecewise-constant structured (4 sub-levels)
    levels = torch.tensor([0.0, 1.0, -0.5, 0.7])
    sub = max(seg_len // 4, 1)
    for i in range(4):
        s = 2 * seg_len + i * sub
        e = 2 * seg_len + (i + 1) * sub if i < 3 else 2 * seg_len + seg_len
        y[:, s:e] = levels[i]
    y[:, 2 * seg_len:3 * seg_len] += torch.randn(
        n_samples, seg_len, generator=g) * 0.01
    # final target column (T index)
    y[:, T] = y[:, T - 1]

    x = y[:, :-1].unsqueeze(-1)
    ytar = y[:, 1:].unsqueeze(-1)
    # segment boundaries in the T-length prediction axis
    bounds = {"smooth": (0, seg_len),
              "noise": (seg_len, 2 * seg_len),
              "structured": (2 * seg_len, 3 * seg_len)}
    return x, ytar, bounds


# homogeneous controls (calibration) — from r280
def make_toy_sin(T=96, n_samples=256, seed=0):
    t = torch.linspace(0, 1, T + 1).unsqueeze(0).repeat(n_samples, 1)
    y = torch.sin(2 * math.pi * t)
    return y[:, :-1].unsqueeze(-1), y[:, 1:].unsqueeze(-1), None


DATA_FACTORIES = {"mixed_regime": make_mixed_regime, "toy_sin": make_toy_sin}


class ReadoutHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, h):
        return self.head(h)


class SeqModel(nn.Module):
    def __init__(self, cell, hidden_size, entropy_lambda):
        super().__init__()
        self.cell = cell
        self.head = ReadoutHead(hidden_size)
        self.entropy_lambda = float(entropy_lambda)

    def forward(self, x):
        out, _ = self.cell(x)
        return self.head(out)

    def extra_loss(self):
        if self.entropy_lambda <= 0:
            return torch.tensor(0.0)
        return self.cell.extra_loss()


MODES = {
    "static_tau": dict(kind="static"),
    "liquid_tau": dict(kind="liquid"),
    "gated_vel": dict(kind="gated_vel", pred_gate_beta=4.0, ema_gamma=0.5),
    "gated_accel": dict(kind="gated_accel", pred_gate_beta=4.0, ema_gamma=0.5),
    "gated_blend": dict(kind="gated_blend", pred_gate_beta=4.0, ema_gamma=0.5),
}
_COMMON = dict(input_size=1, hidden_size=192, density=0.3,
               ste_temperature=1.0, entropy_lambda=0.1)


def make_model(cfg):
    if cfg["kind"] == "static":
        cell = STEWithEntropy(**_COMMON)
    elif cfg["kind"] == "liquid":
        cell = LiquidTauSTECfCCell(liquid_tau_strength=1.0, **_COMMON)
    elif cfg["kind"] == "gated_vel":
        cell = PredictabilityGatedLiquidTauCfCCell(
            liquid_tau_strength=1.0, pred_gate_beta=cfg["pred_gate_beta"],
            ema_gamma=cfg["ema_gamma"], **_COMMON)
    elif cfg["kind"] == "gated_accel":
        cell = AccelGatedLiquidTauCfCCell(
            liquid_tau_strength=1.0, pred_gate_beta=cfg["pred_gate_beta"],
            ema_gamma=cfg["ema_gamma"], diff_order=2, **_COMMON)
    else:  # gated_blend
        cell = BlendGatedLiquidTauCfCCell(
            liquid_tau_strength=1.0, pred_gate_beta=cfg["pred_gate_beta"],
            ema_gamma=cfg["ema_gamma"], gate_mode="blend", **_COMMON)
    return SeqModel(cell, _COMMON["hidden_size"], _COMMON["entropy_lambda"])


def train_one(model, x_tr, y_tr, x_ev, y_ev, bounds, epochs, lr,
              batch_size, device):
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = x_tr.shape[0]
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        xs, ys = x_tr[perm], y_tr[perm]
        ep, nb = 0.0, 0
        for i in range(0, N, batch_size):
            xb = xs[i:i + batch_size].to(device)
            yb = ys[i:i + batch_size].to(device)
            if xb.shape[0] == 0:
                continue
            pred = model(xb)
            mse = (pred - yb).pow(2).mean()
            loss = mse + model.extra_loss()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            ep += float(mse.item()); nb += 1
        losses.append(ep / max(nb, 1))
    model.eval()
    with torch.no_grad():
        pred = model(x_ev.to(device))
        yv = y_ev.to(device)
        overall = float((pred - yv).pow(2).mean().item())
        per_seg = {}
        if bounds is not None:
            for name, (s, e) in bounds.items():
                per_seg[name] = float(
                    (pred[:, s:e] - yv[:, s:e]).pow(2).mean().item())
    return {"test_mse": overall, "per_segment": per_seg,
            "train_loss_last": losses[-1]}


def collect_diag(model, x_sample, bounds):
    cell = model.cell
    diag = {"n_params": sum(p.numel() for p in model.parameters())}
    with torch.no_grad():
        if isinstance(cell, PredictabilityGatedLiquidTauCfCCell):
            out, _, aux = cell(x_sample, return_aux=True)
            diag.update(gate_mean=aux["gate_mean"],
                        tau_temporal_std=aux["tau_temporal_std"])
            # per-segment gate mean (re-run to grab per-step gate)
        elif isinstance(cell, LiquidTauSTECfCCell):
            _, _, aux = cell(x_sample, return_aux=True)
            diag.update(gate_mean=1.0, tau_temporal_std=aux["tau_temporal_std"])
        else:
            diag.update(gate_mean=float("nan"), tau_temporal_std=0.0)
    return diag


def per_segment_gate(model, x_sample, bounds):
    """Return mean gate value within each regime segment via the cell's
    per-step gate diagnostics. Empty if the cell has no gate."""
    cell = model.cell
    if bounds is None or not isinstance(
            cell, PredictabilityGatedLiquidTauCfCCell):
        return {}
    with torch.no_grad():
        _, _, aux = cell(x_sample, return_aux=True)
    # aux exposes only scalar gate summaries (gate_mean over all steps),
    # not the per-step series, so per-segment gate is not available here;
    # the overall gate_mean in collect_diag is the reported proxy.
    return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seg-len", type=int, default=32)
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--datasets", nargs="+", default=["mixed_regime"])
    p.add_argument("--out", type=str,
                   default="analysis/mixed_regime_gates_bench.json")
    p.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bench] device={device}")

    results = {"config": {"epochs": args.epochs, "lr": args.lr,
                          "batch_size": args.batch_size,
                          "seg_len": args.seg_len, "seeds": args.seeds,
                          "datasets": args.datasets, "modes": args.modes},
               "cells": []}

    for mode_name in args.modes:
        cfg = MODES[mode_name]
        for ds in args.datasets:
            for seed in args.seeds:
                torch.manual_seed(seed)
                if ds == "mixed_regime":
                    x, y, bounds = make_mixed_regime(
                        seg_len=args.seg_len, n_samples=args.n_samples,
                        seed=seed)
                else:
                    x, y, bounds = make_toy_sin(
                        T=3 * args.seg_len, n_samples=args.n_samples, seed=seed)
                ntr = int(0.8 * x.shape[0])
                model = make_model(cfg)
                t0 = time.time()
                out = train_one(model, x[:ntr], y[:ntr], x[ntr:], y[ntr:],
                                 bounds, args.epochs, args.lr,
                                 args.batch_size, device)
                el = time.time() - t0
                diag = collect_diag(model, x[ntr:][:16].to(device), bounds)
                results["cells"].append({
                    "mode": mode_name, "dataset": ds, "seed": seed,
                    "test_mse": out["test_mse"],
                    "per_segment": out["per_segment"],
                    "train_loss_last": out["train_loss_last"],
                    "elapsed_sec": round(el, 2), "diagnostics": diag})
                ps = out["per_segment"]
                ps_str = " ".join(f"{k}={v:.5f}" for k, v in ps.items())
                print(f"[bench] {mode_name:13s} {ds:12s} seed={seed} "
                      f"overall={out['test_mse']:.5f} [{ps_str}] "
                      f"gate={diag['gate_mean']:.3f} ({el:.1f}s)")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {outp}")

    # Summary: overall + per-segment means by mode (mixed_regime only).
    print("\n[bench] mixed_regime mean MSE (overall | smooth | noise | structured):")
    print(f"{'mode':13s} | {'overall':>9s} | {'smooth':>9s} | "
          f"{'noise':>9s} | {'structured':>10s}")
    agg = {}
    for c in results["cells"]:
        if c["dataset"] != "mixed_regime":
            continue
        a = agg.setdefault(c["mode"], {"overall": [], "smooth": [],
                                       "noise": [], "structured": []})
        a["overall"].append(c["test_mse"])
        for k in ("smooth", "noise", "structured"):
            if k in c["per_segment"]:
                a[k].append(c["per_segment"][k])

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    for m in args.modes:
        if m not in agg:
            continue
        a = agg[m]
        print(f"{m:13s} | {mean(a['overall']):>9.5f} | "
              f"{mean(a['smooth']):>9.5f} | {mean(a['noise']):>9.5f} | "
              f"{mean(a['structured']):>10.5f}")


if __name__ == "__main__":
    main()
