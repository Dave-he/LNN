#!/usr/bin/env python3
"""Benchmark for PulseGatedLiquidTauCfCCell (round 284).

Grounded in arXiv:2603.00153 (Pulse-Driven Neural Architecture). Tests
whether adding a learnable oscillatory pulse ``A·sin(ω·t+φ(h))`` on top of
the r280 blend-gated liquid τ (a) helps the periodic toy_sin task, (b) is
safe on noise, (c) improves robustness to eval-time input gaps, and (d)
does so via oscillatory STRUCTURE — not added capacity — verified by an
RMS-matched non-oscillatory 'noise' control.

Modes:
  * static_tau   — r267 production (static per-neuron τ)
  * gated_blend  — r280 production (blend gate, no pulse)  [baseline]
  * pulse_sin    — r284 NEW (blend gate + learnable sin pulse)
  * pulse_noise  — r284 mechanism control (blend gate + RMS-matched
                   non-oscillatory drive)

Robustness (H3): each trained model is also evaluated on a gap-corrupted
test set where a fraction of input timesteps are zeroed (temporal dropout).
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
from lnn.core.pulse_gated_liquid_tau_cfc import (  # noqa: E402
    PulseGatedLiquidTauCfCCell,
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
    """Zero a fraction ``p`` of input timesteps (temporal dropout / gaps)."""
    g = torch.Generator().manual_seed(seed)
    B, T, _ = x.shape
    keep = (torch.rand(B, T, 1, generator=g) > p).float()
    return x * keep


class SeqModel(nn.Module):
    def __init__(self, cell, hidden_size, entropy_lambda):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(hidden_size, 1)
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
    "gated_blend": dict(kind="blend"),
    "pulse_sin": dict(kind="pulse", pulse_mode="sin"),
    "pulse_noise": dict(kind="pulse", pulse_mode="noise"),
}
_COMMON = dict(input_size=1, hidden_size=128, density=0.3,
               ste_temperature=1.0, entropy_lambda=0.1)


def make_model(cfg):
    if cfg["kind"] == "static":
        cell = STEWithEntropy(**_COMMON)
    elif cfg["kind"] == "blend":
        cell = BlendGatedLiquidTauCfCCell(
            liquid_tau_strength=1.0, pred_gate_beta=4.0, ema_gamma=0.5,
            gate_mode="blend", **_COMMON)
    else:  # pulse
        cell = PulseGatedLiquidTauCfCCell(
            liquid_tau_strength=1.0, pred_gate_beta=4.0, ema_gamma=0.5,
            gate_mode="blend", pulse_strength=1.0, pulse_amp_init=0.1,
            pulse_mode=cfg["pulse_mode"], state_phase=True, **_COMMON)
    return SeqModel(cell, _COMMON["hidden_size"], _COMMON["entropy_lambda"])


def train_one(model, x_tr, y_tr, x_ev, y_ev, x_gap, epochs, lr, bs, device):
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = x_tr.shape[0]
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(N)
        xb_all, yb_all = x_tr[perm], y_tr[perm]
        ep, nb = 0.0, 0
        for i in range(0, N, bs):
            xb = xb_all[i:i + bs].to(device)
            yb = yb_all[i:i + bs].to(device)
            if xb.shape[0] == 0:
                continue
            pred = model(xb)
            mse = (pred - yb).pow(2).mean()
            loss = mse + model.extra_loss()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep += float(mse.item())
            nb += 1
        last = ep / max(nb, 1)
    model.eval()
    with torch.no_grad():
        clean = float((model(x_ev.to(device)) - y_ev.to(device)).pow(2).mean())
        gap = float((model(x_gap.to(device)) - y_ev.to(device)).pow(2).mean())
    return {"test_mse": clean, "gap_mse": gap,
            "gap_ratio": gap / max(clean, 1e-12), "train_loss_last": last}


def pulse_diag(model, x_sample):
    cell = model.cell
    d = {"n_params": sum(p.numel() for p in model.parameters())}
    if isinstance(cell, PulseGatedLiquidTauCfCCell):
        with torch.no_grad():
            _, _, aux = cell(x_sample, return_aux=True)
        d.update(pulse_amp_mean=aux["pulse_amp_mean"],
                 pulse_rms=aux["pulse_rms"], gate_mean=aux["gate_mean"])
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
                    default="analysis/pulse_gated_bench.json")
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
                diag = pulse_diag(model, x[ntr:][:16].to(device))
                results["cells"].append({
                    "mode": mode, "dataset": ds, "seed": seed,
                    "test_mse": out["test_mse"], "gap_mse": out["gap_mse"],
                    "gap_ratio": out["gap_ratio"],
                    "train_loss_last": out["train_loss_last"],
                    "elapsed_sec": round(el, 2), "diagnostics": diag})
                amp = diag.get("pulse_amp_mean", float("nan"))
                print(f"[bench] {mode:12s} {ds:10s} s{seed} "
                      f"mse={out['test_mse']:.5f} gap={out['gap_mse']:.5f} "
                      f"gr={out['gap_ratio']:.2f} amp={amp:.3f} ({el:.1f}s)")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {outp}")

    # Summaries.
    def agg(key):
        s = {}
        for c in results["cells"]:
            s.setdefault((c["mode"], c["dataset"]), []).append(c[key])
        return s
    mse, gr = agg("test_mse"), agg("gap_ratio")
    print("\n[bench] mean test_mse (clean) | gap_ratio:")
    hdr = " | ".join(f"{d:>11s}" for d in args.datasets)
    print(f"{'mode':12s} | {hdr}")
    for mode in args.modes:
        cells = []
        for d in args.datasets:
            v = mse.get((mode, d), [])
            r = gr.get((mode, d), [])
            vm = sum(v) / len(v) if v else float("nan")
            rm = sum(r) / len(r) if r else float("nan")
            cells.append(f"{vm:.5f}/{rm:.2f}")
        print(f"{mode:12s} | " + " | ".join(f"{c:>11s}" for c in cells))

    print("\n[bench] Δ%% vs gated_blend (negative=better) on clean mse:")
    for d in args.datasets:
        base = mse.get(("gated_blend", d), [])
        if not base:
            continue
        bm = sum(base) / len(base)
        line = f"  {d:11s}: blend={bm:.5f}"
        for m in ("pulse_sin", "pulse_noise"):
            v = mse.get((m, d), [])
            if v:
                vm = sum(v) / len(v)
                line += f"  {m}={vm:.5f} ({100*(vm-bm)/max(abs(bm),1e-12):+.1f}%)"
        print(line)


if __name__ == "__main__":
    main()
