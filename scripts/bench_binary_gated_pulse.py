#!/usr/bin/env python3
"""Benchmark for BinaryGatedPulseCfCCell (round 287).

Round 287 abandons the multiplicative-gate family (r284/r285/r286 all
target-dependent) and tests an *additive / threshold gate*:
    `pulse = (g_t > τ) · A · sin(...)`.
The pulse is full strength or exactly zero; no per-step attenuation
that the optimizer must compensate for.

Hypotheses:
  H1 (structured gap_ratio ≤ r284 = 61) at τ ∈ {0.3, 0.5}.
  H2 (random Δ% ≤ +5% vs blend) at τ ∈ {0.3, 0.5}.
  H3 (random pulse_amp ≤ 0.20) — A-chase killed.
  H4 (H1 ∧ H2 ∧ H3) → strict-positive default — first in the line.
  H5 (threshold=0 ≡ r284) — unit-tested.
  H6 (threshold=10 ≡ r280) — unit-tested.

Modes:
  * static_tau        — r267 baseline
  * blend_gated       — r280 (primary baseline)
  * pulse_sin         — r284 (unconditional pulse)
  * binary_tau_03     — r287 threshold τ=0.3
  * binary_tau_05     — r287 threshold τ=0.5 (default)
  * binary_tau_07     — r287 threshold τ=0.7
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
from lnn.core.binary_gated_pulse_cfc import (  # noqa: E402
    BinaryGatedPulseCfCCell,
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
    "blend_gated": dict(kind="blend"),
    "pulse_sin": dict(kind="pulse"),
    "binary_tau_03": dict(kind="binary", threshold=0.3),
    "binary_tau_05": dict(kind="binary", threshold=0.5),
    "binary_tau_07": dict(kind="binary", threshold=0.7),
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
    elif cfg["kind"] == "pulse":
        cell = PulseGatedLiquidTauCfCCell(
            liquid_tau_strength=1.0, pred_gate_beta=4.0, ema_gamma=0.5,
            gate_mode="blend", pulse_strength=1.0, pulse_amp_init=0.1,
            pulse_mode="sin", state_phase=True, **_COMMON)
    else:  # binary
        cell = BinaryGatedPulseCfCCell(
            liquid_tau_strength=1.0, pred_gate_beta=4.0, ema_gamma=0.5,
            gate_mode="blend", pulse_strength=1.0, pulse_amp_init=0.1,
            pulse_mode="sin", state_phase=True,
            threshold=cfg["threshold"], **_COMMON)
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
    if isinstance(cell, (PulseGatedLiquidTauCfCCell,
                          BinaryGatedPulseCfCCell)):
        with torch.no_grad():
            _, _, aux = cell(x_sample, return_aux=True)
        d.update(pulse_amp_mean=aux["pulse_amp_mean"],
                 pulse_rms=aux["pulse_rms"], gate_mean=aux["gate_mean"],
                 gate_min=aux.get("gate_min", float("nan")))
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
                    default="analysis/binary_gated_pulse_bench.json")
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
                gm = diag.get("gate_mean", float("nan"))
                print(f"[bench] {mode:14s} {ds:10s} s{seed} "
                      f"mse={out['test_mse']:.5f} gap={out['gap_mse']:.5f} "
                      f"gr={out['gap_ratio']:.2f} amp={amp:.3f} "
                      f"gate={gm:.3f} ({el:.1f}s)")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {outp}")

    def agg(key):
        s = {}
        for c in results["cells"]:
            s.setdefault((c["mode"], c["dataset"]), []).append(c[key])
        return s
    mse, gr, amp_d = agg("test_mse"), agg("gap_ratio"), agg("diagnostics")

    print("\n[bench] mean test_mse (clean) | gap_ratio | pulse_amp:")
    hdr = " | ".join(f"{d:>22s}" for d in args.datasets)
    print(f"{'mode':14s} | {hdr}")
    for mode in args.modes:
        cells = []
        for d in args.datasets:
            v = mse.get((mode, d), [])
            r = gr.get((mode, d), [])
            a = amp_d.get((mode, d), [])
            vm = sum(v) / len(v) if v else float("nan")
            rm = sum(r) / len(r) if r else float("nan")
            am = (sum(x.get("pulse_amp_mean", float("nan")) for x in a)
                  / len(a)) if a else float("nan")
            cells.append(f"{vm:.5f}/{rm:.2f}/{am:.3f}")
        print(f"{mode:14s} | " + " | ".join(f"{c:>22s}" for c in cells))

    print("\n[bench] Δ%% vs blend_gated (negative=better) on clean mse:")
    for d in args.datasets:
        base = mse.get(("blend_gated", d), [])
        if not base:
            continue
        bm = sum(base) / len(base)
        line = f"  {d:11s}: blend={bm:.5f}"
        for m in ("pulse_sin", "binary_tau_03", "binary_tau_05", "binary_tau_07"):
            v = mse.get((m, d), [])
            if v:
                vm = sum(v) / len(v)
                line += f"  {m}={vm:.5f} ({100*(vm-bm)/max(abs(bm),1e-12):+.1f}%)"
        print(line)

    # Hypothesis check.
    print("\n[bench] H1/H2/H3 acceptance check (r287 → strict-positive?):")

    def mean(key, mode, ds):
        vals = mse.get((mode, ds), []) if key == "mse" else (
            gr.get((mode, ds), []) if key == "gr" else
            [x.get("pulse_amp_mean", float("nan"))
             for x in amp_d.get((mode, ds), [])])
        return sum(vals) / len(vals) if vals else float("nan")

    r284_gr = mean("gr", "pulse_sin", "structured")
    print("  H1 structured gap_ratio ≤ r284=61:")
    h1_ok = False
    for tau in (0.3, 0.5, 0.7):
        v = mean("gr", f"binary_tau_{int(tau*10):02d}", "structured")
        ok = v <= r284_gr
        print(f"     τ={tau}: r287 gap_ratio={v:.2f}  {'OK' if ok else 'FAIL'}")
        if ok:
            h1_ok = True
    if not h1_ok:
        print("     (no τ satisfies H1)")

    base_rnd = mean("mse", "blend_gated", "random")
    print("  H2 random Δ%% ≤ +5%:")
    h2_ok = False
    for tau in (0.3, 0.5, 0.7):
        v = mean("mse", f"binary_tau_{int(tau*10):02d}", "random")
        delta = 100 * (v - base_rnd) / max(abs(base_rnd), 1e-12)
        ok = abs(delta) <= 5.0
        print(f"     τ={tau}: r287 Δ%%={delta:+.1f}%  {'OK' if ok else 'FAIL'}")
        if ok:
            h2_ok = True

    print("  H3 random pulse_amp ≤ 0.20:")
    h3_ok = False
    for tau in (0.3, 0.5, 0.7):
        v = mean("amp", f"binary_tau_{int(tau*10):02d}", "random")
        ok = v <= 0.20
        print(f"     τ={tau}: r287 pulse_amp={v:.3f}  {'OK' if ok else 'FAIL'}")
        if ok:
            h3_ok = True

    h4_ok = h1_ok and h2_ok and h3_ok
    print(f"  H4 ALL pass (strict-positive default): "
          f"{'YES — STRICT POSITIVE' if h4_ok else 'NO — TD / NEGATIVE'}")


if __name__ == "__main__":
    main()