#!/usr/bin/env python3
"""Benchmark for PredictabilityGatedPulseCfCCell (round 285).

Round 285 is the direct follow-up the r284 report itself recommended:
gate the r284 oscillatory pulse by the r280 predictability score
``g_t`` so the pulse is suppressed exactly when input is erratic
(restoring noise safety) but active on predictable / gappy data
(keeping the structured gap-robustness).

Hypotheses:
  H1 (robustness preserved): structured gap_ratio (gated_pulse) ≤
     blend (368) AND ≤ r284 (61).
  H2 (safety restored, THE FIX): random Δ% (gated_pulse vs blend)
     ≤ +5% (r284 was +44.6%).
  H3 (amplitude no longer chases noise): on random, final
     pulse_amp.abs().mean() ≤ 0.20 (r284 grew to 0.40).
  H4 (superset): gate_pulse=False ≡ r284 (unit-tested separately).
  H5 (gating not just clamping): on structured+gap, post-training
     gate.mean() ≥ 0.5.

Modes:
  * static_tau    — r267 production baseline (static per-neuron τ)
  * blend_gated   — r280 production (blend gate, no pulse)  [primary baseline]
  * pulse_sin     — r284 (blend gate + ungated sin pulse)   [direct comparator]
  * gated_pulse   — r285 NEW (blend gate + g_t-gated sin pulse)  [round under test]

Robustness (H1): each trained model is also evaluated on a gap-corrupted
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
from lnn.core.predictability_gated_pulse_cfc import (  # noqa: E402
    PredictabilityGatedPulseCfCCell,
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
    "blend_gated": dict(kind="blend"),
    "pulse_sin": dict(kind="pulse", pulse_mode="sin", gate_pulse=False),
    "gated_pulse": dict(kind="pulse", pulse_mode="sin", gate_pulse=True),
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
    else:  # pulse (with or without gate_pulse)
        if cfg["gate_pulse"]:
            cell = PredictabilityGatedPulseCfCCell(
                liquid_tau_strength=1.0, pred_gate_beta=4.0, ema_gamma=0.5,
                gate_mode="blend", pulse_strength=1.0, pulse_amp_init=0.1,
                pulse_mode=cfg["pulse_mode"], state_phase=True,
                gate_pulse=True, **_COMMON)
        else:
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
    if isinstance(cell, (PulseGatedLiquidTauCfCCell,
                          PredictabilityGatedPulseCfCCell)):
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
                    default="analysis/predictability_gated_pulse_bench.json")
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
                print(f"[bench] {mode:12s} {ds:10s} s{seed} "
                      f"mse={out['test_mse']:.5f} gap={out['gap_mse']:.5f} "
                      f"gr={out['gap_ratio']:.2f} amp={amp:.3f} "
                      f"gate={gm:.3f} ({el:.1f}s)")

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
    mse, gr, amp_d = agg("test_mse"), agg("gap_ratio"), agg("diagnostics")

    print("\n[bench] mean test_mse (clean) | gap_ratio | pulse_amp:")
    hdr = " | ".join(f"{d:>22s}" for d in args.datasets)
    print(f"{'mode':12s} | {hdr}")
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
        print(f"{mode:12s} | " + " | ".join(f"{c:>22s}" for c in cells))

    print("\n[bench] Δ%% vs blend_gated (negative=better) on clean mse:")
    for d in args.datasets:
        base = mse.get(("blend_gated", d), [])
        if not base:
            continue
        bm = sum(base) / len(base)
        line = f"  {d:11s}: blend={bm:.5f}"
        for m in ("pulse_sin", "gated_pulse"):
            v = mse.get((m, d), [])
            if v:
                vm = sum(v) / len(v)
                line += f"  {m}={vm:.5f} ({100*(vm-bm)/max(abs(bm),1e-12):+.1f}%)"
        print(line)

    # Hypothesis check (informational).
    print("\n[bench] H1/H2/H3/H5 hypothesis check (r285 acceptance):")
    base_struct_gr = sum(gr.get(("blend_gated", "structured"), [])) / max(
        len(gr.get(("blend_gated", "structured"), [])), 1)
    r284_struct_gr = sum(gr.get(("pulse_sin", "structured"), [])) / max(
        len(gr.get(("pulse_sin", "structured"), [])), 1)
    r285_struct_gr = sum(gr.get(("gated_pulse", "structured"), [])) / max(
        len(gr.get(("gated_pulse", "structured"), [])), 1)
    print(f"  H1 structured gap_ratio: blend={base_struct_gr:.2f} "
          f"r284={r284_struct_gr:.2f} r285={r285_struct_gr:.2f}  "
          f"{'OK' if r285_struct_gr <= r284_struct_gr else 'FAIL'}")

    base_rnd = sum(mse.get(("blend_gated", "random"), [])) / max(
        len(mse.get(("blend_gated", "random"), [])), 1)
    r285_rnd = sum(mse.get(("gated_pulse", "random"), [])) / max(
        len(mse.get(("gated_pulse", "random"), [])), 1)
    r284_rnd = sum(mse.get(("pulse_sin", "random"), [])) / max(
        len(mse.get(("pulse_sin", "random"), [])), 1)
    delta_pct = 100 * (r285_rnd - base_rnd) / max(abs(base_rnd), 1e-12)
    delta_r284_pct = 100 * (r284_rnd - base_rnd) / max(abs(base_rnd), 1e-12)
    print(f"  H2 random Δ%%: r284={delta_r284_pct:+.1f}%  r285={delta_pct:+.1f}%  "
          f"{'OK' if abs(delta_pct) <= 5.0 else 'FAIL'}")

    rnd_amps = [x.get("pulse_amp_mean", float("nan"))
                for x in amp_d.get(("gated_pulse", "random"), [])]
    rnd_amps_mean = (sum(rnd_amps) / len(rnd_amps)) if rnd_amps else float("nan")
    r284_rnd_amps = [x.get("pulse_amp_mean", float("nan"))
                     for x in amp_d.get(("pulse_sin", "random"), [])]
    r284_rnd_amps_mean = (
        sum(r284_rnd_amps) / len(r284_rnd_amps)) if r284_rnd_amps else float("nan")
    print(f"  H3 pulse_amp on random: r284={r284_rnd_amps_mean:.3f} "
          f"r285={rnd_amps_mean:.3f}  "
          f"{'OK' if rnd_amps_mean <= 0.20 else 'FAIL'}")

    struct_gates = [x.get("gate_mean", float("nan"))
                    for x in amp_d.get(("gated_pulse", "structured"), [])]
    struct_gates_mean = (
        sum(struct_gates) / len(struct_gates)) if struct_gates else float("nan")
    print(f"  H5 gate.mean() on structured: r285={struct_gates_mean:.3f}  "
          f"{'OK' if struct_gates_mean >= 0.5 else 'FAIL'}")


if __name__ == "__main__":
    main()