#!/usr/bin/env python3
"""Large-budget cross_attn sweep on real EMMA rover.

The §26.4 finding (parallel session): video_only at hidden=64,
ep=80 reaches test MSE 19.88 - *26.4x better* than the
small-budget (hidden=16, ep=20) baseline of 525.19. The +51% gain
of cross_attn that the previous 22 rounds documented was therefore
a *small-budget regularisation phenomenon*, not a fundamental
information-fusion advantage.

This script is the critical follow-up: does cross_attn also drop
to ~20 MSE at the same large budget, or does it stay stuck at
~250 (i.e. is the *real* contribution of cross-attention more
than a regulariser)?

4 runs at hidden=64, ep=80, n=200, K=1, seed=42:
  - video_only (already 19.88 - reproduced here as control)
  - uni_video_xattn
  - cross_attn(audio=normal)
  - cross_attn(audio=zero)

Hypothesis (falsifiable):
  * If the +51% gain is "small-budget regularisation only",
    cross_attn should drop to roughly the same 20-30 MSE as
    video_only at the same large budget.
  * If cross_attn provides *real* information fusion, it should
    be substantially *better* than video_only at the same budget
    (e.g. test MSE 5-15).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys
import time
from typing import Any

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.mdn import mdn_mean, mdn_negative_log_likelihood
from lnn.core.multimodal_physreg import (
    CrossModalAttnBiCfCNADWithMDN,
    UniVideoSelfXAttnWithMDN,
)
from lnn.core.noise_adaptive_cfc import BiCfCNADWithMDN
from lnn.data.emma_rover_regression import (
    EmmaRoverRegressionDataset,
    create_emma_rover_dataloaders,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _move(b, d):
    return {k: v.to(d) for k, v in b.items()}


def _forward(model, kind, batch, audio_mode="normal"):
    if kind == "video_only":
        return model(torch.cat([batch["video"], batch["audio"]], dim=-1))
    if kind == "uni_video_xattn":
        return model(batch["video"])
    if kind == "cross_attn":
        if audio_mode == "zero":
            return model(batch["video"], torch.zeros_like(batch["audio"]))
        return model(batch["video"], batch["audio"])
    raise ValueError(kind)


def _train(model, kind, loader, opt, device, audio_mode):
    model.train()
    losses = []
    for batch, target in loader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        opt.zero_grad(set_to_none=True)
        out = _forward(model, kind, batch, audio_mode)
        final = {k: v[:, -1] for k, v in out.items()}
        loss = mdn_negative_log_likelihood(final, target["params"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def _eval(model, kind, loader, device, audio_mode):
    model.eval()
    sq = []
    for batch, target in loader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        out = _forward(model, kind, batch, audio_mode)
        final = {k: v[:, -1] for k, v in out.items()}
        mean = mdn_mean(final)
        sq.append((mean - target["params"]).pow(2).sum(dim=-1))
    sq = torch.cat(sq)
    return float(sq.mean().item())


def _build_model(kind, hidden_size, num_mixtures):
    if kind == "video_only":
        return BiCfCNADWithMDN(input_size=4, hidden_size=hidden_size, output_size=5, num_mixtures=num_mixtures)
    if kind == "uni_video_xattn":
        return UniVideoSelfXAttnWithMDN(video_dim=3, audio_dim=3, hidden_size=hidden_size, output_size=5, num_mixtures=num_mixtures)
    if kind == "cross_attn":
        return CrossModalAttnBiCfCNADWithMDN(video_dim=3, audio_dim=1, hidden_size=hidden_size, output_size=5, num_mixtures=num_mixtures)
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--feature-noise-std", type=float, default=0.02)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-mixtures", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="analysis/emma_rover")
    args = p.parse_args()
    device = torch.device("cpu")
    print(f"=== Large-Budget Cross-Attn Sweep on Real EMMA Rover ===")
    print(f"hidden_size={args.hidden_size} | epochs={args.epochs} | n={args.num_samples}")
    print(f"Per-run estimated: ~{4 * args.epochs * args.num_samples / args.batch_size} gradient steps")

    dataset = EmmaRoverRegressionDataset(
        num_samples=args.num_samples, window=args.window,
        feature_noise_std=args.feature_noise_std, seed=args.seed,
    )
    train_loader, _, test_loader = create_emma_rover_dataloaders(
        dataset, batch_size=args.batch_size, seed=args.seed,
    )

    runs = []
    for kind, audio_mode in [
        ("video_only", "normal"),
        ("uni_video_xattn", "normal"),
        ("cross_attn", "normal"),
        ("cross_attn", "zero"),
    ]:
        torch.manual_seed(args.seed)
        model = _build_model(kind, args.hidden_size, args.num_mixtures).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        start = time.perf_counter()
        for _ in range(args.epochs):
            _train(model, kind, train_loader, opt, device, audio_mode)
        elapsed = time.perf_counter() - start
        test = _eval(model, kind, test_loader, device, audio_mode)
        runs.append({
            "model_kind": kind, "audio_mode": audio_mode,
            "parameters": sum(p.numel() for p in model.parameters()),
            "elapsed_seconds": elapsed, "test_param_mse": test,
        })
        print(f"  {kind:18s} | audio={audio_mode:6s} | params={runs[-1]['parameters']:>6d} | test MSE = {test:.4f}")

    v_mse = next(r["test_param_mse"] for r in runs if r["model_kind"] == "video_only")
    print()
    print("=== Test param MSE at large budget (hidden=64, ep=80) ===")
    for r in runs:
        gain = (v_mse - r["test_param_mse"]) / v_mse if v_mse > 0 else 0
        marker = ""
        if r["model_kind"].startswith("cross_attn") and r["audio_mode"] == "normal":
            if gain > 0.20:
                marker = "  <-- REAL-FUSION VERIFIED (gain > 20% at large budget)"
            elif gain > 0.05:
                marker = "  <-- PARTIAL (gain 5-20%, fusion small)"
            else:
                marker = "  <-- SMALL-BUDGET REGULARISATION FALSIFIED (gain < 5% at large budget)"
        print(f"  {r['model_kind']:18s} | audio={r['audio_mode']:6s} | test MSE = {r['test_param_mse']:.4f} | gain = {gain * 100:+5.1f}%{marker}")

    # Cross-check: small-budget reference values
    print()
    print("=== Cross-check: small-budget (hidden=16, ep=20) reference ===")
    print(f"  video_only              | test MSE = 525.19 | (baseline)")
    print(f"  cross_attn(audio=normal) | test MSE = 260.80 | gain = +50.3%")
    print(f"  cross_attn(audio=zero)   | test MSE = 248.64 | gain = +52.7%")
    print()
    print("If large-budget cross_attn drops to roughly the same MSE as")
    print("video_only (~20), the +51% small-budget gain is a regularisation")
    print("phenomenon, NOT information fusion.")

    now = dt.datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}_large_budget_sweep.json"
    json_path.write_text(
        json.dumps({"run_id": run_id, "config": vars(args), "runs": runs}, indent=2),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
