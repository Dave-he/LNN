#!/usr/bin/env python3
"""Register-token minimal reproduction probe.

Hypothesis (falsifiable, §21.6 W+1 #1):
  If cross_attn(audio=zero)'s +14.9pp over uni_video_xattn comes from
  'stream2 being a free pool the encoder can specialise to a
  register-token-like representation', then a model whose *second
  encoder input is a learnable constant* (independent of the data)
  should reproduce the gain.

Test: on real EMMA rover, compare
  - video_only (baseline, no second encoder)
  - uni_video_xattn (second encoder fed video, +32.2pp over video_only)
  - cross_attn(audio=normal)  (full cross-attn with informative audio, +51.0pp)
  - cross_attn(audio=zero)    (the "free pool" condition, +47.1pp)
  - **register_token (NEW)**   (second encoder fed a learnable constant)

Falsifiable: register_token's gain over video_only should be
*comparable to cross_attn(audio=zero)* (i.e. reach at least +47%).
If it's only as good as uni_video_xattn (+32.2%) or worse, the
register-token hypothesis is rejected.
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
    LSTMEncoderXAttnWithMDN,
    NonRecurrentSelfXAttnWithMDN,
    RegisterTokenSelfXAttnWithMDN,
    UniVideoSelfXAttnWithMDN,
    VanillaCfCXAttnWithMDN,
)
from lnn.core.noise_adaptive_cfc import BiCfCNADWithMDN
from lnn.data.emma_rover_regression import (
    EmmaRoverRegressionDataset,
    create_emma_rover_dataloaders,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _move(b, d):
    return {k: v.to(d) for k, v in b.items()}


def _forward(model, kind, batch, dataset_audio_mode="normal"):
    if kind == "video_only":
        return model(torch.cat([batch["video"], batch["audio"]], dim=-1))
    if kind in ("register_token", "non_recurrent_xattn", "vanilla_cfc_xattn", "lstm_xattn"):
        return model(batch["video"])
    if kind == "uni_video_xattn":
        return model(batch["video"])
    if kind == "cross_attn":
        if dataset_audio_mode == "zero":
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
    if kind == "register_token":
        return RegisterTokenSelfXAttnWithMDN(video_dim=3, audio_dim=3, hidden_size=hidden_size, output_size=5, num_mixtures=num_mixtures)
    if kind == "non_recurrent_xattn":
        return NonRecurrentSelfXAttnWithMDN(video_dim=3, audio_dim=3, hidden_size=hidden_size, output_size=5, num_mixtures=num_mixtures)
    if kind == "vanilla_cfc_xattn":
        return VanillaCfCXAttnWithMDN(video_dim=3, audio_dim=3, hidden_size=hidden_size, output_size=5, num_mixtures=num_mixtures)
    if kind == "lstm_xattn":
        return LSTMEncoderXAttnWithMDN(video_dim=3, audio_dim=3, hidden_size=hidden_size, output_size=5, num_mixtures=num_mixtures)
    if kind == "cross_attn":
        return CrossModalAttnBiCfCNADWithMDN(video_dim=3, audio_dim=1, hidden_size=hidden_size, output_size=5, num_mixtures=num_mixtures)
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--feature-noise-std", type=float, default=0.02)
    p.add_argument("--hidden-size", type=int, default=16)
    p.add_argument("--num-mixtures", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="analysis/emma_rover")
    args = p.parse_args()
    device = torch.device("cpu")
    print("=== Register-Token Minimal Reproduction ===")
    print(f"epochs={args.epochs} n={args.num_samples} window={args.window} hidden={args.hidden_size}")

    dataset = EmmaRoverRegressionDataset(
        num_samples=args.num_samples, window=args.window,
        feature_noise_std=args.feature_noise_std, seed=args.seed,
    )
    train_loader, _, test_loader = create_emma_rover_dataloaders(
        dataset, batch_size=args.batch_size, seed=args.seed,
    )

    runs: list[dict[str, Any]] = []
    for kind, audio_mode in [
        ("video_only", "normal"),
        ("uni_video_xattn", "normal"),
        ("register_token", "normal"),
        ("non_recurrent_xattn", "normal"),
        ("vanilla_cfc_xattn", "normal"),
        ("lstm_xattn", "normal"),
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
        print(f"  {kind:18s} | audio={audio_mode:6s} | params={runs[-1]['parameters']:>5d} | test MSE = {test:.4f}")

    v_mse = next(r["test_param_mse"] for r in runs if r["model_kind"] == "video_only")
    print()
    print("=== Gain over video_only ===")
    for r in runs:
        gain = (v_mse - r["test_param_mse"]) / v_mse if v_mse > 0 else 0
        marker = ""
        if r["model_kind"] == "register_token":
            # The hypothesis: register_token gain should be close to
            # cross_attn(audio=zero) gain (=+14.9pp over uni_video, or
            # +47.1pp over video_only).
            if gain > 0.45:
                marker = "  <-- HYPOTHESIS PASS (>+45%)"
            elif gain > 0.32:
                marker = "  <-- PARTIAL (between +32% and +45%)"
            else:
                marker = "  <-- HYPOTHESIS FAIL (<+32%)"
        print(f"  {r['model_kind']:18s} | audio={r['audio_mode']:6s} | test MSE = {r['test_param_mse']:.4f} | gain = {gain * 100:+5.1f}%{marker}")
    now = dt.datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}_register_token.json"
    json_path.write_text(json.dumps({"run_id": run_id, "config": vars(args), "runs": runs}, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
