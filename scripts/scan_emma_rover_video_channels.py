#!/usr/bin/env python3
"""Scan video-channel subsets on EMMA rover real data.

The W+1 plan from §17.5 suggested a "architecture-vs-information
continuous probe": vary the video information content and see how
the cross_attn vs video_only gain responds.  This script implements
the cheap version using the ``video_channels`` knob added to
``EmmaRoverRegressionDataset``: for each subset of {0, 1, 2}
(motion_magnitude, centroid_x, centroid_y), run both video_only and
cross_attn and report the gain.

Hypothesis (falsifiable):
  If cross_attn's gain comes from a *separate physical-prior
  encoder*, the gain should be roughly invariant to *how much* video
  information is fed in (the audio encoder's contribution stays the
  same; the video encoder has less to work with, but the gap is
  buffered by the audio side).
  Conversely, if the gain is from per-modality fine-grained fusion,
  removing video information should reduce the gain proportionally.

Output JSON has all 8 runs (4 channel sets × 2 architectures) plus
the gain curve.  ``--quick`` skips the full 20-epoch run and just
prints 8-epoch numbers for fast iteration.
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
from lnn.core.multimodal_physreg import CrossModalAttnBiCfCNADWithMDN
from lnn.core.noise_adaptive_cfc import BiCfCNADWithMDN, mdn_predicted_std
from lnn.data.emma_rover_regression import (
    EmmaRoverRegressionDataset,
    create_emma_rover_dataloaders,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _video_only_forward(model: BiCfCNADWithMDN, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    video = batch["video"]
    audio = batch["audio"]
    fused = torch.cat([video, audio], dim=-1)
    return model(fused)


def _train(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_video_only_branch: bool,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch, target in dataloader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        optimizer.zero_grad(set_to_none=True)
        if use_video_only_branch:
            assert isinstance(model, BiCfCNADWithMDN)
            mdn_params = _video_only_forward(model, batch)
        else:
            assert isinstance(model, CrossModalAttnBiCfCNADWithMDN)
            mdn_params = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in mdn_params.items()}
        loss = mdn_negative_log_likelihood(final, target["params"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_video_only_branch: bool,
) -> dict[str, float]:
    model.eval()
    sq_errs: list[torch.Tensor] = []
    for batch, target in dataloader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        if use_video_only_branch:
            assert isinstance(model, BiCfCNADWithMDN)
            mdn_params = _video_only_forward(model, batch)
        else:
            assert isinstance(model, CrossModalAttnBiCfCNADWithMDN)
            mdn_params = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in mdn_params.items()}
        mean = mdn_mean(final)
        sq_errs.append((mean - target["params"]).pow(2).sum(dim=-1))
    sq = torch.cat(sq_errs)
    return {"param_mse": float(sq.mean().item())}


def _run_one(
    model_kind: str,
    video_channels: tuple[int, ...] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    dataset = EmmaRoverRegressionDataset(
        num_samples=args.num_samples,
        window=args.window,
        feature_noise_std=args.feature_noise_std,
        seed=args.seed,
        video_channels=video_channels,
    )
    train_loader, _, test_loader = create_emma_rover_dataloaders(
        dataset, batch_size=args.batch_size, seed=args.seed,
    )
    video_dim = dataset.video_dim
    if model_kind == "video_only":
        model = BiCfCNADWithMDN(
            input_size=video_dim + 1,
            hidden_size=args.hidden_size,
            output_size=5,
            num_mixtures=args.num_mixtures,
        ).to(device)
    elif model_kind == "cross_attn":
        model = CrossModalAttnBiCfCNADWithMDN(
            video_dim=video_dim, audio_dim=1,
            hidden_size=args.hidden_size, output_size=5, num_mixtures=args.num_mixtures,
        ).to(device)
    else:
        raise ValueError(model_kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_video_only = model_kind == "video_only"
    start = time.perf_counter()
    for _ in range(args.epochs):
        _train(model, train_loader, optimizer, device, use_video_only)
    elapsed = time.perf_counter() - start
    test_metrics = _evaluate(model, test_loader, device, use_video_only)
    return {
        "model_kind": model_kind,
        "video_channels": list(video_channels) if video_channels is not None else [0, 1, 2],
        "video_dim": video_dim,
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": elapsed,
        "test": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan video-channel subsets on EMMA rover")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--feature-noise-std", type=float, default=0.02)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-mixtures", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="analysis/emma_rover")
    args = parser.parse_args()

    device = torch.device("cpu")
    print("=== EMMA Rover Video-Channel Scan ===")
    print(f"epochs={args.epochs} samples={args.num_samples} window={args.window} noise={args.feature_noise_std}")

    channel_sets: list[tuple[int, ...] | None] = [None, (0,), (1,), (2,)]
    results: list[dict[str, Any]] = []
    for ch in channel_sets:
        label = "all (0,1,2)" if ch is None else f"{ch}"
        for kind in ("video_only", "cross_attn"):
            print(f"  running {kind:11s} | video_channels={label} ...", end="", flush=True)
            r = _run_one(kind, ch, args, device)
            print(f" test MSE = {r['test']['param_mse']:.4f}")
            results.append(r)

    # Compute the gain curve
    by_setting: dict[str, dict[str, float]] = {}
    for r in results:
        ch_key = tuple(r["video_channels"])
        by_setting.setdefault(str(ch_key), {})[r["model_kind"]] = r["test"]["param_mse"]

    print("\n=== Summary (test param MSE) ===")
    print(f"{'channels':15s} | {'video_only':>11s} | {'cross_attn':>11s} | {'gain':>8s}")
    gain_curve: list[dict[str, Any]] = []
    for ch_key in ("(0, 1, 2)", "[0]", "[1]", "[2]"):
        if ch_key not in by_setting:
            continue
        v = by_setting[ch_key]["video_only"]
        c = by_setting[ch_key]["cross_attn"]
        gain = (v - c) / v if v > 0 else 0.0
        print(f"{ch_key:15s} | {v:11.4f} | {c:11.4f} | {gain * 100:+7.1f}%")
        gain_curve.append({
            "video_channels": ch_key,
            "video_only_mse": v,
            "cross_attn_mse": c,
            "gain": gain,
        })

    # Headline: how much does the gain vary across channel subsets?
    if len(gain_curve) >= 2:
        gains = [g["gain"] for g in gain_curve]
        gain_range = max(gains) - min(gains)
        gain_mean = sum(gains) / len(gains)
        gain_std = (sum((g - gain_mean) ** 2 for g in gains) / len(gains)) ** 0.5
        print(f"\nGain range across channel subsets: {gain_range * 100:.1f}pp")
        print(f"Gain mean ± std: {gain_mean * 100:+.1f}% ± {gain_std * 100:.1f}pp")
        # Falsifiable check: if gain is invariant to video info content,
        # range should be < 10pp.  If it's information-dependent, range > 30pp.
        architecture_centric = gain_range < 0.10
        print(
            f"=> Architecture-centric verdict: "
            f"{'YES (range < 10pp)' if architecture_centric else 'NO (range >= 10pp)'}"
        )

    now = dt.datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}_video_channel_scan.json"
    payload = {
        "run_id": run_id,
        "config": vars(args),
        "results": results,
        "gain_curve": gain_curve,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults saved to: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
