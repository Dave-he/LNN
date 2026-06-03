#!/usr/bin/env python3
"""Adaptive freeze-after training for EMMA rover (round 25).

Round-24 §28 NEGATIVE result diagnosed the fresh MDN head as the
catastrophe driver of the encoder-transfer adaptive strategy. This
script tests a cleaner alternative: keep the same CrossModalAttnBiCfCNADWithMDN
model throughout, run normal training for K epochs, then FREEZE the
audio-side path (audio_encoder and optionally the cross-attn audio
projections) and continue training. Optimizer state is preserved; head
is never re-initialised.

Falsifiable claim at (h=32, ep=80, K=40): the freeze variant should
beat both pure cross_attn (60.84) and pure video_only(input=4) (37.59)
— i.e. test MSE strictly < 37.59.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lnn.core.mdn import mdn_mean, mdn_negative_log_likelihood
from lnn.core.multimodal_physreg import CrossModalAttnBiCfCNADWithMDN
from lnn.data.emma_rover_regression import (
    EmmaRoverRegressionDataset,
    create_emma_rover_dataloaders,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    batches = 0
    for batch, target in dataloader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        params = target["params"]
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in out.items()}
        loss = mdn_negative_log_likelihood(final, params)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def _eval(model, dataloader, device):
    model.eval()
    sq_errs = []
    for batch, target in dataloader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        params = target["params"]
        out = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in out.items()}
        mean = mdn_mean(final)
        sq_errs.append((mean - params).pow(2).sum(dim=-1))
    sq = torch.cat(sq_errs)
    return {"param_mse": float(sq.mean().item())}


def _freeze_audio_path(model, targets):
    if targets not in {"audio_only", "all_xattn"}:
        raise ValueError(f"freeze targets must be 'audio_only' or 'all_xattn', got {targets!r}")
    frozen = 0
    for p in model.audio_encoder.parameters():
        p.requires_grad = False
        frozen += 1
    if targets == "all_xattn":
        for proj in (model.q_v, model.k_a, model.v_a, model.q_a, model.k_v, model.v_v, model.fuse_proj):
            for p in proj.parameters():
                p.requires_grad = False
                frozen += 1
    return frozen


def _adaptive_freeze_run(args, device, dataset):
    torch.manual_seed(args.seed)
    train_loader, val_loader, test_loader = create_emma_rover_dataloaders(
        dataset, batch_size=args.batch_size, seed=args.seed,
    )
    model = CrossModalAttnBiCfCNADWithMDN(
        video_dim=dataset.video_dim,
        audio_dim=1,
        hidden_size=args.hidden_size,
        output_size=5,
        num_mixtures=args.num_mixtures,
    ).to(device)
    history = {"phase": [], "train_loss": [], "val_mse": []}
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.perf_counter()
    for epoch in range(1, args.warmup_epochs + 1):
        loss = _train_epoch(model, train_loader, optim, device)
        val_mse = _eval(model, val_loader, device)["param_mse"]
        history["phase"].append("warmup")
        history["train_loss"].append(loss)
        history["val_mse"].append(val_mse)
        if epoch in (1, args.warmup_epochs) or epoch % max(args.epochs // 5, 1) == 0:
            print(f"[warmup       ] epoch {epoch:3d}/{args.warmup_epochs}    | "
                  f"train NLL {loss:.4f} | val MSE {val_mse:.4f}")
    frozen_count = _freeze_audio_path(model, args.freeze_targets)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"[freeze       ] {frozen_count} tensors frozen ({args.freeze_targets}); "
          f"{trainable_count}/{total_count} params still trainable")
    optim = torch.optim.Adam(trainable_params, lr=args.lr)
    remaining = args.epochs - args.warmup_epochs
    for epoch in range(1, remaining + 1):
        loss = _train_epoch(model, train_loader, optim, device)
        val_mse = _eval(model, val_loader, device)["param_mse"]
        history["phase"].append("frozen-audio")
        history["train_loss"].append(loss)
        history["val_mse"].append(val_mse)
        global_epoch = args.warmup_epochs + epoch
        if epoch == 1 or epoch == remaining or global_epoch % max(args.epochs // 5, 1) == 0:
            print(f"[frozen-audio ] epoch {epoch:3d}/{remaining}    | "
                  f"train NLL {loss:.4f} | val MSE {val_mse:.4f}")
    elapsed = time.perf_counter() - start
    test_metrics = _eval(model, test_loader, device)
    return {
        "warmup_epochs": args.warmup_epochs,
        "total_epochs": args.epochs,
        "freeze_targets": args.freeze_targets,
        "frozen_tensors": frozen_count,
        "trainable_params": trainable_count,
        "total_params": total_count,
        "elapsed_seconds": elapsed,
        "history": history,
        "test": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive freeze-after-warmup trainer for EMMA rover (round 25)"
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup-epochs", type=int, default=40)
    parser.add_argument("--freeze-targets", choices=["audio_only", "all_xattn"],
                        default="audio_only")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--feature-noise-std", type=float, default=0.02)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-mixtures", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--video-path", default="/tmp/RoverVideo.mp4")
    parser.add_argument("--video-channels", default="0,1,2")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = EmmaRoverRegressionDataset(
        num_samples=args.num_samples,
        window=args.window,
        feature_noise_std=args.feature_noise_std,
        seed=args.seed,
        video_path=args.video_path,
        video_channels=tuple(int(c) for c in args.video_channels.split(",")),
        audio_mode="normal",
    )
    print(
        f"=== Adaptive freeze benchmark (h={args.hidden_size}, ep={args.epochs}, "
        f"warmup={args.warmup_epochs}, targets={args.freeze_targets}) ==="
    )
    result = _adaptive_freeze_run(args, device, dataset)
    print(f"\nFinal test MSE: {result['test']['param_mse']:.4f}")
    out_dir = ROOT / "analysis/emma_rover"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now()
    json_path = out_dir / (
        f"2026-06-03_freeze_h{args.hidden_size}_{args.freeze_targets}_K{args.warmup_epochs}.json"
    )
    json_path.write_text(json.dumps({
        "config": vars(args),
        "generated_at": now.isoformat(),
        "result": result,
    }, indent=2) + "\n")
    print(f"\nResults saved to: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
