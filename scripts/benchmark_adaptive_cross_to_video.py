#!/usr/bin/env python3
"""Adaptive two-phase training for EMMA rover.

Round-24 follow-up to §27 transition curve: cross_attn is +36.5% at
mid-budget (h=32, ep=40) but −61.8% at (h=32, ep=80) because video_only
has caught up by ep=80. This script tests whether a two-phase training
schedule can keep the early cross_attn regularisation while avoiding the
late-stage optimisation burden:

  Phase 1 (epochs 1..K):
      Train CrossModalAttnBiCfCNADWithMDN normally on (video, audio).
  Phase 2 (epochs K+1..total):
      Drop the second encoder + cross-attention; transfer the warmed-up
      video_encoder state into a fresh BiCfCNADWithMDN(input_size=3,
      no audio concat) and continue training video-only.

Falsifiable claim: with the transition (h=32, ep=80) where pure
cross_attn fails at −61.8% (MSE 60.84) and pure video_only with audio
concat reaches MSE 37.59, the adaptive variant should achieve
**test MSE strictly less than both** at some K.

Usage:
    python scripts/benchmark_adaptive_cross_to_video.py --warmup 40
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


def _train_xattn_epoch(
    model: CrossModalAttnBiCfCNADWithMDN,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
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


def _train_vo_epoch(
    model: BiCfCNADWithMDN,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch, target in dataloader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        params = target["params"]
        optimizer.zero_grad(set_to_none=True)
        # Video-only: feed pure video tensor (input_size=3); no audio concat.
        out = model(batch["video"])
        final = {k: v[:, -1] for k, v in out.items()}
        loss = mdn_negative_log_likelihood(final, params)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def _eval_xattn(
    model: CrossModalAttnBiCfCNADWithMDN,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sq_errs: list[torch.Tensor] = []
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


@torch.no_grad()
def _eval_vo(
    model: BiCfCNADWithMDN,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sq_errs: list[torch.Tensor] = []
    for batch, target in dataloader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        params = target["params"]
        out = model(batch["video"])
        final = {k: v[:, -1] for k, v in out.items()}
        mean = mdn_mean(final)
        sq_errs.append((mean - params).pow(2).sum(dim=-1))
    sq = torch.cat(sq_errs)
    return {"param_mse": float(sq.mean().item())}


def _transfer_video_encoder(xattn: CrossModalAttnBiCfCNADWithMDN, vo: BiCfCNADWithMDN) -> None:
    """Copy the trained Bi-CfC-NAD weights from cross_attn's video encoder
    into the fresh video_only model's encoder.

    Note ``xattn.video_encoder`` is a ``_SingleStreamEncoder`` wrapping the
    actual :class:`BidirectionalNoiseAdaptiveCfC` at ``.encoder``; the
    video_only target uses :class:`BiCfCNADWithMDN` whose recurrent stack
    is exposed directly at ``.encoder``. So we load
    ``xattn.video_encoder.encoder.state_dict()`` -> ``vo.encoder``.
    """
    vo.encoder.load_state_dict(xattn.video_encoder.encoder.state_dict())


def _adaptive_run(
    args: argparse.Namespace,
    device: torch.device,
    dataset: EmmaRoverRegressionDataset,
    warmup_epochs: int,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    train_loader, val_loader, test_loader = create_emma_rover_dataloaders(
        dataset, batch_size=args.batch_size, seed=args.seed,
    )

    # Phase 1: train cross_attn for warmup_epochs.
    xattn = CrossModalAttnBiCfCNADWithMDN(
        video_dim=dataset.video_dim,
        audio_dim=1,
        hidden_size=args.hidden_size,
        output_size=5,
        num_mixtures=args.num_mixtures,
    ).to(device)
    optim1 = torch.optim.Adam(xattn.parameters(), lr=args.lr)
    history = {"phase": [], "train_loss": [], "val_mse": []}
    start = time.perf_counter()
    for epoch in range(1, warmup_epochs + 1):
        loss = _train_xattn_epoch(xattn, train_loader, optim1, device)
        val_mse = _eval_xattn(xattn, val_loader, device)["param_mse"]
        history["phase"].append("xattn")
        history["train_loss"].append(loss)
        history["val_mse"].append(val_mse)
        if epoch in (1, warmup_epochs) or epoch % max(args.epochs // 5, 1) == 0:
            print(f"[xattn-warmup ] epoch {epoch:3d}/{warmup_epochs} | "
                  f"train NLL {loss:.4f} | val MSE {val_mse:.4f}")

    # Phase 2: spin up fresh BiCfCNADWithMDN(input_size=3), transfer
    # video_encoder weights, train for remaining epochs.
    vo = BiCfCNADWithMDN(
        input_size=dataset.video_dim,
        hidden_size=args.hidden_size,
        output_size=5,
        num_mixtures=args.num_mixtures,
    ).to(device)
    _transfer_video_encoder(xattn, vo)
    optim2 = torch.optim.Adam(vo.parameters(), lr=args.lr)
    remaining = args.epochs - warmup_epochs
    for epoch in range(1, remaining + 1):
        loss = _train_vo_epoch(vo, train_loader, optim2, device)
        val_mse = _eval_vo(vo, val_loader, device)["param_mse"]
        history["phase"].append("vo")
        history["train_loss"].append(loss)
        history["val_mse"].append(val_mse)
        global_epoch = warmup_epochs + epoch
        if epoch == 1 or epoch == remaining or global_epoch % max(args.epochs // 5, 1) == 0:
            print(f"[vo-finetune  ] epoch {epoch:3d}/{remaining}    | "
                  f"train NLL {loss:.4f} | val MSE {val_mse:.4f}")
    elapsed = time.perf_counter() - start
    test_metrics = _eval_vo(vo, test_loader, device)
    return {
        "warmup_epochs": warmup_epochs,
        "total_epochs": args.epochs,
        "elapsed_seconds": elapsed,
        "history": history,
        "test": test_metrics,
        "xattn_params": sum(p.numel() for p in xattn.parameters()),
        "final_vo_params": sum(p.numel() for p in vo.parameters()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive cross→video two-phase trainer (round 24)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup-epochs", type=int, default=40,
                        help="Epochs to train cross_attn before switching to video_only.")
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

    print(f"=== Adaptive cross→video benchmark (h={args.hidden_size}, ep={args.epochs}, "
          f"warmup={args.warmup_epochs}) ===")
    result = _adaptive_run(args, device, dataset, args.warmup_epochs)
    print(f"\nFinal test MSE: {result['test']['param_mse']:.4f}")

    out_dir = ROOT / "analysis/emma_rover"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now()
    json_path = out_dir / f"{now.strftime('%Y-%m-%d_%H%M%S')}_adaptive_warmup{args.warmup_epochs}.json"
    json_path.write_text(json.dumps({
        "config": vars(args),
        "result": result,
    }, indent=2) + "\n")
    print(f"\nResults saved to: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
