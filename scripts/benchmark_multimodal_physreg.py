#!/usr/bin/env python3
"""Benchmark: multimodal (video+audio) vs video-only Bi-CfC-NAD for physics
parameter regression on a synthetic damped-harmonic-oscillator task.

EMMA-inspired hypothesis: a two-stream model that ingests both a noisy
position trajectory ("video") and a noisy instantaneous frequency
("audio") should recover the oscillator's spring constant k and damping
c with substantially lower MSE than a video-only model of matched
hidden width.

Usage
-----
    python scripts/benchmark_multimodal_physreg.py --epochs 16
    python scripts/benchmark_multimodal_physreg.py --epochs 16 --num-samples 600 --fusion mean
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
from lnn.core.multimodal_physreg import MultimodalBiCfCNADWithMDN
from lnn.core.noise_adaptive_cfc import BiCfCNADWithMDN, mdn_predicted_std
from lnn.data.multimodal_physreg import (
    MultimodalPhysicsDataset,
    create_multimodal_physics_dataloaders,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _apply_occlusion(
    batch: dict[str, torch.Tensor],
    seq_len: int,
    video_mask_second_half: bool,
    audio_mask_first_half: bool,
) -> dict[str, torch.Tensor]:
    """Zero out the chosen half of each modality to simulate occlusion.

    The split is at ``seq_len // 2`` to keep the boundary stable across runs.
    Mirrors the EMMA rover setting where wheel-pose (video) is visible at the
    start of the clip but the motor-tone (audio) carries the hidden actuation
    information the video cannot see.
    """
    if not (video_mask_second_half or audio_mask_first_half):
        return batch
    out = dict(batch)
    half = seq_len // 2
    if video_mask_second_half:
        v = out["video"].clone()
        v[:, half:, :] = 0.0
        out["video"] = v
    if audio_mask_first_half:
        a = out["audio"].clone()
        a[:, :half, :] = 0.0
        out["audio"] = a
    return out


def _video_only_forward(model: BiCfCNADWithMDN, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A single-modality forward used as the baseline.

    Concatenates the noisy audio frequency into the video stream so the
    baseline has access to the same raw numbers but no dedicated encoder
    branch.  This is the "matched-capacity" control.
    """
    video = batch["video"]
    audio = batch["audio"]
    fused = torch.cat([video, audio], dim=-1)
    return model(fused)


def _train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_video_only_branch: bool,
    seq_len: int,
    video_mask_second_half: bool,
    audio_mask_first_half: bool,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch, target in dataloader:
        batch = _apply_occlusion(batch, seq_len, video_mask_second_half, audio_mask_first_half)
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        params = target["params"]
        optimizer.zero_grad(set_to_none=True)
        if use_video_only_branch:
            assert isinstance(model, BiCfCNADWithMDN)
            mdn_params = _video_only_forward(model, batch)
        else:
            assert isinstance(model, MultimodalBiCfCNADWithMDN)
            mdn_params = model(batch["video"], batch["audio"])
        # Train against the final-step parameter (sequence-to-sequence → final).
        final = {k: v[:, -1] for k, v in mdn_params.items()}
        loss = mdn_negative_log_likelihood(final, params)
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
    seq_len: int,
    video_mask_second_half: bool,
    audio_mask_first_half: bool,
) -> dict[str, float]:
    model.eval()
    sq_errs: list[torch.Tensor] = []
    pred_stds: list[torch.Tensor] = []
    for batch, target in dataloader:
        batch = _apply_occlusion(batch, seq_len, video_mask_second_half, audio_mask_first_half)
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        params = target["params"]
        if use_video_only_branch:
            assert isinstance(model, BiCfCNADWithMDN)
            mdn_params = _video_only_forward(model, batch)
        else:
            assert isinstance(model, MultimodalBiCfCNADWithMDN)
            mdn_params = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in mdn_params.items()}
        mean = mdn_mean(final)
        std = mdn_predicted_std(final)
        sq_errs.append((mean - params).pow(2).sum(dim=-1))
        pred_stds.append(std)
    sq = torch.cat(sq_errs)
    stds = torch.cat(pred_stds)
    return {
        "param_mse": float(sq.mean().item()),
        "param_mae": float(sq.sqrt().mean().item()),
        "avg_predicted_std": float(stds.mean().item()),
    }


def _build_model(
    model_kind: str,
    hidden_size: int,
    num_mixtures: int,
    fusion: str,
) -> nn.Module:
    if model_kind == "multimodal":
        return MultimodalBiCfCNADWithMDN(
            video_dim=1,
            audio_dim=1,
            hidden_size=hidden_size,
            output_size=2,
            num_mixtures=num_mixtures,
            fusion=fusion,
        )
    if model_kind == "video_only":
        return BiCfCNADWithMDN(
            input_size=2,  # video + audio concatenated as a single channel
            hidden_size=hidden_size,
            output_size=2,
            num_mixtures=num_mixtures,
        )
    raise ValueError(f"unknown model_kind {model_kind!r}")


def _run(
    model_kind: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    dataset = MultimodalPhysicsDataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        dt=args.dt,
        video_noise_std=args.video_noise_std,
        audio_noise_std=args.audio_noise_std,
        seed=args.seed,
    )
    train_loader, val_loader, test_loader = create_multimodal_physics_dataloaders(
        dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = _build_model(model_kind, args.hidden_size, args.num_mixtures, args.fusion).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_video_only_branch = model_kind == "video_only"
    history = {"train_loss": [], "val_param_mse": []}
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, device, use_video_only_branch,
            args.seq_len, args.video_mask_second_half, args.audio_mask_first_half,
        )
        val_metrics = _evaluate(
            model, val_loader, device, use_video_only_branch,
            args.seq_len, args.video_mask_second_half, args.audio_mask_first_half,
        )
        history["train_loss"].append(train_loss)
        history["val_param_mse"].append(val_metrics["param_mse"])
        if epoch == 1 or epoch == args.epochs or epoch % max(args.epochs // 5, 1) == 0:
            print(
                f"[{model_kind:>10s}] epoch {epoch:3d}/{args.epochs} | "
                f"train NLL {train_loss:.4f} | val param MSE {val_metrics['param_mse']:.6f}"
            )
    elapsed = time.perf_counter() - start
    test_metrics = _evaluate(
        model, test_loader, device, use_video_only_branch,
        args.seq_len, args.video_mask_second_half, args.audio_mask_first_half,
    )
    return {
        "model_kind": model_kind,
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": elapsed,
        "history": history,
        "test": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal vs video-only Bi-CfC-NAD physics regression benchmark")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--video-noise-std", type=float, default=0.05)
    parser.add_argument("--audio-noise-std", type=float, default=0.05)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-mixtures", type=int, default=1)
    parser.add_argument("--fusion", choices=["concat", "mean"], default="concat")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-mask-second-half", action="store_true",
                        help="If set, zero out the second half of the video stream "
                             "to simulate occlusion (audio must fill the gap).")
    parser.add_argument("--audio-mask-first-half", action="store_true",
                        help="If set, zero out the first half of the audio stream "
                             "(only meaningful with --video-mask-second-half).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="analysis/multimodal_physreg")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=== Multimodal Physics Parameter Regression Benchmark ===")
    print(f"Device: {device} | seq_len: {args.seq_len} | samples: {args.num_samples}")
    print(f"Hidden: {args.hidden_size} | mixtures: {args.num_mixtures} | fusion: {args.fusion}")

    video_only = _run("video_only", args, device)
    multimodal = _run("multimodal", args, device)

    v_mse = video_only["test"]["param_mse"]
    m_mse = multimodal["test"]["param_mse"]
    relative_improvement = (v_mse - m_mse) / v_mse if v_mse > 0 else 0.0
    claim_threshold = 0.20
    claim_passed = relative_improvement >= claim_threshold

    print("\n=== Test Results ===")
    print(f"video_only  | params {video_only['parameters']:>5d} | val MSE {v_mse:.6f}")
    print(f"multimodal  | params {multimodal['parameters']:>5d} | val MSE {m_mse:.6f}")
    print(
        f"multimodal improvement over video-only: {relative_improvement * 100:.1f}% "
        f"(claim threshold >= {claim_threshold * 100:.0f}%) → "
        f"{'PASS' if claim_passed else 'FAIL'}"
    )

    now = dt.datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}_multimodal_physreg.json"
    payload = {
        "run_id": run_id,
        "run_date": now.date().isoformat(),
        "generated_at": now.isoformat(),
        "config": vars(args),
        "video_only": video_only,
        "multimodal": multimodal,
        "summary": {
            "video_only_param_mse": v_mse,
            "multimodal_param_mse": m_mse,
            "relative_improvement": relative_improvement,
            "claim_threshold": claim_threshold,
            "claim_passed": bool(claim_passed),
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults saved to: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
