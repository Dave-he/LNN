#!/usr/bin/env python3
"""Benchmark: EMMA rover real-data regression.

Applies the three LNN multimodal variants (video-only concat baseline,
multimodal with concat fusion, cross-modal attention) to the EMMA rover
real video+audio features extracted in
``lnn/data/emma_rover_features.py``.

Targets are the 5 ground-truth parameters of the differential-drive
rover (a, b, r, m, CM) from EMMA paper Table 4(c).  Because every
sample is the same physical system, this benchmark measures how well
the model fits a noisy re-observation of one trajectory (low val
param MSE = good consistency), not parameter discrimination.

Usage:
    python scripts/benchmark_emma_rover.py --epochs 16 --num-samples 200
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
    MultimodalBiCfCNADWithMDN,
)
from lnn.core.noise_adaptive_cfc import BiCfCNADWithMDN, mdn_predicted_std
from lnn.data.emma_rover_regression import (
    EmmaRoverRegressionDataset,
    create_emma_rover_dataloaders,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _video_only_forward(model: BiCfCNADWithMDN, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Concat the noisy audio (1 ch) into the video (3 ch) stream so the
    baseline has access to the same raw numbers but no dedicated encoder
    branch.
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
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch, target in dataloader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        params = target["params"]
        optimizer.zero_grad(set_to_none=True)
        if use_video_only_branch:
            assert isinstance(model, BiCfCNADWithMDN)
            mdn_params = _video_only_forward(model, batch)
        else:
            assert isinstance(model, (MultimodalBiCfCNADWithMDN, CrossModalAttnBiCfCNADWithMDN)) or model.__class__.__name__ == "UniVideoSelfXAttnWithMDN"
            mdn_params = model(batch["video"], batch["audio"])
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
) -> dict[str, float]:
    model.eval()
    sq_errs: list[torch.Tensor] = []
    pred_stds: list[torch.Tensor] = []
    for batch, target in dataloader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        params = target["params"]
        if use_video_only_branch:
            assert isinstance(model, BiCfCNADWithMDN)
            mdn_params = _video_only_forward(model, batch)
        else:
            assert isinstance(model, (MultimodalBiCfCNADWithMDN, CrossModalAttnBiCfCNADWithMDN)) or model.__class__.__name__ == "UniVideoSelfXAttnWithMDN"
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
    video_dim: int,
    audio_dim: int,
    hidden_size: int,
    num_mixtures: int,
) -> nn.Module:
    if model_kind == "video_only":
        return BiCfCNADWithMDN(
            input_size=video_dim + audio_dim,
            hidden_size=hidden_size,
            output_size=5,
            num_mixtures=num_mixtures,
        )
    if model_kind == "multimodal":
        return MultimodalBiCfCNADWithMDN(
            video_dim=video_dim,
            audio_dim=audio_dim,
            hidden_size=hidden_size,
            output_size=5,
            num_mixtures=num_mixtures,
        )
    if model_kind == "cross_attn":
        return CrossModalAttnBiCfCNADWithMDN(
            video_dim=video_dim,
            audio_dim=audio_dim,
            hidden_size=hidden_size,
            output_size=5,
            num_mixtures=num_mixtures,
        )
    if model_kind == "uni_video_xattn":
        from lnn.core.multimodal_physreg import UniVideoSelfXAttnWithMDN
        return UniVideoSelfXAttnWithMDN(
            video_dim=video_dim,
            audio_dim=audio_dim,  # ignored at forward time
            hidden_size=hidden_size,
            output_size=5,
            num_mixtures=num_mixtures,
        )
    raise ValueError(f"unknown model_kind {model_kind!r}")


def _run(
    model_kind: str,
    args: argparse.Namespace,
    device: torch.device,
    dataset: EmmaRoverRegressionDataset,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    train_loader, val_loader, test_loader = create_emma_rover_dataloaders(
        dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = _build_model(
        model_kind,
        video_dim=3,
        audio_dim=1,
        hidden_size=args.hidden_size,
        num_mixtures=args.num_mixtures,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_video_only_branch = model_kind == "video_only"
    history = {"train_loss": [], "val_param_mse": []}
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, device, use_video_only_branch)
        val_metrics = _evaluate(model, val_loader, device, use_video_only_branch)
        history["train_loss"].append(train_loss)
        history["val_param_mse"].append(val_metrics["param_mse"])
        if epoch == 1 or epoch == args.epochs or epoch % max(args.epochs // 5, 1) == 0:
            print(
                f"[{model_kind:>10s}] epoch {epoch:3d}/{args.epochs} | "
                f"train NLL {train_loss:.4f} | val param MSE {val_metrics['param_mse']:.4f}"
            )
    elapsed = time.perf_counter() - start
    test_metrics = _evaluate(model, test_loader, device, use_video_only_branch)
    return {
        "model_kind": model_kind,
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": elapsed,
        "history": history,
        "test": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EMMA rover real-data regression benchmark")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--feature-noise-std", type=float, default=0.02)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-mixtures", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="analysis/emma_rover")
    parser.add_argument("--video-path", default="/tmp/RoverVideo.mp4")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=== EMMA Rover Real-Data Regression Benchmark ===")
    print(f"Device: {device} | samples: {args.num_samples} | window: {args.window} | noise_std: {args.feature_noise_std}")
    print(f"Video: {args.video_path}")

    dataset = EmmaRoverRegressionDataset(
        num_samples=args.num_samples,
        window=args.window,
        feature_noise_std=args.feature_noise_std,
        seed=args.seed,
        video_path=args.video_path,
    )

    video_only = _run("video_only", args, device, dataset)
    multimodal = _run("multimodal", args, device, dataset)
    cross_attn = _run("cross_attn", args, device, dataset)
    uni_video = _run("uni_video_xattn", args, device, dataset)

    v_mse = video_only["test"]["param_mse"]
    m_mse = multimodal["test"]["param_mse"]
    c_mse = cross_attn["test"]["param_mse"]
    u_mse = uni_video["test"]["param_mse"]
    claim_threshold = 0.20
    cross_attn_vs_videoonly = (v_mse - c_mse) / v_mse if v_mse > 0 else 0.0
    cross_attn_vs_multimodal = (m_mse - c_mse) / m_mse if m_mse > 0 else 0.0
    uni_video_vs_videoonly = (v_mse - u_mse) / v_mse if v_mse > 0 else 0.0
    uni_video_vs_crossattn = (c_mse - u_mse) / c_mse if c_mse > 0 else 0.0
    claim_passed = cross_attn_vs_videoonly >= claim_threshold

    print("\n=== Test Results (5-dim param MSE) ===")
    print(f"video_only      | params {video_only['parameters']:>5d} | test MSE {v_mse:.4f}")
    print(f"multimodal      | params {multimodal['parameters']:>5d} | test MSE {m_mse:.4f}")
    print(f"cross_attn      | params {cross_attn['parameters']:>5d} | test MSE {c_mse:.4f}")
    print(f"uni_video_xattn | params {uni_video['parameters']:>5d} | test MSE {u_mse:.4f}")
    print(
        f"cross_attn      vs video_only : {cross_attn_vs_videoonly * 100:+.1f}% "
        f"(claim >= {claim_threshold * 100:.0f}%) → "
        f"{'PASS' if claim_passed else 'FAIL'}"
    )
    print(f"cross_attn      vs multimodal : {cross_attn_vs_multimodal * 100:+.1f}%")
    print(
        f"uni_video_xattn vs video_only : {uni_video_vs_videoonly * 100:+.1f}% "
        "(architecture-only ablation: large gap below cross_attn ⇒ audio carries unique info)"
    )
    print(f"uni_video_xattn vs cross_attn : {uni_video_vs_crossattn * 100:+.1f}%")

    now = dt.datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}_emma_rover.json"
    payload = {
        "run_id": run_id,
        "run_date": now.date().isoformat(),
        "generated_at": now.isoformat(),
        "config": vars(args),
        "video_only": video_only,
        "multimodal": multimodal,
        "cross_attn": cross_attn,
        "uni_video_xattn": uni_video,
        "summary": {
            "video_only_param_mse": v_mse,
            "multimodal_param_mse": m_mse,
            "cross_attn_param_mse": c_mse,
            "uni_video_xattn_param_mse": u_mse,
            "cross_attn_vs_videoonly_rel": cross_attn_vs_videoonly,
            "cross_attn_vs_multimodal_rel": cross_attn_vs_multimodal,
            "uni_video_vs_videoonly_rel": uni_video_vs_videoonly,
            "uni_video_vs_crossattn_rel": uni_video_vs_crossattn,
            "claim_threshold": claim_threshold,
            "claim_passed": bool(claim_passed),
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults saved to: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
