#!/usr/bin/env python3
"""Scan hidden_size on the EMMA rover real data.

W+1 from §19.6: now that we know cross_attn's gain is mostly
*regularization-from-having-a-second-encoder-pathway* (not audio
content), the next question is whether the gain *scales with
capacity* (hidden_size) or is roughly capacity-invariant.

Hypothesis (falsifiable):
  If the gain is "second-encoder regularization", it should be
  *present even at small hidden_size* (e.g. 4) because the cross-
  attention mechanism itself is the regularizer, not the parameters.
  Conversely, if the gain comes from extra *capacity*, it should
  scale with hidden_size and may even disappear at small sizes.

3 model kinds × 4 hidden sizes = 12 runs.
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


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _forward(model: nn.Module, model_kind: str, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if model_kind == "video_only":
        fused = torch.cat([batch["video"], batch["audio"]], dim=-1)
        return model(fused)
    return model(batch["video"], batch["audio"])


def _train_one_epoch(
    model: nn.Module, model_kind: str, dataloader, optimizer, device,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch, target in dataloader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        optimizer.zero_grad(set_to_none=True)
        mdn_params = _forward(model, model_kind, batch)
        final = {k: v[:, -1] for k, v in mdn_params.items()}
        loss = mdn_negative_log_likelihood(final, target["params"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def _evaluate(model, model_kind, dataloader, device) -> dict[str, float]:
    model.eval()
    sq_errs = []
    for batch, target in dataloader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        mdn_params = _forward(model, model_kind, batch)
        final = {k: v[:, -1] for k, v in mdn_params.items()}
        mean = mdn_mean(final)
        sq_errs.append((mean - target["params"]).pow(2).sum(dim=-1))
    sq = torch.cat(sq_errs)
    return {"param_mse": float(sq.mean().item())}


def _build_model(model_kind: str, hidden_size: int, num_mixtures: int) -> nn.Module:
    if model_kind == "video_only":
        return BiCfCNADWithMDN(input_size=4, hidden_size=hidden_size, output_size=5, num_mixtures=num_mixtures)
    if model_kind == "cross_attn":
        return CrossModalAttnBiCfCNADWithMDN(
            video_dim=3, audio_dim=1, hidden_size=hidden_size, output_size=5, num_mixtures=num_mixtures,
        )
    if model_kind == "uni_video_xattn":
        return UniVideoSelfXAttnWithMDN(
            video_dim=3, audio_dim=3, hidden_size=hidden_size, output_size=5, num_mixtures=num_mixtures,
        )
    raise ValueError(model_kind)


def _run(model_kind: str, hidden_size: int, args, device, dataset) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    train_loader, _, test_loader = create_emma_rover_dataloaders(dataset, batch_size=args.batch_size, seed=args.seed)
    model = _build_model(model_kind, hidden_size, args.num_mixtures).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.perf_counter()
    for _ in range(args.epochs):
        _train_one_epoch(model, model_kind, train_loader, optimizer, device)
    elapsed = time.perf_counter() - start
    test = _evaluate(model, model_kind, test_loader, device)
    return {
        "model_kind": model_kind,
        "hidden_size": hidden_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": elapsed,
        "test": test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="hidden_size scan on EMMA rover")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--feature-noise-std", type=float, default=0.02)
    parser.add_argument("--num-mixtures", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--output-dir", default="analysis/emma_rover")
    args = parser.parse_args()
    device = torch.device("cpu")
    print("=== EMMA Rover hidden_size Scan ===")
    print(f"hidden_sizes: {args.hidden_sizes} | epochs={args.epochs} | n={args.num_samples}")
    dataset = EmmaRoverRegressionDataset(
        num_samples=args.num_samples, window=args.window,
        feature_noise_std=args.feature_noise_std, seed=args.seed,
    )
    results = []
    for h in args.hidden_sizes:
        for kind in ("video_only", "uni_video_xattn", "cross_attn"):
            r = _run(kind, h, args, device, dataset)
            print(f"  hidden={h:>3d} | {kind:18s} | params={r['parameters']:>5d} | test MSE = {r['test']['param_mse']:.4f}")
            results.append(r)
    # Gain tables
    by_h: dict[int, dict[str, float]] = {}
    for r in results:
        by_h.setdefault(r["hidden_size"], {})[r["model_kind"]] = r["test"]["param_mse"]
    print("\n=== Test param MSE by hidden_size ===")
    print(f"{'hidden':>6s} | {'video_only':>11s} | {'uni_video_xattn':>15s} | {'cross_attn':>11s} | {'xattn_gain':>11s} | {'ca_gain':>9s}")
    ca_gains = []
    uv_gains = []
    for h in sorted(by_h):
        v = by_h[h]["video_only"]
        uv = by_h[h]["uni_video_xattn"]
        ca = by_h[h]["cross_attn"]
        uv_gain = (v - uv) / v if v > 0 else 0
        ca_gain = (v - ca) / v if v > 0 else 0
        ca_gains.append(ca_gain)
        uv_gains.append(uv_gain)
        print(f"{h:>6d} | {v:11.4f} | {uv:15.4f} | {ca:11.4f} | {uv_gain * 100:+10.1f}% | {ca_gain * 100:+8.1f}%")
    if len(ca_gains) >= 2:
        ca_range = max(ca_gains) - min(ca_gains)
        uv_range = max(uv_gains) - min(uv_gains)
        print(f"\ncross_attn gain range: {ca_range * 100:.1f}pp")
        print(f"uni_video_xattn gain range: {uv_range * 100:.1f}pp")
        capacity_centric = ca_range > 0.20
        print(
            f"=> Capacity-architecture verdict: "
            f"{'YES (cross_attn gain varies >20pp with capacity)' if capacity_centric else 'NO (gain is roughly capacity-invariant)'}"
        )
    now = dt.datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}_hidden_size_scan.json"
    payload = {"run_id": run_id, "config": vars(args), "results": results}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults saved to: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
