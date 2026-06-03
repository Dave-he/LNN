#!/usr/bin/env python3
"""Round 33: REAL adaptive freeze (cross_attn + audio_encoder freeze) on
EMMA segment LOO. Cron round 32 (commit ff4bedd) ran 'adaptive_freeze' on
the segment LOO dataset but the script actually trained pure video_only
(no cross-attention, no audio_encoder to freeze). This script runs the
ACTUAL 32-round SOTA recipe: cross_attn warmup K=40 → freeze
audio_encoder → continue training 40 more epochs.

Hypothesis: if the SOTA 0.31 from round 26 reflects a real LNN
multimodal advantage (not just data leakage), the SAME recipe should
also yield a meaningful gain on the strict segment-LOO test. If
adaptive freeze mean LOO MSE < cron's video_only mean (14.89), recipe
has REAL generalisation value; otherwise SOTA is segment-overfitting.
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
from lnn.data.emma_rover_temporal_folds import (
    TemporalSegmentRegressionDataset,
    create_segment_loo_dataloaders,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _train_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    n = 0
    for batch, target in loader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        opt.zero_grad(set_to_none=True)
        out = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in out.items()}
        loss = mdn_negative_log_likelihood(final, target["params"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    sq = []
    for batch, target in loader:
        batch = _move(batch, device)
        target = {k: v.to(device) for k, v in target.items()}
        out = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in out.items()}
        mean = mdn_mean(final)
        sq.append((mean - target["params"]).pow(2).sum(dim=-1))
    return float(torch.cat(sq).mean().item())


def _real_adaptive_freeze_run(train_loader, test_loader, hidden_size, total_epochs, warmup_epochs, device, seed):
    torch.manual_seed(seed)
    model = CrossModalAttnBiCfCNADWithMDN(
        video_dim=3, audio_dim=1, hidden_size=hidden_size,
        output_size=5, num_mixtures=1,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    # Phase 1
    for _ in range(warmup_epochs):
        _train_epoch(model, train_loader, opt, device)
    # Freeze audio_encoder
    for p in model.audio_encoder.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=5e-3)
    # Phase 2
    for _ in range(total_epochs - warmup_epochs):
        _train_epoch(model, train_loader, opt, device)
    return _eval(model, test_loader, device)


def main():
    p = argparse.ArgumentParser(description="REAL adaptive freeze on EMMA segment LOO")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--warmup-epochs", type=int, default=40)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="analysis/emma_rover")
    args = p.parse_args()
    device = torch.device("cpu")
    print(f"=== REAL adaptive freeze on EMMA Segment LOO ===")
    print(f"epochs={args.epochs} warmup={args.warmup_epochs} hidden={args.hidden_size}")
    dataset = TemporalSegmentRegressionDataset(seed=args.seed)
    runs = []
    for fold in range(4):
        train_loader, test_loader = create_segment_loo_dataloaders(
            dataset, held_out_fold=fold, batch_size=8,
        )
        start = time.perf_counter()
        test_mse = _real_adaptive_freeze_run(
            train_loader, test_loader, args.hidden_size, args.epochs, args.warmup_epochs,
            device, args.seed,
        )
        elapsed = time.perf_counter() - start
        runs.append({
            "fold": fold, "test_param_mse": test_mse, "elapsed_seconds": elapsed,
        })
        print(f"  fold {fold} (held out) | test MSE = {test_mse:.4f}  ({elapsed:.1f}s)")
    mses = [r["test_param_mse"] for r in runs]
    mean = sum(mses) / len(mses)
    std = (sum((m - mean) ** 2 for m in mses) / len(mses)) ** 0.5
    print()
    print(f"=== REAL adaptive freeze cross-segment LOO summary ===")
    print(f"  per-fold MSE: {[f'{m:.4f}' for m in mses]}")
    print(f"  mean = {mean:.4f}, std = {std:.4f}")
    cron_vo_mean = 14.89
    cron_vo_per_fold = [4.36, 5.28, 18.07, 31.83]
    diff = (cron_vo_mean - mean) / cron_vo_mean * 100
    print(f"  vs cron pure-vo LOO mean (14.89): {'+' if diff > 0 else ''}{diff:.1f}%")
    if mean < cron_vo_mean:
        verdict = f"REAL adaptive freeze BEATS pure video_only on LOO by {diff:.1f}%"
    else:
        verdict = f"REAL adaptive freeze does NOT beat pure video_only on LOO ({mean:.2f} >= 14.89)"
    print(f"=> {verdict}")
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now()
    json_path = out_dir / f"{now.strftime('%Y-%m-%d_%H%M%S')}_segment_loo_real_adaptive_freeze.json"
    json_path.write_text(json.dumps({
        "config": vars(args),
        "runs": runs,
        "summary": {"mean": mean, "std": std, "verdict": verdict,
                    "vs_cron_vo_mean_diff_pct": diff},
    }, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
