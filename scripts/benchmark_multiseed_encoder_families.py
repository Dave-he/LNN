#!/usr/bin/env python3
"""Multi-seed encoder-family probe.

Round 43 (commit 1bb78af) revealed the round-38 LOO SOTA 0.42 is
*seed-lucky*: 5-seed mean 8.16 ± 6.78 (3 of 4 new seeds 27-37× worse).
This script extends the same multi-seed protocol to the encoder-family
axis (Bi-CfC-NAD vs LSTM vs GRU) to answer the 24th meta-question:

    "Is seed-sensitivity an *architecture* property or a
     *cross-modal SOTA recipe* property?"

Hypotheses (falsifiable):
  H_a: All three families have similar multi-seed variance
       -> seed-sensitivity is recipe-general, not family-specific.
  H_b: Bi-CfC-NAD has *lower* multi-seed variance than LSTM/GRU
       -> "Bi-CfC family" is more reliable, supporting the
          round 21 family-necessary claim.
  H_c: LSTM (round 25 §28: +36.1%) is *robust* across seeds
       -> LSTM is a more reliable production baseline than
          round-38 single-seed Bi-CfC SOTA.

Output JSON (analysis/emma_rover/):
  {
    "config": {epochs, hidden_size, K, audio_mode, seeds, n_seeds},
    "families": {
      "bi_cfc_nad": {"per_seed": [...], "mean": ..., "std": ...},
      "lstm":       {"per_seed": [...], "mean": ..., "std": ...},
      "gru":        {"per_seed": [...], "mean": ..., "std": ...},
    },
    "ensemble": {
      "bi_cfc_nad_avg_5": ...,   # mean of 5 seeds' predictions
      "lstm_avg_5": ...,
      "gru_avg_5": ...,
    }
  }
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
    GRUEncoderXAttnWithMDN,
    LSTMEncoderXAttnWithMDN,
)
from lnn.core.noise_adaptive_cfc import BiCfCNADWithMDN
from lnn.data.emma_rover_regression import (
    EmmaRoverRegressionDataset,
    create_emma_rover_dataloaders,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _move(b, d):
    return {k: v.to(d) for k, v in b.items()}


def _build_model(family: str, hidden_size: int, num_mixtures: int) -> nn.Module:
    if family == "bi_cfc_nad":
        return CrossModalAttnBiCfCNADWithMDN(
            video_dim=3, audio_dim=1, hidden_size=hidden_size,
            output_size=5, num_mixtures=num_mixtures,
        )
    if family == "lstm":
        return LSTMEncoderXAttnWithMDN(
            video_dim=3, audio_dim=3, hidden_size=hidden_size,
            output_size=5, num_mixtures=num_mixtures,
        )
    if family == "gru":
        return GRUEncoderXAttnWithMDN(
            video_dim=3, audio_dim=1, hidden_size=hidden_size,
            output_size=5, num_mixtures=num_mixtures,
        )
    raise ValueError(family)


def _train(model, loader, opt, device, audio_mode: str = "normal") -> float:
    model.train()
    losses = []
    for batch, target in loader:
        target = _move(target, device)
        # ignore batch["video"] positionals; let the data loader's collate
        # supply them. Cross-modal input is the (video, audio) pair.
        # We need to unpack and feed per-model below.
        losses.append(_step_one(model, batch, target, opt, device, audio_mode))
    return float(sum(losses) / max(1, len(losses)))


def _step_one(model, batch, target, opt, device, audio_mode):
    video = batch["video"].to(device)
    audio = batch.get("audio")
    if audio is not None:
        audio = audio.to(device)
    if audio_mode == "zero":
        audio = torch.zeros_like(video[:, :, :1])  # only the audio dim differs
    opt.zero_grad()
    out = model(video, audio)
    final = {k: v[:, -1] for k, v in out.items()}
    loss = mdn_negative_log_likelihood(final, target["params"])
    loss.backward()
    opt.step()
    return float(loss.item())


def _eval(model, loader, device, audio_mode: str = "normal") -> float:
    model.eval()
    sq = []
    with torch.no_grad():
        for batch, target in loader:
            target = _move(target, device)
            video = batch["video"].to(device)
            audio = batch.get("audio")
            if audio is not None:
                audio = audio.to(device)
            if audio_mode == "zero":
                audio = torch.zeros_like(video[:, :, :1])
            out = model(video, audio)
            final = {k: v[:, -1] for k, v in out.items()}
            mean = mdn_mean(final)
            sq.append((mean - target["params"]).pow(2).sum(dim=-1))
    sq = torch.cat(sq)
    return float(sq.mean().item())


def _collect_predictions(model, loader, device, audio_mode: str = "normal"):
    """Return all per-sample predictions (for ensemble averaging)."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch, target in loader:
            target = _move(target, device)
            video = batch["video"].to(device)
            audio = batch.get("audio")
            if audio is not None:
                audio = audio.to(device)
            if audio_mode == "zero":
                audio = torch.zeros_like(video[:, :, :1])
            out = model(video, audio)
            final = {k: v[:, -1] for k, v in out.items()}
            mean = mdn_mean(final)
            preds.append(mean.cpu())
            targets.append(target["params"].cpu())
    return torch.cat(preds), torch.cat(targets)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--hidden-size", type=int, default=16)
    p.add_argument("--num-mixtures", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--audio-mode", default="normal", choices=["normal", "zero", "random"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 7, 42])
    p.add_argument("--families", nargs="+", default=["bi_cfc_nad", "lstm", "gru"])
    p.add_argument("--output-dir", default="analysis/emma_rover")
    args = p.parse_args()
    device = torch.device("cpu")

    print("=== Multi-Seed Encoder-Family Probe (round 43 follow-up) ===")
    print(
        f"epochs={args.epochs} hidden={args.hidden_size} n={args.num_samples} "
        f"audio={args.audio_mode} seeds={args.seeds} families={args.families}"
    )

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H%M%S")

    families_results: dict[str, Any] = {}
    families_ensemble_preds: dict[str, list[torch.Tensor]] = {
        fam: [] for fam in args.families
    }
    targets_ref = None

    for family in args.families:
        per_seed = []
        per_seed_times = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            dataset = EmmaRoverRegressionDataset(
                num_samples=args.num_samples, window=args.window,
                feature_noise_std=0.02, seed=seed,
            )
            train_loader, _, test_loader = create_emma_rover_dataloaders(
                dataset, batch_size=args.batch_size, seed=seed,
            )
            model = _build_model(family, args.hidden_size, args.num_mixtures).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            start = time.perf_counter()
            for _ in range(args.epochs):
                _train(model, train_loader, opt, device, args.audio_mode)
            elapsed = time.perf_counter() - start
            test_mse = _eval(model, test_loader, device, args.audio_mode)
            per_seed.append(test_mse)
            per_seed_times.append(elapsed)

            # collect predictions for ensemble
            preds, tgts = _collect_predictions(model, test_loader, device, args.audio_mode)
            families_ensemble_preds[family].append(preds)
            if targets_ref is None:
                targets_ref = tgts

            print(
                f"  {family:12s} | seed={seed:>3d} | test MSE = {test_mse:>8.4f} | "
                f"elapsed={elapsed:>5.1f}s"
            )

        mean = float(sum(per_seed) / len(per_seed))
        var = float(sum((m - mean) ** 2 for m in per_seed) / max(1, len(per_seed) - 1))
        std = var ** 0.5
        families_results[family] = {
            "per_seed_mse": per_seed,
            "per_seed_elapsed_s": per_seed_times,
            "mean_mse": mean,
            "std_mse": std,
            "min_mse": min(per_seed),
            "max_mse": max(per_seed),
            "n_seeds": len(per_seed),
        }
        print(
            f"  {family:12s} | mean = {mean:>8.4f} ± {std:.4f}  "
            f"min = {min(per_seed):>8.4f}  max = {max(per_seed):>8.4f}"
        )

    # Ensemble: average predictions across seeds for each family
    ensemble: dict[str, float] = {}
    for family, preds_list in families_ensemble_preds.items():
        if not preds_list:
            continue
        avg_pred = torch.stack(preds_list, dim=0).mean(dim=0)
        assert targets_ref is not None
        mse = float(((avg_pred - targets_ref) ** 2).sum(dim=-1).mean().item())
        ensemble[f"{family}_avg"] = mse
        print(f"  ENSEMBLE  {family:12s} | seed-avg MSE = {mse:>8.4f}")

    payload = {
        "config": {
            "epochs": args.epochs,
            "hidden_size": args.hidden_size,
            "num_mixtures": args.num_mixtures,
            "num_samples": args.num_samples,
            "window": args.window,
            "lr": args.lr,
            "audio_mode": args.audio_mode,
            "seeds": args.seeds,
            "families": args.families,
        },
        "families": families_results,
        "ensemble": ensemble,
        "metadata": {
            "round": 44,
            "follows_up": "1bb78af (round 43, seed-lucky refutation of round-38 SOTA 0.42)",
            "hypotheses": {
                "H_a": "all families have similar multi-seed variance",
                "H_b": "Bi-CfC-NAD has lower multi-seed variance than LSTM/GRU",
                "H_c": "LSTM is robust across seeds (round 25 §28 +36.1% claim re-validated)",
            },
        },
    }
    json_path = output_dir / f"{run_id}_multiseed_encoder_families.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote: {json_path}")

    # Quick verdict
    print("\n=== Verdict (vs round 43 single-seed SOTA 0.42) ===")
    for family, res in families_results.items():
        if res["mean_mse"] < 1.0:
            print(f"  {family:12s} MEAN < 1.0  -> single-seed SOTA NOT seed-lucky for this family")
        else:
            print(
                f"  {family:12s} MEAN = {res['mean_mse']:>6.2f} (single-seed lucky to be <1.0)"
            )


if __name__ == "__main__":
    main()
