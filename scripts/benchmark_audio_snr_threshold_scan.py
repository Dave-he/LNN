#!/usr/bin/env python3
"""Audio SNR threshold scan — find the vanilla_cfc -> Bi-CfC-NAD switch point.

Round 47 (4 family x 3 audio x 5 seed = 60 runs) revealed:
  - audio=normal/zero: vanilla_cfc (474) is BEST
  - audio=random:     Bi-CfC-NAD (493) is BEST, vanilla_cfc (504) second

Question: at what audio noise level does Bi-CfC-NAD overtake vanilla_cfc?
This script scans 5 noise levels (0.0 / 0.1 / 0.5 / 1.0 / 2.0) for 4
families, with the audio stream being the original audio + N(0, sigma^2)
added on top.

Hypotheses (falsifiable):
  H_a: Bi-CfC-NAD overtakes vanilla_cfc immediately as sigma > 0
       -> NAD is robust even to tiny noise.
  H_b: There is a sharp switch point at sigma ~ 0.5-1.0
       -> the mechanism is threshold-driven, not continuous.
  H_c: vanilla_cfc stays best up to sigma ~ 2.0 (high tolerance)
       -> ODE family has implicit noise tolerance without NAD.

5 noise levels x 4 families x 3 seeds = 60 runs at h=16, ep=20.
Output JSON: analysis/emma_rover/<date>_audio_snr_threshold_scan.json
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
    VanillaCfCXAttnWithMDN,
)
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
    if family == "vanilla_cfc":
        return VanillaCfCXAttnWithMDN(
            video_dim=3, audio_dim=3, hidden_size=hidden_size,
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


def _add_audio_noise(audio: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add N(0, sigma^2) noise on top of audio. sigma=0 means no noise."""
    if sigma == 0:
        return audio
    return audio + torch.randn_like(audio) * sigma


def _step_one(model, batch, target, opt, device, sigma):
    video = batch["video"].to(device)
    audio = batch.get("audio")
    if audio is not None:
        audio = audio.to(device)
    audio = _add_audio_noise(audio, sigma)
    opt.zero_grad()
    out = model(video, audio)
    final = {k: v[:, -1] for k, v in out.items()}
    loss = mdn_negative_log_likelihood(final, target["params"])
    loss.backward()
    opt.step()
    return float(loss.item())


def _eval(model, loader, device, sigma):
    model.eval()
    sq = []
    with torch.no_grad():
        for batch, target in loader:
            target = _move(target, device)
            video = batch["video"].to(device)
            audio = batch.get("audio")
            if audio is not None:
                audio = audio.to(device)
            audio = _add_audio_noise(audio, sigma)
            out = model(video, audio)
            final = {k: v[:, -1] for k, v in out.items()}
            mean = mdn_mean(final)
            sq.append((mean - target["params"]).pow(2).sum(dim=-1))
    sq = torch.cat(sq)
    return float(sq.mean().item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden-size", type=int, default=16)
    p.add_argument("--num-mixtures", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--families", nargs="+",
                    default=["vanilla_cfc", "bi_cfc_nad", "lstm", "gru"])
    p.add_argument("--noise-levels", type=float, nargs="+",
                    default=[0.0, 0.1, 0.5, 1.0, 2.0])
    p.add_argument("--output-dir", default="analysis/emma_rover")
    args = p.parse_args()

    print("=== Audio SNR Threshold Scan (round 48) ===")
    print(
        f"epochs={args.epochs} hidden={args.hidden_size} n={args.num_samples} "
        f"seeds={args.seeds} families={args.families} "
        f"noise_levels={args.noise_levels}"
    )
    print(f"Total runs: {len(args.seeds) * len(args.families) * len(args.noise_levels)}")

    results: dict[str, Any] = {}
    # matrix[fam][sigma] = {per_seed, mean, std}
    matrix: dict[str, dict[str, dict[str, float]]] = {
        fam: {str(s): {} for s in args.noise_levels} for fam in args.families
    }

    for family in args.families:
        for sigma in args.noise_levels:
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
                model = _build_model(family, args.hidden_size, args.num_mixtures).to(torch.device("cpu"))
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                start = time.perf_counter()
                for _ in range(args.epochs):
                    model.train()
                    for batch, target in train_loader:
                        target = _move(target, torch.device("cpu"))
                        _step_one(model, batch, target, opt, torch.device("cpu"), sigma)
                elapsed = time.perf_counter() - start
                test_mse = _eval(model, test_loader, torch.device("cpu"), sigma)
                per_seed.append(test_mse)
                per_seed_times.append(elapsed)
                print(
                    f"  {family:12s} | sigma={sigma:>4.1f} | seed={seed:>3d} | "
                    f"test MSE = {test_mse:>8.4f} | elapsed={elapsed:>5.1f}s"
                )
            mean = sum(per_seed) / len(per_seed)
            var = sum((m - mean) ** 2 for m in per_seed) / max(1, len(per_seed) - 1)
            std = var ** 0.5
            key = f"{family}__sigma{sigma}"
            results[key] = {
                "family": family,
                "sigma": sigma,
                "per_seed_mse": per_seed,
                "per_seed_elapsed_s": per_seed_times,
                "mean_mse": mean,
                "std_mse": std,
                "min_mse": min(per_seed),
                "max_mse": max(per_seed),
                "n_seeds": len(per_seed),
            }
            matrix[family][str(sigma)] = {"mean": mean, "std": std}
            print(
                f"  {family:12s} | sigma={sigma:>4.1f} | "
                f"mean = {mean:>8.4f} +- {std:.4f}"
            )

    # Find switch points: for each family, where does Bi-CfC overtake?
    # We compute for each sigma, the rank of Bi-CfC vs vanilla_cfc.
    print("\n=== Switch point analysis (Bi-CfC-NAD vs vanilla_cfc) ===")
    for sigma in args.noise_levels:
        bi_cfc_mse = matrix["bi_cfc_nad"][str(sigma)]["mean"]
        van_mse = matrix["vanilla_cfc"][str(sigma)]["mean"]
        delta = bi_cfc_mse - van_mse
        winner = "Bi-CfC-NAD" if delta < 0 else "vanilla_cfc"
        print(
            f"  sigma={sigma:>4.1f} | Bi-CfC={bi_cfc_mse:.2f}  vanilla_cfc={van_mse:.2f}  "
            f"delta={delta:+.2f}  winner={winner}"
        )

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    json_path = output_dir / f"{now}_audio_snr_threshold_scan.json"

    payload = {
        "config": {
            "epochs": args.epochs,
            "hidden_size": args.hidden_size,
            "num_mixtures": args.num_mixtures,
            "num_samples": args.num_samples,
            "window": args.window,
            "lr": args.lr,
            "seeds": args.seeds,
            "families": args.families,
            "noise_levels": args.noise_levels,
        },
        "results": results,
        "matrix_view": matrix,
        "switch_point_analysis": {
            str(sigma): {
                "bi_cfc_nad_mean": matrix["bi_cfc_nad"][str(sigma)]["mean"],
                "vanilla_cfc_mean": matrix["vanilla_cfc"][str(sigma)]["mean"],
                "delta": matrix["bi_cfc_nad"][str(sigma)]["mean"]
                        - matrix["vanilla_cfc"][str(sigma)]["mean"],
                "winner": ("Bi-CfC-NAD"
                            if matrix["bi_cfc_nad"][str(sigma)]["mean"]
                                < matrix["vanilla_cfc"][str(sigma)]["mean"]
                            else "vanilla_cfc"),
            }
            for sigma in args.noise_levels
        },
        "metadata": {
            "round": 48,
            "follows_up": "round 47 (60 runs, vanilla_cfc wins in clean audio, Bi-CfC wins in random)",
            "hypotheses": {
                "H_a": "Bi-CfC overtakes vanilla_cfc immediately (sigma>0)",
                "H_b": "Sharp switch point at sigma~0.5-1.0",
                "H_c": "vanilla_cfc stays best up to sigma=2.0 (ODE noise tolerance)",
            },
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote: {json_path}")


if __name__ == "__main__":
    main()
