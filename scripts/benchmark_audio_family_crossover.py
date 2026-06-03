#!/usr/bin/env python3
"""Audio mode x encoder family crossover probe.

Round 45 (loop r45, 5-seed x 3 family LOO) revealed encoder family
ranking is regime-conditional.  This script adds the third
dimension: **audio mode** (normal / zero / random) crossed with
the 3 encoder families.

Hypotheses (falsifiable):
  H_a: Audio mode has NO effect on family ranking
       -> family dominance is uniform across input modalities.
  H_b: Audio mode interacts with family
       -> e.g. Bi-CfC stays robust under audio=random (NAD
          down-weights noise), but LSTM/GRU degrade.
  H_c: Audio mode is family-independent (~5pp shift)
       -> consistent with round 35 finding that audio content
          contributes <=5pp on Bi-CfC.

5 seeds x 3 audio modes x 3 families = 45 runs at h=16/ep=20
(small-budget for speed; LOO would be ~1 hour).  Audio mode is
applied to the audio_encoder input stream only; the second
encoder still receives the video stream (round 21 protocol).

Output JSON (analysis/emma_rover/):
  {
    "config": {...},
    "results": {
      "(family, audio_mode)": {"per_seed_mse": [...], "mean": ..., "std": ...},
      ...
    },
    "matrix_view": {
      "family -> {normal: ..., zero: ..., random: ...}"
    },
    "interaction_2way_anova_F": ...   # crude proxy
  }
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
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
    if family == "vanilla_cfc":
        return VanillaCfCXAttnWithMDN(
            video_dim=3, audio_dim=3, hidden_size=hidden_size,
            output_size=5, num_mixtures=num_mixtures,
        )
    raise ValueError(family)


def _audio_apply(audio: torch.Tensor | None, video: torch.Tensor, mode: str) -> torch.Tensor | None:
    """Apply audio mode transformation.  Return None to drop audio (zero case for non-zero flows)."""
    if mode == "normal":
        return audio
    if mode == "zero":
        if audio is not None:
            return torch.zeros_like(audio)
        return torch.zeros_like(video[:, :, :1])
    if mode == "random":
        if audio is not None:
            return torch.randn_like(audio)
        return torch.randn_like(video[:, :, :1])
    raise ValueError(mode)


def _step_one(model, batch, target, opt, device, audio_mode):
    video = batch["video"].to(device)
    audio = batch.get("audio")
    if audio is not None:
        audio = audio.to(device)
    audio = _audio_apply(audio, video, audio_mode)
    opt.zero_grad()
    out = model(video, audio)
    final = {k: v[:, -1] for k, v in out.items()}
    loss = mdn_negative_log_likelihood(final, target["params"])
    loss.backward()
    opt.step()
    return float(loss.item())


def _eval(model, loader, device, audio_mode):
    model.eval()
    sq = []
    with torch.no_grad():
        for batch, target in loader:
            target = _move(target, device)
            video = batch["video"].to(device)
            audio = batch.get("audio")
            if audio is not None:
                audio = audio.to(device)
            audio = _audio_apply(audio, video, audio_mode)
            out = model(video, audio)
            final = {k: v[:, -1] for k, v in out.items()}
            mean = mdn_mean(final)
            sq.append((mean - target["params"]).pow(2).sum(dim=-1))
    sq = torch.cat(sq)
    return float(sq.mean().item())


def _run_one(family: str, audio_mode: str, seed: int, hidden_size: int, num_mixtures: int,
             num_samples: int, window: int, batch_size: int, lr: float, epochs: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    dataset = EmmaRoverRegressionDataset(
        num_samples=num_samples, window=window,
        feature_noise_std=0.02, seed=seed,
    )
    train_loader, _, test_loader = create_emma_rover_dataloaders(
        dataset, batch_size=batch_size, seed=seed,
    )
    model = _build_model(family, hidden_size, num_mixtures).to(torch.device("cpu"))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for batch, target in train_loader:
            target = _move(target, torch.device("cpu"))
            _step_one(model, batch, target, opt, torch.device("cpu"), audio_mode)
    elapsed = time.perf_counter() - start
    test_mse = _eval(model, test_loader, torch.device("cpu"), audio_mode)
    return test_mse, elapsed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden-size", type=int, default=16)
    p.add_argument("--num-mixtures", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 7, 42])
    p.add_argument("--families", nargs="+", default=["bi_cfc_nad", "vanilla_cfc", "lstm", "gru"])
    p.add_argument("--audio-modes", nargs="+", default=["normal", "zero", "random"])
    p.add_argument("--output-dir", default="analysis/emma_rover")
    args = p.parse_args()

    print("=== Audio Mode x Encoder Family Crossover Probe (round 46) ===")
    print(
        f"epochs={args.epochs} hidden={args.hidden_size} n={args.num_samples} "
        f"seeds={args.seeds} families={args.families} audio_modes={args.audio_modes}"
    )
    print(f"Total runs: {len(args.seeds) * len(args.families) * len(args.audio_modes)}")

    results: dict[str, Any] = {}
    matrix: dict[str, dict[str, dict[str, float]]] = {fam: {am: {} for am in args.audio_modes} for fam in args.families}

    for family in args.families:
        for audio_mode in args.audio_modes:
            per_seed = []
            per_seed_times = []
            for seed in args.seeds:
                test_mse, elapsed = _run_one(
                    family, audio_mode, seed, args.hidden_size, args.num_mixtures,
                    args.num_samples, args.window, args.batch_size, args.lr, args.epochs,
                )
                per_seed.append(test_mse)
                per_seed_times.append(elapsed)
                print(
                    f"  {family:12s} | audio={audio_mode:6s} | seed={seed:>3d} | "
                    f"test MSE = {test_mse:>8.4f} | elapsed={elapsed:>5.1f}s"
                )
            mean = sum(per_seed) / len(per_seed)
            var = sum((m - mean) ** 2 for m in per_seed) / max(1, len(per_seed) - 1)
            std = var ** 0.5
            key = f"{family}__{audio_mode}"
            results[key] = {
                "family": family,
                "audio_mode": audio_mode,
                "per_seed_mse": per_seed,
                "per_seed_elapsed_s": per_seed_times,
                "mean_mse": mean,
                "std_mse": std,
                "min_mse": min(per_seed),
                "max_mse": max(per_seed),
                "n_seeds": len(per_seed),
            }
            matrix[family][audio_mode] = {"mean": mean, "std": std}
            print(
                f"  {family:12s} | audio={audio_mode:6s} | "
                f"mean = {mean:>8.4f} +- {std:.4f}"
            )

    # Crude 2-way ANOVA proxy: variance decomposition
    # Total variance / family variance / audio variance / interaction variance
    all_means = [[results[f"{f}__{a}"]["mean_mse"] for a in args.audio_modes] for f in args.families]
    grand_mean = sum(v for row in all_means for v in row) / (len(args.families) * len(args.audio_modes))
    family_means = [sum(row) / len(row) for row in all_means]
    audio_means = [sum(all_means[f][a] for f in range(len(args.families))) / len(args.families)
                    for a in range(len(args.audio_modes))]
    n_per_cell = len(args.seeds)
    ss_total = sum((cell - grand_mean) ** 2 for row in all_means for cell in row) * n_per_cell
    ss_family = n_per_cell * len(args.audio_modes) * sum((fm - grand_mean) ** 2 for fm in family_means)
    ss_audio = n_per_cell * len(args.families) * sum((am - grand_mean) ** 2 for am in audio_means)
    ss_interaction = ss_total - ss_family - ss_audio

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    json_path = output_dir / f"{now}_audio_family_crossover.json"

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
            "audio_modes": args.audio_modes,
            "protocol": "EmmaRoverRegressionDataset random-window (matches round 45 protocol)",
        },
        "results": results,
        "matrix_view": matrix,
        "anova_proxy": {
            "grand_mean": grand_mean,
            "family_means": dict(zip(args.families, family_means)),
            "audio_means": dict(zip(args.audio_modes, audio_means)),
            "ss_total": ss_total,
            "ss_family": ss_family,
            "ss_audio": ss_audio,
            "ss_interaction": ss_interaction,
            "frac_family": ss_family / max(1e-9, ss_total),
            "frac_audio": ss_audio / max(1e-9, ss_total),
            "frac_interaction": ss_interaction / max(1e-9, ss_total),
        },
        "metadata": {
            "round": 46,
            "follows_up": [
                "round 45 (5-seed x 3 family LOO, 25th meta-refinement)",
                "round 35 (audio mode x Bi-CfC only, 5pp contribution)",
            ],
            "hypotheses": {
                "H_a": "audio mode has no effect on family ranking (REFUTED if not)",
                "H_b": "audio mode interacts with family (NAD down-weights noisy input)",
                "H_c": "audio mode is family-independent (~5pp shift, consistent with round 35)",
            },
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote: {json_path}")

    # Final verdict
    print("\n=== Verdict ===")
    print("Family ranking by audio mode:")
    for audio_mode in args.audio_modes:
        ranking = sorted(
            args.families,
            key=lambda f: results[f"{f}__{audio_mode}"]["mean_mse"],
        )
        print(f"  audio={audio_mode:6s} : " + " > ".join(
            f"{f} ({results[f'{f}__{audio_mode}']['mean_mse']:.2f})" for f in ranking
        ))

    print("\nANOVA proxy (fraction of variance):")
    ap = payload["anova_proxy"]
    print(f"  family:     {ap['frac_family']:.2%}")
    print(f"  audio:      {ap['frac_audio']:.2%}")
    print(f"  interaction: {ap['frac_interaction']:.2%}")


if __name__ == "__main__":
    main()
