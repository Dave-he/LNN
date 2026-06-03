#!/usr/bin/env python3
"""NAD gate visualization probe.

Round 48 (4 family x 5 noise level x 3 seed = 60 runs SNR scan) found:
  - Bi-CfC-NAD sigma=0.0 (clean audio) WORST (MSE 581.50)
  - Bi-CfC-NAD sigma=0.1 (slight noise) BEST (MSE 478.77)

Hypothesis: noise_gate sigmoid value (a.k.a. 'retain' = how much the
cell pulls toward h_under_noise vs h_cfc) depends on audio noise.
  - sigma=0.0: NAD has no noise to detect, retain stays near 1
    (always use h_cfc) -> cell behaves like vanilla CfC
  - sigma>=0.1: NAD detects noise, retain drops near 0
    (always use h = h_from_past) -> cell freezes h, ignoring audio

This script trains Bi-CfC-NAD at sigma=0.0/0.1/0.5/2.0 and dumps the
average retain value from the noise_gate_proj layer at each timestep
and each layer.

Hypotheses (falsifiable):
  H_a: retain(sigma=0) > retain(sigma=0.1) consistently
       -> NAD is the gating switch.
  H_b: retain(sigma=0.1+) is ~ 0.5 (binary switch)
       OR retain(sigma=0.1+) is ~ 0 (full h_under_noise dominance)
  H_c: retain dynamics are stable across seeds (not seed-lucky)

Output JSON: analysis/emma_rover/<date>_nad_gate_visualization.json
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
from lnn.data.emma_rover_regression import (
    EmmaRoverRegressionDataset,
    create_emma_rover_dataloaders,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _add_audio_noise(audio: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma == 0:
        return audio
    return audio + torch.randn_like(audio) * sigma


def train_with_retain_capture(
    family_model: nn.Module, train_loader, test_loader, sigma: float,
    epochs: int, lr: float, device: torch.device,
) -> tuple[float, dict[str, list[float]]]:
    """Train CrossModalAttnBiCfCNADWithMDN, capturing noise_gate values per epoch.

    The model has 2 noise_gate_proj layers (one per direction in Bi-CfC-NAD
    bidirectional). We capture sigmoid output of each at every step.
    """
    # Find all noise_gate_proj layers in the model
    targets = []
    for name, module in family_model.named_modules():
        if name.endswith("noise_gate_proj"):
            targets.append((name, module))

    # Register hooks to capture sigmoid output of each noise_gate_proj
    captured: dict[str, list[float]] = {name: [] for name, _ in targets}
    hooks = []
    for name, module in targets:
        def make_hook(n):
            def hook(module, input, output):
                # output is the logit; apply sigmoid
                gate = torch.sigmoid(output)
                captured[n].append(float(gate.mean().item()))
            return hook
        hooks.append(module.register_forward_hook(make_hook(name)))

    # Get the Bi-CfC-NAD cells (for finding audio_encoder.audio_cfc_*.cells)
    # The model structure is:
    #   CrossModalAttnBiCfCNADWithMDN
    #     .video_encoder = BiCfCNADWithMDN -> BidirectionalNoiseAdaptiveCfC
    #     .audio_encoder = BiCfCNADWithMDN -> BidirectionalNoiseAdaptiveCfC
    # Each direction has a NoiseAdaptiveCfCNetwork with cells having noise_gate_proj
    # Hooks on those layers are captured above.

    opt = torch.optim.Adam(family_model.parameters(), lr=lr)
    for _ in range(epochs):
        family_model.train()
        for batch, target in train_loader:
            target = {k: v.to(device) for k, v in target.items()}
            video = batch["video"].to(device)
            audio = batch.get("audio")
            if audio is not None:
                audio = audio.to(device)
            audio = _add_audio_noise(audio, sigma)
            opt.zero_grad()
            out = family_model(video, audio)
            final = {k: v[:, -1] for k, v in out.items()}
            loss = mdn_negative_log_likelihood(final, target["params"])
            loss.backward()
            opt.step()

    # eval
    family_model.eval()
    sq = []
    with torch.no_grad():
        for batch, target in test_loader:
            target = {k: v.to(device) for k, v in target.items()}
            video = batch["video"].to(device)
            audio = batch.get("audio")
            if audio is not None:
                audio = audio.to(device)
            audio = _add_audio_noise(audio, sigma)
            out = family_model(video, audio)
            final = {k: v[:, -1] for k, v in out.items()}
            mean = mdn_mean(final)
            sq.append((mean - target["params"]).pow(2).sum(dim=-1))
    test_mse = float(torch.cat(sq).mean().item())

    for h in hooks:
        h.remove()

    return test_mse, captured


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
    p.add_argument("--noise-levels", type=float, nargs="+",
                    default=[0.0, 0.1, 0.5, 1.0, 2.0])
    p.add_argument("--output-dir", default="analysis/emma_rover")
    args = p.parse_args()
    device = torch.device("cpu")

    print("=== NAD Gate Visualization Probe (round 49) ===")
    print(f"epochs={args.epochs} hidden={args.hidden_size} seeds={args.seeds}")
    print(f"noise_levels={args.noise_levels}")

    results: dict[str, Any] = {}

    for sigma in args.noise_levels:
        for seed in args.seeds:
            torch.manual_seed(seed)
            dataset = EmmaRoverRegressionDataset(
                num_samples=args.num_samples, window=args.window,
                feature_noise_std=0.02, seed=seed,
            )
            train_loader, _, test_loader = create_emma_rover_dataloaders(
                dataset, batch_size=args.batch_size, seed=seed,
            )
            model = CrossModalAttnBiCfCNADWithMDN(
                video_dim=3, audio_dim=1,
                hidden_size=args.hidden_size,
                output_size=5, num_mixtures=args.num_mixtures,
            ).to(device)
            test_mse, captured = train_with_retain_capture(
                model, train_loader, test_loader, sigma, args.epochs, args.lr, device,
            )

            # Aggregate: for each noise_gate_proj, compute mean retain over the run
            retain_summary = {}
            for name, vals in captured.items():
                if vals:
                    retain_summary[name] = {
                        "mean_retain": sum(vals) / len(vals),
                        "min_retain": min(vals),
                        "max_retain": max(vals),
                        "final_retain": vals[-1] if vals else None,
                        "n_steps": len(vals),
                    }
            key = f"sigma{sigma}__seed{seed}"
            results[key] = {
                "sigma": sigma,
                "seed": seed,
                "test_mse": test_mse,
                "retain_per_layer": retain_summary,
            }
            short = " | ".join(
                f"{n.split('.')[-2]}.{n.split('.')[-1]}: {r['mean_retain']:.3f}"
                for n, r in retain_summary.items()
            )
            print(
                f"  sigma={sigma:>4.1f} | seed={seed:>3d} | "
                f"test MSE = {test_mse:>8.4f} | retain: {short}"
            )

    # Aggregate per sigma
    print("\n=== Per-sigma aggregate (mean retain over all layers and seeds) ===")
    per_sigma: dict[str, Any] = {}
    for sigma in args.noise_levels:
        all_retain = []
        per_seed_mse = []
        for seed in args.seeds:
            key = f"sigma{sigma}__seed{seed}"
            r = results[key]
            for layer_name, lr_data in r["retain_per_layer"].items():
                all_retain.append(lr_data["mean_retain"])
            per_seed_mse.append(r["test_mse"])
        per_sigma[str(sigma)] = {
            "n_layers_times_seeds": len(all_retain),
            "mean_retain_overall": sum(all_retain) / len(all_retain) if all_retain else None,
            "min_retain_overall": min(all_retain) if all_retain else None,
            "max_retain_overall": max(all_retain) if all_retain else None,
            "per_seed_mse": per_seed_mse,
            "mean_mse": sum(per_seed_mse) / len(per_seed_mse) if per_seed_mse else None,
        }
        print(
            f"  sigma={sigma:>4.1f} | mean_retain = "
            f"{per_sigma[str(sigma)]['mean_retain_overall']:.4f} | "
            f"min={per_sigma[str(sigma)]['min_retain_overall']:.4f} "
            f"max={per_sigma[str(sigma)]['max_retain_overall']:.4f} | "
            f"mean_mse={per_sigma[str(sigma)]['mean_mse']:.2f}"
        )

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    json_path = output_dir / f"{now}_nad_gate_visualization.json"

    payload = {
        "config": {
            "epochs": args.epochs,
            "hidden_size": args.hidden_size,
            "seeds": args.seeds,
            "noise_levels": args.noise_levels,
            "model": "CrossModalAttnBiCfCNADWithMDN (Bi-CfC-NAD)",
            "note": "retain = sigmoid(noise_gate_proj(noise_score)) in NoiseAdaptiveCfCCell; "
                    "high retain = use h_under_noise (h from past); low retain = use h_cfc (new ODE update)",
        },
        "per_run": results,
        "per_sigma_aggregate": per_sigma,
        "metadata": {
            "round": 49,
            "follows_up": "round 48 SNR threshold scan (28th meta-refinement)",
            "hypotheses": {
                "H_a": "retain(sigma=0) > retain(sigma=0.1)",
                "H_b": "retain(sigma>=0.1) is near 0 (NAD fully gates noise)",
                "H_c": "retain is stable across seeds",
            },
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote: {json_path}")


if __name__ == "__main__":
    main()
