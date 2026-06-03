#!/usr/bin/env python3
"""CfC cell branches dump — locate the sub-branch that responds to audio noise.

Round 50 cross-attn ablation (30 runs) located the sigma-switch
mechanism in cross-attn q/k/v projections.  This script goes one
level deeper:  *which* sub-component of the CfC cell responds to
the noisy audio features that pass through cross-attn?

The CfC cell has 3 sub-branches:
  - f_gate:  Linear + Sigmoid  (controls decay rate)
  - g_branch: Linear + Tanh     (new state candidate)
  - h_branch: Linear + Tanh     (stable state candidate)

The final hidden is:  h_new = decay * g + (1-decay) * h_out
where decay = sigmoid(-f * time_scale * dt).

We forward-hook the output of each sub-branch (after activation) at
each timestep and aggregate mean over the run.

Hypotheses (falsifiable):
  H_a: f_gate mean shifts with sigma -> f_gate drives sigma-switch
  H_b: g_branch mean shifts with sigma -> g_branch drives
  H_c: h_branch mean shifts with sigma -> h_branch drives
  H_d: all three are sigma-invariant -> mechanism is elsewhere
       (e.g. in time_scale, h state, or the audio_encoder itself)

3 seeds x 5 noise levels x 3 sub-branches (plus time_scale) = 60
measurements on Bi-CfC-NAD audio_encoder (the most-affected one).
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


def _set_forward_hooks_on_cells(model: nn.Module) -> tuple[dict[str, list[float]], list]:
    """Register forward hooks on all f_gate / g_branch / h_branch sub-modules.

    Targets NoiseAdaptiveCfCCell, which uses the SAME CfCCell architecture
    but with extra noise gating.  The 3 sub-branches are inside each cell.
    """
    captured: dict[str, list[float]] = {}
    hooks = []

    for name, module in model.named_modules():
        # Match submodule names that end in 'f_gate', 'g_branch', 'h_branch'
        # and are nn.Sequential or nn.Linear (or anything that has a forward)
        if any(name.endswith(f".{tag}") or name == tag
                for tag in ("f_gate", "g_branch", "h_branch")):
            # Skip the OUTER f_gate/g_branch/h_branch in CfCCell (not in NoiseAdaptiveCfCCell).
            # NoiseAdaptiveCfCCell uses cell.f_gate, cell.g_branch, cell.h_branch.
            # These are inside NoiseAdaptiveCfCCell.
            captured[name] = []

            def make_hook(n):
                def hook(module, input, output):
                    captured[n].append(float(output.mean().item()))
                return hook
            hooks.append(module.register_forward_hook(make_hook(name)))
    return captured, hooks


def train_with_hooks(
    family_model, train_loader, test_loader, sigma, epochs, lr, device,
) -> tuple[float, dict[str, list[float]]]:
    captured, hooks = _set_forward_hooks_on_cells(family_model)

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

    print("=== CfC Cell Branches Dump Probe (round 51) ===")
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
                output_size=5, num_mixtures=1,
            ).to(device)
            test_mse, captured = train_with_hooks(
                model, train_loader, test_loader, sigma, args.epochs, args.lr, device,
            )

            branch_summary = {}
            for name, vals in captured.items():
                if vals:
                    branch_summary[name] = {
                        "mean": sum(vals) / len(vals),
                        "min": min(vals),
                        "max": max(vals),
                        "final": vals[-1] if vals else None,
                        "n": len(vals),
                    }
            key = f"sigma{sigma}__seed{seed}"
            results[key] = {
                "sigma": sigma,
                "seed": seed,
                "test_mse": test_mse,
                "branches": branch_summary,
            }
            # short printout: just f_gate/g_branch/h_branch average across all
            # matching layers
            f_means = [v["mean"] for n, v in branch_summary.items() if n.endswith(".f_gate")]
            g_means = [v["mean"] for n, v in branch_summary.items() if n.endswith(".g_branch")]
            h_means = [v["mean"] for n, v in branch_summary.items() if n.endswith(".h_branch")]
            print(
                f"  sigma={sigma:>4.1f} | seed={seed:>3d} | "
                f"test MSE = {test_mse:>8.4f} | "
                f"f_gate={sum(f_means)/len(f_means):.3f} "
                f"g_branch={sum(g_means)/len(g_means):.3f} "
                f"h_branch={sum(h_means)/len(h_means):.3f}"
            )

    # Aggregate per sigma
    print("\n=== Per-sigma aggregate (mean over seeds, by branch) ===")
    per_sigma: dict[str, Any] = {}
    for sigma in args.noise_levels:
        f_pool, g_pool, h_pool = [], [], []
        mse_pool = []
        for seed in args.seeds:
            key = f"sigma{sigma}__seed{seed}"
            r = results[key]
            for n, v in r["branches"].items():
                if n.endswith(".f_gate"):
                    f_pool.append(v["mean"])
                elif n.endswith(".g_branch"):
                    g_pool.append(v["mean"])
                elif n.endswith(".h_branch"):
                    h_pool.append(v["mean"])
            mse_pool.append(r["test_mse"])
        per_sigma[str(sigma)] = {
            "f_gate_mean": sum(f_pool) / len(f_pool) if f_pool else None,
            "g_branch_mean": sum(g_pool) / len(g_pool) if g_pool else None,
            "h_branch_mean": sum(h_pool) / len(h_pool) if h_pool else None,
            "mse_mean": sum(mse_pool) / len(mse_pool) if mse_pool else None,
            "f_gate_std": (sum((x - sum(f_pool)/len(f_pool))**2 for x in f_pool) / max(1, len(f_pool)-1)) ** 0.5 if f_pool else None,
            "g_branch_std": (sum((x - sum(g_pool)/len(g_pool))**2 for x in g_pool) / max(1, len(g_pool)-1)) ** 0.5 if g_pool else None,
            "h_branch_std": (sum((x - sum(h_pool)/len(h_pool))**2 for x in h_pool) / max(1, len(h_pool)-1)) ** 0.5 if h_pool else None,
        }
        ps = per_sigma[str(sigma)]
        print(
            f"  sigma={sigma:>4.1f} | f_gate={ps['f_gate_mean']:.4f}±{ps['f_gate_std']:.4f}  "
            f"g_branch={ps['g_branch_mean']:.4f}±{ps['g_branch_std']:.4f}  "
            f"h_branch={ps['h_branch_mean']:.4f}±{ps['h_branch_std']:.4f}  "
            f"MSE={ps['mse_mean']:.2f}"
        )

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    json_path = output_dir / f"{now}_cfc_cell_branches_dump.json"
    payload = {
        "config": {
            "epochs": args.epochs,
            "hidden_size": args.hidden_size,
            "seeds": args.seeds,
            "noise_levels": args.noise_levels,
            "model": "CrossModalAttnBiCfCNADWithMDN (Bi-CfC-NAD)",
            "hooks_on": "f_gate, g_branch, h_branch (post-activation outputs)",
        },
        "per_run": results,
        "per_sigma_aggregate": per_sigma,
        "metadata": {
            "round": 51,
            "follows_up": "round 50 cross-attn ablation (located mechanism in cross-attn q/k/v)",
            "hypotheses": {
                "H_a": "f_gate mean shifts with sigma (f_gate drives sigma-switch)",
                "H_b": "g_branch mean shifts with sigma",
                "H_c": "h_branch mean shifts with sigma",
                "H_d": "all three are sigma-invariant (mechanism elsewhere)",
            },
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote: {json_path}")


if __name__ == "__main__":
    main()
