"""Inject recipe at large-budget h=64/ep=80 + finer inject scan.

Two probes in one:

Probe 1: Does inject=0.1 production recipe GENERALIZE to large-budget
h=64/ep=80 (5x compute)?
  - 2 inject (0.0 vs 0.1) x 1 large-budget (h=64, ep=80) x 3 seeds
  - 6 runs total
  - Hypothesis: inject=0.1 wins at h=64/ep=80 too (recipe is budget-
    invariant)

Probe 2: Finer inject scan (0.05/0.1/0.15/0.2) at small-budget
h=16/ep=20 to find optimal inject.
  - 4 inject levels x 3 seeds = 12 runs
  - Hypothesis: 0.1 is the optimal point (matches round 53 peak)
  - Alternative: there's a U-shape with 0.1 being near the bottom

Total: 18 runs.
"""
import os, sys, json, datetime as dt, pathlib, time
sys.path.insert(0, "/Users/hyx/workspace/LNN")

import torch
from lnn.core.mdn import mdn_mean, mdn_negative_log_likelihood
from lnn.core.multimodal_physreg import CrossModalAttnBiCfCNADWithMDN
from lnn.data.emma_rover_regression import (
    EmmaRoverRegressionDataset, create_emma_rover_dataloaders,
)

ROOT = pathlib.Path("/Users/hyx/workspace/LNN")
device = torch.device("cpu")
LR = 5e-3
SEEDS = [1, 2, 3]

def audio_apply(audio, sigma):
    if sigma == 0: return audio
    return audio + torch.randn_like(audio) * sigma

def train_eval(model, tl, te, inject_sigma, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        model.train()
        for batch, target in tl:
            target = {k: v.to(device) for k, v in target.items()}
            video = batch["video"].to(device)
            audio = batch.get("audio")
            if audio is not None: audio = audio.to(device)
            audio = audio_apply(audio, inject_sigma)
            opt.zero_grad()
            out = model(video, audio)
            final = {k: v[:, -1] for k, v in out.items()}
            loss = mdn_negative_log_likelihood(final, target["params"])
            loss.backward()
            opt.step()
    model.eval()
    sq = []
    with torch.no_grad():
        for batch, target in te:
            target = {k: v.to(device) for k, v in target.items()}
            video = batch["video"].to(device)
            audio = batch.get("audio")
            if audio is not None: audio = audio.to(device)
            # Test with sigma=0.0 (clean) - worst case for the recipe
            audio = audio_apply(audio, 0.0)
            out = model(video, audio)
            final = {k: v[:, -1] for k, v in out.items()}
            mean = mdn_mean(final)
            sq.append((mean - target["params"]).pow(2).sum(dim=-1))
    return float(torch.cat(sq).mean().item())

# Probe 1: large-budget h=64/ep=80, inject=0.0 vs 0.1
print("=== Probe 1: inject recipe at large-budget h=64/ep=80 ===")
print("(epochs=80 hidden=64 test_sigma=0.0)")
probe1_results = {}
for inject in [0.0, 0.1]:
    per_seed = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        ds = EmmaRoverRegressionDataset(num_samples=200, window=16, feature_noise_std=0.02, seed=seed)
        tl, _, te = create_emma_rover_dataloaders(ds, batch_size=16, seed=seed)
        model = CrossModalAttnBiCfCNADWithMDN(
            video_dim=3, audio_dim=1, hidden_size=64,
            output_size=5, num_mixtures=1,
        ).to(device)
        start = time.perf_counter()
        mse = train_eval(model, tl, te, inject, epochs=80)
        elapsed = time.perf_counter() - start
        per_seed.append(mse)
        print(f"  inject={inject:>4.1f} | seed={seed:>3d} | MSE={mse:>8.4f} | elapsed={elapsed:>5.1f}s")
    mean = sum(per_seed) / len(per_seed)
    std = (sum((m - mean)**2 for m in per_seed) / max(1, len(per_seed)-1)) ** 0.5
    probe1_results[f"inject{inject}"] = {"inject": inject, "per_seed_mse": per_seed, "mean": mean, "std": std}
    print(f"  inject={inject:>4.1f} | mean = {mean:.2f} ± {std:.2f}")

# Probe 2: finer inject scan at small-budget
print("\n=== Probe 2: finer inject scan (0.0/0.05/0.1/0.15/0.2) at h=16/ep=20 ===")
probe2_results = {}
for inject in [0.0, 0.05, 0.1, 0.15, 0.2]:
    per_seed = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        ds = EmmaRoverRegressionDataset(num_samples=200, window=16, feature_noise_std=0.02, seed=seed)
        tl, _, te = create_emma_rover_dataloaders(ds, batch_size=32, seed=seed)
        model = CrossModalAttnBiCfCNADWithMDN(
            video_dim=3, audio_dim=1, hidden_size=16,
            output_size=5, num_mixtures=1,
        ).to(device)
        mse = train_eval(model, tl, te, inject, epochs=20)
        per_seed.append(mse)
        print(f"  inject={inject:>4.2f} | seed={seed:>3d} | MSE={mse:>8.4f}")
    mean = sum(per_seed) / len(per_seed)
    std = (sum((m - mean)**2 for m in per_seed) / max(1, len(per_seed)-1)) ** 0.5
    probe2_results[f"inject{inject}"] = {"inject": inject, "per_seed_mse": per_seed, "mean": mean, "std": std}
    print(f"  inject={inject:>4.2f} | mean = {mean:.2f} ± {std:.2f}")

# Save
out = {
    "config": {"lr": LR, "seeds": SEEDS, "test_sigma": 0.0},
    "probe1_large_budget": {
        "description": "h=64, ep=80, test_sigma=0.0, inject in {0.0, 0.1}",
        "results": probe1_results,
    },
    "probe2_finer_inject": {
        "description": "h=16, ep=20, test_sigma=0.0, inject in {0.0, 0.05, 0.1, 0.15, 0.2}",
        "results": probe2_results,
    },
    "metadata": {
        "round": 55,
        "follows_up": "round 54 noise injection production recipe (34th meta-refinement)",
        "hypotheses": {
            "H_a": "inject=0.1 wins at h=64/ep=80 (recipe is budget-invariant)",
            "H_b": "0.1 is the optimal inject (V-shape or U-shape around 0.1)",
        },
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = ROOT / "analysis" / "emma_rover" / f"{now}_inject_recipe_large_budget.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")

# Verdict
print("\n=== Verdict ===")
print(f"Probe 1 (h=64/ep=80):")
for k, v in probe1_results.items():
    print(f"  inject={v['inject']:>4.1f} -> MSE {v['mean']:.2f} ± {v['std']:.2f}")
if probe1_results['inject0.1']['mean'] < probe1_results['inject0.0']['mean']:
    print(f"  -> inject=0.1 WINS at large-budget (delta = {probe1_results['inject0.1']['mean'] - probe1_results['inject0.0']['mean']:+.2f})")
else:
    print(f"  -> inject=0.0 wins at large-budget (delta = {probe1_results['inject0.1']['mean'] - probe1_results['inject0.0']['mean']:+.2f})")

print(f"\nProbe 2 (finer inject scan):")
for k, v in probe2_results.items():
    print(f"  inject={v['inject']:>4.2f} -> MSE {v['mean']:.2f} ± {v['std']:.2f}")
best = min(probe2_results.values(), key=lambda v: v['mean'])
print(f"  -> BEST: inject={best['inject']:.2f} MSE {best['mean']:.2f}")
