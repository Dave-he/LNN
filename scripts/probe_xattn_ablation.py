"""Cross-attention ablation: does freezing the 6 attention linear layers
remove the sigma-switch?

Hypothesis:
  - If frozen cross-attn STILL shows sigma-switch (sigma=0 -> 581, sigma=0.1 -> 478),
    then mechanism is NOT cross-attn; it's in f_gate/g_branch/h_branch
  - If frozen cross-attn REMOVES sigma-switch (sigma=0 and 0.1 give same MSE),
    then cross-attn IS the mechanism

Run: 2 conditions (normal vs frozen_xattn) x 5 sigmas x 3 seeds = 30 runs
"""
import os, sys, time, json, datetime as dt, pathlib
sys.path.insert(0, "/Users/hyx/workspace/LNN")

import torch
import torch.nn as nn
from lnn.core.mdn import mdn_mean, mdn_negative_log_likelihood
from lnn.core.multimodal_physreg import CrossModalAttnBiCfCNADWithMDN
from lnn.data.emma_rover_regression import (
    EmmaRoverRegressionDataset, create_emma_rover_dataloaders,
)

ROOT = pathlib.Path("/Users/hyx/workspace/LNN")
device = torch.device("cpu")
EPOCHS = 20
HIDDEN = 16
LR = 5e-3
SEEDS = [1, 2, 3]
SIGMAS = [0.0, 0.1, 0.5, 1.0, 2.0]

def audio_apply(audio, sigma):
    if sigma == 0: return audio
    return audio + torch.randn_like(audio) * sigma

def freeze_xattn(model):
    """Freeze the 6 cross-attention linear layers to their init values."""
    for attr in ['q_v', 'k_a', 'v_a', 'q_a', 'k_v', 'v_v']:
        proj = getattr(model, attr)
        # Replace with identity-like: zero weights, zero bias => output is 0
        # Better: replace with constant random small weights
        with torch.no_grad():
            proj.weight.zero_()
            proj.bias.zero_()
        for p in proj.parameters():
            p.requires_grad = False

def train(model, tl, sigma):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for _ in range(EPOCHS):
        model.train()
        for batch, target in tl:
            target = {k: v.to(device) for k, v in target.items()}
            video = batch["video"].to(device)
            audio = batch.get("audio")
            if audio is not None: audio = audio.to(device)
            audio = audio_apply(audio, sigma)
            opt.zero_grad()
            out = model(video, audio)
            final = {k: v[:, -1] for k, v in out.items()}
            loss = mdn_negative_log_likelihood(final, target["params"])
            loss.backward()
            opt.step()

def eval_mse(model, te, sigma):
    model.eval()
    sq = []
    with torch.no_grad():
        for batch, target in te:
            target = {k: v.to(device) for k, v in target.items()}
            video = batch["video"].to(device)
            audio = batch.get("audio")
            if audio is not None: audio = audio.to(device)
            audio = audio_apply(audio, sigma)
            out = model(video, audio)
            final = {k: v[:, -1] for k, v in out.items()}
            mean = mdn_mean(final)
            sq.append((mean - target["params"]).pow(2).sum(dim=-1))
    return float(torch.cat(sq).mean().item())

print("=== Cross-Attention Ablation Probe (round 50) ===")
print(f"epochs={EPOCHS} hidden={HIDDEN} seeds={SEEDS} sigmas={SIGMAS}")

results = {}
for condition in ["normal_xattn", "frozen_xattn"]:
    for sigma in SIGMAS:
        per_seed = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            ds = EmmaRoverRegressionDataset(num_samples=200, window=16, feature_noise_std=0.02, seed=seed)
            tl, _, te = create_emma_rover_dataloaders(ds, batch_size=32, seed=seed)
            model = CrossModalAttnBiCfCNADWithMDN(
                video_dim=3, audio_dim=1, hidden_size=HIDDEN,
                output_size=5, num_mixtures=1,
            ).to(device)
            if condition == "frozen_xattn":
                freeze_xattn(model)
            train(model, tl, sigma)
            mse = eval_mse(model, te, sigma)
            per_seed.append(mse)
            print(f"  {condition:14s} | sigma={sigma:>4.1f} | seed={seed:>3d} | test MSE = {mse:.4f}")
        mean = sum(per_seed) / len(per_seed)
        std = (sum((m - mean)**2 for m in per_seed) / max(1, len(per_seed)-1)) ** 0.5
        results[f"{condition}__sigma{sigma}"] = {
            "condition": condition, "sigma": sigma, "per_seed": per_seed,
            "mean": mean, "std": std,
        }
        print(f"  {condition:14s} | sigma={sigma:>4.1f} | mean = {mean:.4f} ± {std:.4f}")

# Save
out = {
    "config": {"epochs": EPOCHS, "hidden_size": HIDDEN, "lr": LR,
               "seeds": SEEDS, "sigmas": SIGMAS, "model": "CrossModalAttnBiCfCNADWithMDN"},
    "results": results,
    "metadata": {
        "round": 50,
        "follows_up": "round 49 NAD retain visualization (REFUTED NAD hypothesis)",
        "hypothesis": "if frozen_xattn removes sigma-switch, cross-attn IS the mechanism",
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = ROOT / "analysis" / "emma_rover" / f"{now}_xattn_ablation.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")

# Compare
print("\n=== Comparison: normal_xattn vs frozen_xattn ===")
for sigma in SIGMAS:
    n = results[f"normal_xattn__sigma{sigma}"]["mean"]
    f_ = results[f"frozen_xattn__sigma{sigma}"]["mean"]
    print(f"  sigma={sigma:>4.1f} | normal={n:.2f} | frozen={f_:.2f} | delta={f_-n:+.2f}")

# Sigma-switch analysis
print("\n=== Sigma-switch analysis ===")
for cond in ["normal_xattn", "frozen_xattn"]:
    s0 = results[f"{cond}__sigma0.0"]["mean"]
    s01 = results[f"{cond}__sigma0.1"]["mean"]
    print(f"  {cond:14s} | sigma=0.0: {s0:.2f} | sigma=0.1: {s01:.2f} | switch delta: {s01-s0:+.2f}")
