"""Direct attention weight dump — the smoking gun for the cross-attn
sigma-switch mechanism.

Round 50 (frozen_xattn) and round 51 (cell branches invariant) together
prove cross-attn IS the mechanism but its sub-modules (q/k/v) propagate
sigma-sensitivity *through the attention weights*.

This script captures attn_va (video queries audio) and attn_av
(audio queries video) attention weight distributions at each sigma.

Hypothesis: the attention weight ENTROPY increases with sigma (more
uniform) -> softer attention -> regularization -> MSE 479 at sigma=0.1
vs MSE 581 at sigma=0.0 (overfit sharp attention).

3 seeds x 5 sigmas = 15 runs.
"""
import os, sys, json, datetime as dt, pathlib
sys.path.insert(0, "/Users/hyx/workspace/LNN")

import torch
import torch.nn.functional as F
from lnn.core.multimodal_physreg import CrossModalAttnBiCfCNADWithMDN
from lnn.core.mdn import mdn_mean, mdn_negative_log_likelihood
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

def entropy(attn):
    """Mean entropy of attention distribution over key dim."""
    # attn shape: (B, H_q, H_k) typically; or (B*H_q, H_k) post-softmax
    eps = 1e-9
    return -(attn * torch.log(attn + eps)).sum(dim=-1).mean().item()

def train_eval(model, tl, te, sigma):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        model.train()
        for batch, target in tl:
            target = {k: v.to(device) for k, v in target.items()}
            video = batch["video"].to(device)
            audio = batch.get("audio")
            if audio is not None: audio = audio.to(device)
            audio = audio_apply(audio, sigma)
            opt.zero_grad()
            out = model(video, audio, return_attention=True)
            final = {k: v[:, -1] for k, v in out.items() if not k.startswith("_")}
            loss = mdn_negative_log_likelihood(final, target["params"])
            loss.backward()
            opt.step()
    # Eval with attention
    model.eval()
    attn_va_entropy, attn_av_entropy = [], []
    sq = []
    with torch.no_grad():
        for batch, target in te:
            target = {k: v.to(device) for k, v in target.items()}
            video = batch["video"].to(device)
            audio = batch.get("audio")
            if audio is not None: audio = audio.to(device)
            audio = audio_apply(audio, sigma)
            out = model(video, audio, return_attention=True)
            final = {k: v[:, -1] for k, v in out.items() if not k.startswith("_")}
            mean = mdn_mean(final)
            sq.append((mean - target["params"]).pow(2).sum(dim=-1))
            if "_attn_video_queries_audio" in out:
                attn_va_entropy.append(entropy(out["_attn_video_queries_audio"]))
            if "_attn_audio_queries_video" in out:
                attn_av_entropy.append(entropy(out["_attn_audio_queries_video"]))
    test_mse = float(torch.cat(sq).mean().item())
    return test_mse, attn_va_entropy, attn_av_entropy

print("=== Cross-Attention Weight Distribution Dump Probe (round 52) ===")
print(f"epochs={EPOCHS} hidden={HIDDEN} seeds={SEEDS} sigmas={SIGMAS}")

results = {}
for sigma in SIGMAS:
    for seed in SEEDS:
        torch.manual_seed(seed)
        ds = EmmaRoverRegressionDataset(num_samples=200, window=16, feature_noise_std=0.02, seed=seed)
        tl, _, te = create_emma_rover_dataloaders(ds, batch_size=32, seed=seed)
        model = CrossModalAttnBiCfCNADWithMDN(
            video_dim=3, audio_dim=1, hidden_size=HIDDEN,
            output_size=5, num_mixtures=1,
        ).to(device)
        test_mse, attn_va_ent, attn_av_ent = train_eval(model, tl, te, sigma)
        va_ent_mean = sum(attn_va_ent) / len(attn_va_ent) if attn_va_ent else None
        av_ent_mean = sum(attn_av_ent) / len(attn_av_ent) if attn_av_ent else None
        key = f"sigma{sigma}__seed{seed}"
        results[key] = {
            "sigma": sigma, "seed": seed, "test_mse": test_mse,
            "attn_va_entropy_mean": va_ent_mean,
            "attn_av_entropy_mean": av_ent_mean,
            "attn_va_entropy_all": attn_va_ent,
            "attn_av_entropy_all": attn_av_ent,
        }
        va_str = f"{va_ent_mean:.3f}" if va_ent_mean is not None else "N/A"
        av_str = f"{av_ent_mean:.3f}" if av_ent_mean is not None else "N/A"
        print(f"  sigma={sigma:>4.1f} | seed={seed:>3d} | test MSE = {test_mse:>8.4f} | attn_va_H={va_str} | attn_av_H={av_str}")

print("\n=== Per-sigma aggregate ===")
per_sigma = {}
for sigma in SIGMAS:
    va_ents, av_ents, mses = [], [], []
    for seed in SEEDS:
        r = results[f"sigma{sigma}__seed{seed}"]
        if r["attn_va_entropy_mean"] is not None:
            va_ents.append(r["attn_va_entropy_mean"])
        if r["attn_av_entropy_mean"] is not None:
            av_ents.append(r["attn_av_entropy_mean"])
        mses.append(r["test_mse"])
    per_sigma[str(sigma)] = {
        "attn_va_entropy_mean": sum(va_ents) / len(va_ents) if va_ents else None,
        "attn_av_entropy_mean": sum(av_ents) / len(av_ents) if av_ents else None,
        "mse_mean": sum(mses) / len(mses),
    }
    va_s = f"{per_sigma[str(sigma)]['attn_va_entropy_mean']:.4f}" if per_sigma[str(sigma)]['attn_va_entropy_mean'] else "N/A"
    av_s = f"{per_sigma[str(sigma)]['attn_av_entropy_mean']:.4f}" if per_sigma[str(sigma)]['attn_av_entropy_mean'] else "N/A"
    print(f"  sigma={sigma:>4.1f} | attn_va_H={va_s} | attn_av_H={av_s} | MSE={per_sigma[str(sigma)]['mse_mean']:.2f}")

out = {
    "config": {"epochs": EPOCHS, "hidden_size": HIDDEN, "lr": LR, "seeds": SEEDS, "sigmas": SIGMAS},
    "results": results,
    "per_sigma_aggregate": per_sigma,
    "metadata": {
        "round": 52,
        "follows_up": "round 50 cross-attn ablation (located mechanism in cross-attn) + round 51 cell branches invariant",
        "hypothesis": "Attention entropy increases with sigma -> regularization -> MSE drop",
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = ROOT / "analysis" / "emma_rover" / f"{now}_attn_weights_dump.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")
