"""Audio features magnitude probe — locate the sigma-sensitivity
that causes the switch, *given* that attention weights are uniform.

Round 52 (15 runs attn weights dump) showed attention entropy is
exactly ln(16) = 2.77 nats (MAXIMUM entropy for h=16). This means
attn is uniform regardless of sigma. So the sigma-sensitivity must
live in:

  audio (noisy) -> audio_encoder (Bi-CfC) -> audio_features
              -> v_a = Linear(audio_features) (BEFORE softmax)
              -> v_from_a = attn_va @ v_a (uniform attn means average)
              -> v_refined = v_feat + v_from_a

The MAGNITUDE of v_from_a is what changes. v_from_a magnitude
= (1/N) * sum(v_a[i] for i in 1..N) magnitude = (1/sqrt(N)) * mean(v_a)
since v_a are i.i.d.

This script captures the magnitude of:
  1. audio features (audio_encoder output, before k_a/v_a projections)
  2. v_a (after v_a Linear projection, before attention)
  3. v_from_a (after cross-attention)
  4. v_refined (v_feat + v_from_a)
  5. fused (final input to mdn)

Hypothesis: at least one of these magnitudes shifts with sigma,
explaining the MSE switch.
"""
import os, sys, json, datetime as dt, pathlib
sys.path.insert(0, "/Users/hyx/workspace/LNN")

import torch
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
            out = model(video, audio)
            final = {k: v[:, -1] for k, v in out.items()}
            loss = mdn_negative_log_likelihood(final, target["params"])
            loss.backward()
            opt.step()
    # Eval with internal magnitude captures
    model.eval()
    magnitudes = {
        "audio": [], "audio_features": [], "v_a": [], "k_a": [],
        "v_from_a": [], "v_refined": [], "fused": [],
    }
    sq = []
    with torch.no_grad():
        for batch, target in te:
            target = {k: v.to(device) for k, v in target.items()}
            video = batch["video"].to(device)
            audio = batch.get("audio")
            if audio is not None: audio = audio.to(device)
            audio = audio_apply(audio, sigma)
            # Manual cross-attn forward to capture intermediate magnitudes
            v_feat = model.video_encoder(video)
            a_feat = model.audio_encoder(audio)
            k_a = model.k_a(a_feat)
            v_a = model.v_a(a_feat)
            k_v = model.k_v(v_feat)
            q_v = model.q_v(v_feat)
            q_a = model.q_a(a_feat)
            v_v = model.v_v(v_feat)
            # attention
            attn_va = torch.softmax(torch.bmm(q_v, k_a.transpose(1, 2)) / (HIDDEN ** 0.5), dim=-1)
            attn_av = torch.softmax(torch.bmm(q_a, k_v.transpose(1, 2)) / (HIDDEN ** 0.5), dim=-1)
            v_from_a = torch.bmm(attn_va, v_a)
            a_from_v = torch.bmm(attn_av, v_v)
            v_refined = v_feat + v_from_a
            a_refined = a_feat + a_from_v
            fused = model.fuse_proj(torch.cat([v_refined, a_refined], dim=-1))
            mdn_out = model.mdn(fused)
            mean = mdn_mean({k: v[:, -1] for k, v in mdn_out.items()})
            sq.append((mean - target["params"]).pow(2).sum(dim=-1))
            # Magnitudes (mean over batch and time)
            magnitudes["audio"].append(float(audio.abs().mean().item()))
            magnitudes["audio_features"].append(float(a_feat.abs().mean().item()))
            magnitudes["k_a"].append(float(k_a.abs().mean().item()))
            magnitudes["v_a"].append(float(v_a.abs().mean().item()))
            magnitudes["v_from_a"].append(float(v_from_a.abs().mean().item()))
            magnitudes["v_refined"].append(float(v_refined.abs().mean().item()))
            magnitudes["fused"].append(float(fused.abs().mean().item()))
    test_mse = float(torch.cat(sq).mean().item())
    agg = {k: sum(v) / len(v) for k, v in magnitudes.items()}
    return test_mse, agg

print("=== Audio Features Magnitude Probe (round 53) ===")
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
        test_mse, mag = train_eval(model, tl, te, sigma)
        key = f"sigma{sigma}__seed{seed}"
        results[key] = {"sigma": sigma, "seed": seed, "test_mse": test_mse, "magnitudes": mag}
        m = mag
        print(
            f"  sigma={sigma:>4.1f} | seed={seed:>3d} | MSE={test_mse:>8.4f} | "
            f"audio={m['audio']:.3f} a_feat={m['audio_features']:.3f} "
            f"k_a={m['k_a']:.3f} v_a={m['v_a']:.3f} v_from_a={m['v_from_a']:.3f} "
            f"fused={m['fused']:.3f}"
        )

print("\n=== Per-sigma aggregate (mean over seeds) ===")
per_sigma = {}
for sigma in SIGMAS:
    keys = ["audio", "audio_features", "k_a", "v_a", "v_from_a", "v_refined", "fused"]
    agg = {k: [] for k in keys}
    mses = []
    for seed in SEEDS:
        r = results[f"sigma{sigma}__seed{seed}"]
        for k in keys:
            agg[k].append(r["magnitudes"][k])
        mses.append(r["test_mse"])
    per_sigma[str(sigma)] = {
        **{k: sum(v) / len(v) for k, v in agg.items()},
        "mse_mean": sum(mses) / len(mses),
    }
    p = per_sigma[str(sigma)]
    print(
        f"  sigma={sigma:>4.1f} | MSE={p['mse_mean']:.2f} | "
        f"audio={p['audio']:.3f} a_feat={p['audio_features']:.3f} "
        f"k_a={p['k_a']:.3f} v_a={p['v_a']:.3f} v_from_a={p['v_from_a']:.3f} "
        f"v_refined={p['v_refined']:.3f} fused={p['fused']:.3f}"
    )

out = {
    "config": {"epochs": EPOCHS, "hidden_size": HIDDEN, "lr": LR, "seeds": SEEDS, "sigmas": SIGMAS},
    "results": results,
    "per_sigma_aggregate": per_sigma,
    "metadata": {
        "round": 53,
        "follows_up": "round 52 attn weights dump (entropy max=ln(16), uniform attn)",
        "hypothesis": "v_a or v_from_a magnitude shifts with sigma, explaining the switch",
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = ROOT / "analysis" / "emma_rover" / f"{now}_audio_features_magnitude.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")
