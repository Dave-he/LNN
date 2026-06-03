"""Active noise injection as data augmentation — verify the 33rd
meta-conclusion's production recipe.

Round 53 (15 runs) discovered a_feat magnitude peaks at sigma=0.1
-> lowest MSE 479. This suggests that for clean-audio training data,
ACTIVELY INJECTING sigma=0.1 noise during training (as data
augmentation) should yield the a_feat peak effect.

Probe design:
  3 conditions x 5 test_sigmas x 3 seeds = 45 runs
  - condition A: no injection (baseline, matches round 53)
  - condition B: inject sigma=0.1 during training, test at varying sigma
  - condition C: inject sigma=0.5 during training, test at varying sigma

Hypotheses (falsifiable):
  H_a: condition B yields MSE 479 REGARDLESS of test sigma
       -> noise injection is a true production recipe
  H_b: condition B is best only at test_sigma=0.0
       -> injection helps generalize to clean test
  H_c: condition C is better than condition B at test_sigma=0.5+
       -> injection matches test distribution
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
EPOCHS = 20
HIDDEN = 16
LR = 5e-3
SEEDS = [1, 2, 3]
TEST_SIGMAS = [0.0, 0.1, 0.5, 1.0, 2.0]
INJECT_CONDITIONS = [0.0, 0.1, 0.5]  # inject sigma during training

def audio_apply(audio, sigma):
    if sigma == 0: return audio
    return audio + torch.randn_like(audio) * sigma

def train(model, tl, inject_sigma):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
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

def eval_mse(model, te, test_sigma):
    model.eval()
    sq = []
    with torch.no_grad():
        for batch, target in te:
            target = {k: v.to(device) for k, v in target.items()}
            video = batch["video"].to(device)
            audio = batch.get("audio")
            if audio is not None: audio = audio.to(device)
            audio = audio_apply(audio, test_sigma)
            out = model(video, audio)
            final = {k: v[:, -1] for k, v in out.items()}
            mean = mdn_mean(final)
            sq.append((mean - target["params"]).pow(2).sum(dim=-1))
    return float(torch.cat(sq).mean().item())

print("=== Noise Injection as Data Augmentation Probe (round 54) ===")
print(f"epochs={EPOCHS} hidden={HIDDEN} seeds={SEEDS}")
print(f"inject conditions={INJECT_CONDITIONS} test_sigmas={TEST_SIGMAS}")

results = {}
for inject_sigma in INJECT_CONDITIONS:
    for test_sigma in TEST_SIGMAS:
        per_seed = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            ds = EmmaRoverRegressionDataset(num_samples=200, window=16, feature_noise_std=0.02, seed=seed)
            tl, _, te = create_emma_rover_dataloaders(ds, batch_size=32, seed=seed)
            model = CrossModalAttnBiCfCNADWithMDN(
                video_dim=3, audio_dim=1, hidden_size=HIDDEN,
                output_size=5, num_mixtures=1,
            ).to(device)
            train(model, tl, inject_sigma)
            mse = eval_mse(model, te, test_sigma)
            per_seed.append(mse)
            print(f"  inject={inject_sigma:>4.1f} | test_sigma={test_sigma:>4.1f} | seed={seed:>3d} | MSE={mse:>8.4f}")
        mean = sum(per_seed) / len(per_seed)
        std = (sum((m - mean)**2 for m in per_seed) / max(1, len(per_seed)-1)) ** 0.5
        key = f"inject{inject_sigma}__test{test_sigma}"
        results[key] = {
            "inject_sigma": inject_sigma, "test_sigma": test_sigma,
            "per_seed_mse": per_seed, "mean_mse": mean, "std_mse": std,
        }
        print(f"  inject={inject_sigma:>4.1f} | test_sigma={test_sigma:>4.1f} | mean={mean:.2f}±{std:.2f}")

# Per-inject summary
print("\n=== Per-inject-condition summary (mean over test_sigmas) ===")
for inject_sigma in INJECT_CONDITIONS:
    mses = [results[f"inject{inject_sigma}__test{ts}"]["mean_mse"] for ts in TEST_SIGMAS]
    print(f"  inject={inject_sigma:>4.1f} | mean over test_sigmas = {sum(mses)/len(mses):.2f}")

# Per-test summary
print("\n=== Per-test-sigma summary (mean over inject conditions) ===")
for test_sigma in TEST_SIGMAS:
    mses = [results[f"inject{is_}__test{test_sigma}"]["mean_mse"] for is_ in INJECT_CONDITIONS]
    print(f"  test_sigma={test_sigma:>4.1f} | mean over injects = {sum(mses)/len(mses):.2f}")

out = {
    "config": {"epochs": EPOCHS, "hidden_size": HIDDEN, "lr": LR, "seeds": SEEDS,
               "inject_conditions": INJECT_CONDITIONS, "test_sigmas": TEST_SIGMAS},
    "results": results,
    "metadata": {
        "round": 54,
        "follows_up": "round 53 a_feat magnitude peaks at sigma=0.1 (33rd meta-refinement)",
        "hypotheses": {
            "H_a": "inject=0.1 yields MSE ~479 regardless of test_sigma",
            "H_b": "inject=0.1 best at test_sigma=0.0 (helps generalize to clean)",
            "H_c": "inject=0.5 best at test_sigma=0.5+ (matches test distribution)",
        },
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = ROOT / "analysis" / "emma_rover" / f"{now}_noise_injection.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")
