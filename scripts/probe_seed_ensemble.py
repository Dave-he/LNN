"""Seed-ensemble probe: does averaging predictions across seeds reduce MSE?

Round 60/61 found:
- 5-seed mean MSE: 7.07
- 10-seed mean MSE: 9.98
- 20-seed mean MSE: 11.63 (rising)

But this is "mean of per-seed MSEs". The PRODUCTION-RELEVANT
metric is "MSE of averaged predictions" — for each test sample,
average the predictions from K seeds and compute MSE.

This probe saves per-sample predictions for 20 seeds × 4 folds,
then computes ensemble MSE at K=1, 2, 5, 10, 20.

Hypotheses (falsifiable):
  H_a: ensemble MSE << mean MSE (ensemble is highly effective)
  H_b: ensemble MSE ≈ mean MSE (ensemble doesn't help)
  H_c: ensemble MSE << single-seed best (e.g. < 0.5)
"""
import os, sys, json, datetime as dt, pathlib, time
sys.path.insert(0, "/Users/hyx/workspace/LNN")

import torch
import numpy as np
from lnn.core.mdn import mdn_mean, mdn_negative_log_likelihood
from lnn.core.multimodal_physreg import CrossModalAttnBiCfCNADWithMDN
from lnn.data.emma_rover_temporal_folds import (
    TemporalSegmentRegressionDataset,
    create_segment_loo_dataloaders,
)

ROOT = pathlib.Path("/Users/hyx/workspace/LNN")
device = torch.device("cpu")
HIDDEN = 96
EPOCHS = 80
WARMUP = 40
LR = 5e-3
# 20 seeds (same as round 61)
SEEDS = [
    1, 2, 3, 7, 42,
    11, 100, 2026, 313, 777,
    55, 99, 314, 555, 888,
    1024, 2027, 3141, 4242, 9999,
]
N_FOLDS = 4
INJECT_SIGMA = 0.10
K_VALUES = [1, 2, 5, 10, 20]

def audio_apply(audio, sigma):
    if sigma == 0: return audio
    return audio + torch.randn_like(audio) * sigma

def _move(b, d):
    return {k: v.to(d) for k, v in b.items()}

def train_epoch(model, loader, opt, inject_sigma):
    model.train()
    total, n = 0.0, 0
    for batch, target in loader:
        batch = _move(batch, device)
        target = _move(target, device)
        if "audio" in batch and inject_sigma > 0:
            batch["audio"] = audio_apply(batch["audio"], inject_sigma)
        opt.zero_grad(set_to_none=True)
        out = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in out.items()}
        loss = mdn_negative_log_likelihood(final, target["params"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)

@torch.no_grad()
def collect_predictions(model, loader):
    """Return per-sample predictions and targets as tensors."""
    model.eval()
    preds, tgts = [], []
    for batch, target in loader:
        batch = _move(batch, device)
        target = _move(target, device)
        out = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in out.items()}
        mean = mdn_mean(final)
        preds.append(mean.cpu())
        tgts.append(target["params"].cpu())
    return torch.cat(preds), torch.cat(tgts)

def adaptive_freeze_run(train_loader, test_loader, phase2_sigma, seed):
    torch.manual_seed(seed)
    model = CrossModalAttnBiCfCNADWithMDN(
        video_dim=3, audio_dim=1, hidden_size=HIDDEN,
        output_size=5, num_mixtures=1,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(WARMUP):
        train_epoch(model, train_loader, opt, 0.0)
    for p in model.audio_encoder.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=LR)
    for _ in range(EPOCHS - WARMUP):
        train_epoch(model, train_loader, opt, phase2_sigma)
    return collect_predictions(model, test_loader)

print("=== Seed-ensemble probe (round 62) ===")
print(f"h={HIDDEN} ep={EPOCHS} warmup={WARMUP} inject={INJECT_SIGMA}")
print(f"seeds={len(SEEDS)} folds={N_FOLDS} K_values={K_VALUES}")
print(f"Total: {len(SEEDS) * N_FOLDS} fold runs (~12 min)")

# For each fold, collect predictions from all 20 seeds
# fold_predictions[fold] = list of (preds, targets) per seed
fold_preds_seeds = {}
fold_targets = {}
for fold in range(N_FOLDS):
    print(f"\n=== Fold {fold} ===")
    fold_preds_seeds[fold] = []
    for i, seed in enumerate(SEEDS):
        ds = TemporalSegmentRegressionDataset(seed=seed, audio_mode="normal")
        tl, te = create_segment_loo_dataloaders(
            ds, held_out_fold=fold, batch_size=8,
        )
        start = time.perf_counter()
        try:
            preds, tgts = adaptive_freeze_run(tl, te, INJECT_SIGMA, seed)
        except Exception as e:
            print(f"  ERROR: seed={seed}: {e}")
            continue
        elapsed = time.perf_counter() - start
        fold_preds_seeds[fold].append(preds)
        if fold not in fold_targets:
            fold_targets[fold] = tgts
        print(f"  [{i+1}/{len(SEEDS)}] seed={seed:>5d} | preds shape={list(preds.shape)} | {elapsed:.1f}s")
    print(f"  Fold {fold} collected {len(fold_preds_seeds[fold])} seed predictions")

# Compute per-seed MSE and ensemble MSE
print("\n=== Per-fold results ===")
results = {}
for fold in range(N_FOLDS):
    print(f"\n--- Fold {fold} ---")
    tgts = fold_targets[fold]
    # Per-seed MSE
    per_seed_mse = []
    for preds in fold_preds_seeds[fold]:
        mse = float(((preds - tgts) ** 2).sum(dim=-1).mean().item())
        per_seed_mse.append(mse)
    print(f"  Per-seed MSEs: {[f'{m:.2f}' for m in per_seed_mse]}")
    mean_mse = sum(per_seed_mse) / len(per_seed_mse)
    print(f"  Mean of per-seed MSEs: {mean_mse:.4f}")

    # Ensemble MSE at various K
    ensemble_mse = {}
    for K in K_VALUES:
        if K > len(fold_preds_seeds[fold]):
            continue
        # Use first K seeds (deterministic order)
        avg_preds = torch.stack(fold_preds_seeds[fold][:K], dim=0).mean(dim=0)
        mse = float(((avg_preds - tgts) ** 2).sum(dim=-1).mean().item())
        ensemble_mse[K] = mse
        print(f"  Ensemble MSE (K={K}): {mse:.4f}")
    results[f"fold{fold}"] = {
        "per_seed_mse": per_seed_mse,
        "mean_mse": mean_mse,
        "ensemble_mse": ensemble_mse,
    }

# Aggregate across folds
print("\n=== Aggregate across folds ===")
agg_mean_mse = sum(r["mean_mse"] for r in results.values()) / N_FOLDS
agg_ensemble = {K: sum(r["ensemble_mse"].get(K, 0) for r in results.values()) / N_FOLDS for K in K_VALUES}
print(f"  Mean of per-seed MSEs (avg over 4 folds): {agg_mean_mse:.4f}")
for K, mse in agg_ensemble.items():
    delta = mse - agg_mean_mse
    pct = (delta / agg_mean_mse) * 100 if agg_mean_mse else 0
    marker = "✅ ensemble wins" if mse < agg_mean_mse else "❌ hurts"
    print(f"  Ensemble K={K}: {mse:.4f} | delta = {delta:+.4f} ({pct:+.1f}%) | {marker}")

# Save
out = {
    "config": {"hidden_size": HIDDEN, "epochs": EPOCHS, "warmup_epochs": WARMUP,
               "lr": LR, "seeds": SEEDS, "inject_sigma": INJECT_SIGMA,
               "folds": N_FOLDS, "K_values": K_VALUES,
               "protocol": "TemporalSegmentRegressionDataset 4-fold LOO",
               "model": "CrossModalAttnBiCfCNADWithMDN"},
    "per_fold_results": results,
    "aggregate": {
        "mean_of_per_seed_mses": agg_mean_mse,
        "ensemble_mse_by_K": agg_ensemble,
    },
    "metadata": {
        "round": 62,
        "follows_up": "round 61 (20-seed mean 11.63, 42nd meta-refinement)",
        "hypotheses": {
            "H_a": "ensemble MSE << mean MSE (ensemble highly effective)",
            "H_b": "ensemble MSE ≈ mean MSE (ensemble doesn't help)",
            "H_c": "ensemble MSE << single-seed best",
        },
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = ROOT / "analysis" / "emma_rover" / f"{now}_seed_ensemble.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")

# Verdict
print("\n=== Final Verdict ===")
best_K = min(agg_ensemble.items(), key=lambda x: x[1])
print(f"  Mean of per-seed MSEs: {agg_mean_mse:.4f}")
print(f"  Best ensemble K={best_K[0]}: {best_K[1]:.4f}")
if best_K[1] < agg_mean_mse * 0.7:
    print(f"  ★ Ensemble is HIGHLY effective (>{30}% reduction)")
elif best_K[1] < agg_mean_mse * 0.9:
    print(f"  ★ Ensemble is effective (>{10}% reduction)")
elif best_K[1] < agg_mean_mse:
    print(f"  ★ Ensemble slightly helps (<10% reduction)")
else:
    print(f"  ★ Ensemble does NOT help (or hurts)")
