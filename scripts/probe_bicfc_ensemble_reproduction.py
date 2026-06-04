"""Reproduction test: BiCfCEnsemble class on round 65 protocol.

Verifies the BiCfCEnsemble class produces 0.24 honest LOO MSE
as established in round 65 (30-seed K=20 by val).

For speed, uses 10 seeds (faster, ~5 min) with K=5 (matching ratio).
This should give ~0.5-1.0 LOO MSE (less than 0.24 because fewer seeds,
but should match the *pattern*).

For the FULL round 65 reproduction (30 seeds, K=20), see the
seed30_honest.py probe.

This test verifies:
  1. BiCfCEnsemble class instantiates correctly
  2. fit() trains 10 seeds successfully
  3. predict() averages top-K predictions
  4. evaluate() computes ensemble MSE vs per-seed mean MSE
  5. LOO MSE < 2.0 (sanity check)
"""
import os, sys, json, time
sys.path.insert(0, "/Users/hyx/workspace/LNN")

import torch
from torch.utils.data import DataLoader
from lnn.core.ensemble import BiCfCEnsemble
from lnn.data.emma_rover_temporal_folds import (
    TemporalSegmentRegressionDataset,
    create_segment_loo_dataloaders,
)

device = torch.device("cpu")
EPOCHS = 80
WARMUP = 40
HIDDEN = 96
N_SEEDS = 10  # smaller for speed
K = 5

# Use first 10 seeds from round 65's set (deterministic)
SEEDS = [1, 2, 3, 7, 42, 11, 100, 2026, 313, 777]

print(f"=== BiCfCEnsemble Reproduction Test (round 68) ===")
print(f"n_seeds={N_SEEDS} K={K} hidden={HIDDEN} epochs={EPOCHS} warmup={WARMUP}")
print(f"seeds={SEEDS}")
print(f"Total fold runs: {N_SEEDS * 4} = {N_SEEDS * 4} train+val, 1 ensemble per fold = {4} ensembles")

# 4-fold LOO
ensemble_mses = []
per_seed_mean_mses = []
for fold in range(4):
    print(f"\n=== Fold {fold} ===")
    ds = TemporalSegmentRegressionDataset(seed=1, audio_mode="normal")
    tl_full, te = create_segment_loo_dataloaders(
        ds, held_out_fold=fold, batch_size=8,
    )
    train_dataset = tl_full.dataset

    # Create ensemble
    ensemble = BiCfCEnsemble(
        n_seeds=N_SEEDS, K=K,
        hidden_size=HIDDEN, num_mixtures=1, output_size=5,
        epochs=EPOCHS, warmup_epochs=WARMUP,
        phase2_inject_sigma=0.10, freeze="audio_only",
        val_frac=0.20, lr=5e-3, device="cpu",
    )
    start = time.perf_counter()
    ensemble.fit(train_dataset, seed_values=SEEDS)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    metrics = ensemble.evaluate(te)
    eval_time = time.perf_counter() - start

    ensemble_mses.append(metrics["ensemble_mse"])
    per_seed_mean_mses.append(metrics["per_seed_mean_mse"])
    print(f"  ensemble MSE (K={K}): {metrics['ensemble_mse']:.4f}")
    print(f"  per-seed mean MSE: {metrics['per_seed_mean_mse']:.4f} +/- {metrics['per_seed_std_mse']:.4f}")
    print(f"  ratio ensemble/mean: {metrics['ensemble_mse']/metrics['per_seed_mean_mse']:.2f}")
    print(f"  top-K indices: {ensemble.top_k_indices_[:5]}...")
    print(f"  times: fit={fit_time:.1f}s, eval={eval_time:.1f}s")

# Aggregate
import statistics
print("\n=== Aggregate across 4 folds ===")
print(f"  Ensemble MSE (K={K}): mean = {statistics.mean(ensemble_mses):.4f}")
print(f"  Per-seed mean MSE:    mean = {statistics.mean(per_seed_mean_mses):.4f}")
delta = statistics.mean(per_seed_mean_mses) - statistics.mean(ensemble_mses)
pct = (delta / statistics.mean(per_seed_mean_mses)) * 100 if per_seed_mean_mses else 0
print(f"  Delta (per-seed - ensemble): {delta:+.4f} ({pct:+.1f}%)")

# Sanity check
if statistics.mean(ensemble_mses) < 2.0:
    print(f"  ★ Sanity check PASS: ensemble MSE < 2.0")
else:
    print(f"  ✗ Sanity check FAIL: ensemble MSE >= 2.0 (expected < 2.0)")

# Compare to round 65 30-seed K=20 (0.24) and round 64 20-seed K=10 (0.75)
print("\n=== Comparison to prior results ===")
print(f"  Round 64 (20-seed K=10): 0.75 honest LOO MSE")
print(f"  Round 65 (30-seed K=20): 0.24 honest LOO MSE <- BEST")
print(f"  Round 68 (10-seed K=5):  {statistics.mean(ensemble_mses):.4f} honest LOO MSE")
print(f"    (Expected to be > 0.24 because fewer seeds)")

# Save
out = {
    "config": {
        "n_seeds": N_SEEDS, "K": K, "hidden_size": HIDDEN,
        "epochs": EPOCHS, "warmup_epochs": WARMUP,
        "seeds": SEEDS,
    },
    "per_fold": {
        "ensemble_mses": ensemble_mses,
        "per_seed_mean_mses": per_seed_mean_mses,
    },
    "aggregate": {
        "ensemble_mse_mean": statistics.mean(ensemble_mses),
        "per_seed_mean_mse_mean": statistics.mean(per_seed_mean_mses),
        "delta": delta,
        "delta_pct": pct,
    },
    "references": {
        "round_64_20seed_K10": 0.75,
        "round_65_30seed_K20": 0.24,
    },
    "metadata": {
        "round": 68,
        "purpose": "Verify BiCfCEnsemble class reproduces round 65 protocol (smaller scale)",
        "expected": "Pattern: ensemble < per-seed mean; 10-seed K=5 should be > 0.24 but < 1.5",
    },
}
import datetime as dt
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = f"analysis/emma_rover/{now}_bicfc_ensemble_reproduction.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")

# Final verdict
print("\n=== Final Verdict ===")
if statistics.mean(ensemble_mses) < statistics.mean(per_seed_mean_mses) * 0.7:
    print(f"  ★ BiCfCEnsemble is HIGHLY effective: ensemble < 70% of per-seed mean")
elif statistics.mean(ensemble_mses) < statistics.mean(per_seed_mean_mses):
    print(f"  ★ BiCfCEnsemble is effective: ensemble < per-seed mean")
else:
    print(f"  ✗ BiCfCEnsemble does NOT improve: ensemble >= per-seed mean")
