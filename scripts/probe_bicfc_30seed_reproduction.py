"""BiCfCEnsemble 完整 30-seed K=20 reproduction probe (round 70).

Verifies the BiCfCEnsemble class reproduces the round 65 0.24 LOO MSE
with the FULL 30-seed K=20 v15 FINAL recipe.

Round 65: 30 seeds × 4 folds = 120 fold runs (~18 min)
- Used explicit training loops (not BiCfCEnsemble class)
- Got honest LOO MSE = 0.24 (NEW BEST reproducible)

Round 70 (this): same recipe via BiCfCEnsemble class
- Should reproduce 0.24
- If yes: BiCfCEnsemble is fully validated
- If no: there's a bug or difference

Probe: 30 seeds × 4 folds = 120 fold runs (~25 min via class)
"""
import os, sys, json, time, datetime as dt, pathlib
sys.path.insert(0, "/Users/hyx/workspace/LNN")

import torch
from torch.utils.data import DataLoader
from lnn.core.ensemble import BiCfCEnsemble
from lnn.data.emma_rover_temporal_folds import (
    TemporalSegmentRegressionDataset,
    create_segment_loo_dataloaders,
)

# 30 seeds from round 65
SEEDS = [
    1, 2, 3, 7, 42,
    11, 100, 2026, 313, 777,
    55, 99, 314, 555, 888,
    1024, 2027, 3141, 4242, 9999,
    17, 88, 256, 512, 1023, 2048, 4096, 8192, 16384, 32768,
]
N_FOLDS = 4

print("=== BiCfCEnsemble 30-seed K=20 reproduction (round 70) ===")
print(f"Seeds: {len(SEEDS)}")
print(f"Folds: {N_FOLDS}")
print(f"Total: {len(SEEDS) * N_FOLDS} = 120 fold runs (~25 min via class)")

ensemble_mses = []
per_seed_mean_mses = []

for fold in range(N_FOLDS):
    print(f"\n=== Fold {fold} ===")
    ds = TemporalSegmentRegressionDataset(seed=1, audio_mode="normal")
    tl_full, te = create_segment_loo_dataloaders(
        ds, held_out_fold=fold, batch_size=8,
    )
    train_dataset = tl_full.dataset

    # Instantiate ensemble with v15 recipe defaults
    ensemble = BiCfCEnsemble(
        n_seeds=30, K=20, hidden_size=96,
        num_mixtures=1, output_size=5,
        epochs=80, warmup_epochs=40,
        phase2_inject_sigma=0.10, freeze="audio_only",
        val_frac=0.20, lr=5e-3, device="cpu",
    )
    print(f"  Fitting 30 seeds on fold {fold} (this takes ~5 min)...")
    start = time.perf_counter()
    ensemble.fit(train_dataset, seed_values=SEEDS)
    fit_time = time.perf_counter() - start
    print(f"  Fit done in {fit_time:.1f}s ({fit_time/30:.1f}s per seed)")

    start = time.perf_counter()
    metrics = ensemble.evaluate(te)
    eval_time = time.perf_counter() - start
    print(f"  Eval done in {eval_time:.1f}s")
    print(f"  Ensemble MSE (K=20): {metrics['ensemble_mse']:.4f}")
    print(f"  Per-seed mean MSE:    {metrics['per_seed_mean_mse']:.4f} +/- {metrics['per_seed_std_mse']:.4f}")
    print(f"  Top 5 selected seed indices: {ensemble.top_k_indices_[:5]}")
    print(f"  Total fold time: {fit_time + eval_time:.1f}s")

    ensemble_mses.append(metrics['ensemble_mse'])
    per_seed_mean_mses.append(metrics['per_seed_mean_mse'])

# Aggregate
import statistics
agg_ensemble = statistics.mean(ensemble_mses)
agg_per_seed = statistics.mean(per_seed_mean_mses)
delta = agg_per_seed - agg_ensemble
pct = (delta / agg_per_seed) * 100 if agg_per_seed else 0

print("\n=== Aggregate across 4 folds ===")
print(f"  Ensemble MSE (K=20 by val): {agg_ensemble:.4f}")
print(f"  Per-seed mean MSE:           {agg_per_seed:.4f}")
print(f"  Delta (per-seed - ensemble): {delta:+.4f} ({pct:+.1f}%)")
print(f"  Per-fold ensemble MSEs: {[f'{m:.4f}' for m in ensemble_mses]}")

# Compare to round 65 reference
print("\n=== Comparison to prior references ===")
print(f"  Round 65 (30-seed K=20 by val):  0.24 honest LOO MSE <- reference")
print(f"  Round 70 (BiCfCEnsemble class):  {agg_ensemble:.4f}")
delta_from_ref = agg_ensemble - 0.24
print(f"  Delta from round 65: {delta_from_ref:+.4f}")

if abs(delta_from_ref) < 0.1:
    print("  ★ BiCfCEnsemble REPRODUCES round 65 (delta < 0.1)")
elif agg_ensemble < 0.5:
    print("  ✓ BiCfCEnsemble achieves < 0.5 LOO MSE (close to round 65)")
else:
    print(f"  ✗ BiCfCEnsemble achieves {agg_ensemble:.4f} (significantly higher than 0.24)")

# Save
out = {
    "config": {
        "n_seeds": 30, "K": 20, "hidden_size": 96,
        "epochs": 80, "warmup_epochs": 40,
        "phase2_inject_sigma": 0.10, "freeze": "audio_only",
        "val_frac": 0.20, "lr": 5e-3, "device": "cpu",
        "seeds": SEEDS,
    },
    "per_fold": {
        "ensemble_mses": ensemble_mses,
        "per_seed_mean_mses": per_seed_mean_mses,
    },
    "aggregate": {
        "ensemble_mse_mean": agg_ensemble,
        "per_seed_mean_mse_mean": agg_per_seed,
        "delta": delta,
        "delta_pct": pct,
    },
    "references": {
        "round_65_30seed_K20": 0.24,
        "round_68_reproduction_10seed_K5": 0.0496,
    },
    "metadata": {
        "round": 70,
        "purpose": "BiCfCEnsemble 完整 30-seed K=20 reproduction (validates class v15 recipe)",
        "round_65_reference": 0.24,
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = f"analysis/emma_rover/{now}_bicfc_30seed_reproduction.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")

# Verdict
print("\n=== Verdict ===")
if abs(agg_ensemble - 0.24) < 0.1:
    print(f"  ★ BiCfCEnsemble REPRODUCES round 65's 0.24 LOO MSE (delta = {agg_ensemble - 0.24:+.4f})")
    print(f"  ★ v15 PERMANENTIZED fully validated")
else:
    print(f"  ✗ BiCfCEnsemble result differs from round 65 reference")
    print(f"    Round 65: 0.24, Round 70: {agg_ensemble:.4f}")
    print(f"    Investigation needed: could be seed-dependent or class bug")
