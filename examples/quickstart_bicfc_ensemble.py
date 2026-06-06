"""BiCfCEnsemble quickstart example.

This is the canonical usage example for the BiCfCEnsemble class
(see lnn/core/ensemble.py), which implements the v15 FINAL production
recipe from rounds 65-71 of the 65+ round ablation program.

What BiCfCEnsemble does:
  - Trains 30 models with different random seeds
  - Ranks models by validation MSE (smart selection)
  - Returns ensemble of top 20 models' predictions
  - Expected honest LOO MSE: 0.24 (verified round 70 reproduction)

This example shows:
  1. Loading the EMMA rover dataset
  2. Running BiCfCEnsemble with default (v15) parameters
  3. Evaluating on the test set
  4. Customizing the recipe (e.g. fewer seeds for budget-constrained)

Run with: python examples/quickstart_bicfc_ensemble.py
"""
from __future__ import annotations

import statistics

import torch
from torch.utils.data import DataLoader

from lnn.core.ensemble import BiCfCEnsemble
from lnn.data.emma_rover_temporal_folds import (
    TemporalSegmentRegressionDataset,
    create_segment_loo_dataloaders,
)


def quickstart_default():
    """Use BiCfCEnsemble with v15 recipe defaults (30 seeds, K=20)."""
    print("=== BiCfCEnsemble Quickstart (v15 recipe defaults) ===")
    print("Loading EMMA rover dataset...")
    ds = TemporalSegmentRegressionDataset(seed=1, audio_mode="normal")

    # 4-fold LOO: use fold 0 as test, rest as train
    tl_full, te = create_segment_loo_dataloaders(ds, held_out_fold=0, batch_size=8)
    train_dataset = tl_full.dataset

    # BiCfCEnsemble with v15 defaults (30 seeds + K=20 by val + phase2 inject)
    print("Instantiating BiCfCEnsemble with v15 defaults...")
    ensemble = BiCfCEnsemble()  # uses 30 seeds, K=20, h=96, ep=80, etc.

    print("Training 30 seeds (this takes ~5 min)...")
    ensemble.fit(train_dataset)

    print("Evaluating on held-out fold 0...")
    metrics = ensemble.evaluate(te)

    print(f"\nResults:")
    print(f"  Ensemble MSE (K=20): {metrics['ensemble_mse']:.4f}")
    print(f"  Per-seed mean MSE:    {metrics['per_seed_mean_mse']:.4f}")
    print(f"  Per-seed std MSE:     {metrics['per_seed_std_mse']:.4f}")
    print(f"  Top 5 selected seeds: {ensemble.top_k_indices_[:5]}")
    print(f"\nExpected: ensemble_mse ≈ 0.24 (round 70 reproduction)")
    return metrics


def quickstart_budget_constrained():
    """Use BiCfCEnsemble with fewer seeds (5) and smaller K (2) for budget."""
    print("\n=== BiCfCEnsemble Quickstart (BUDGET-CONSTRAINED: 5 seeds, K=2) ===")
    ds = TemporalSegmentRegressionDataset(seed=1, audio_mode="normal")
    tl_full, te = create_segment_loo_dataloaders(ds, held_out_fold=0, batch_size=8)
    train_dataset = tl_full.dataset

    # 5 seeds + K=2: 6× faster training, slightly higher MSE
    ensemble = BiCfCEnsemble(
        n_seeds=5, K=2,
        hidden_size=96,  # keep h=96
        epochs=80, warmup_epochs=40,
        phase2_inject_sigma=0.10, freeze="audio_only",
    )
    print("Training 5 seeds (~50 sec)...")
    ensemble.fit(train_dataset, seed_values=[1, 2, 3, 4, 5])
    metrics = ensemble.evaluate(te)
    print(f"  Ensemble MSE (K=2): {metrics['ensemble_mse']:.4f}")
    print(f"  Per-seed mean MSE:   {metrics['per_seed_mean_mse']:.4f}")
    return metrics


def quickstart_predict_only():
    """Use BiCfCEnsemble for inference after training is done."""
    print("\n=== BiCfCEnsemble Quickstart (PREDICT ONLY) ===")
    # After fit() you can save/load models and predict
    ds = TemporalSegmentRegressionDataset(seed=1, audio_mode="normal")
    tl_full, te = create_segment_loo_dataloaders(ds, held_out_fold=0, batch_size=8)
    train_dataset = tl_full.dataset

    # Train
    ensemble = BiCfCEnsemble(n_seeds=3, K=2, hidden_size=8, epochs=2, warmup_epochs=1)
    ensemble.fit(train_dataset, seed_values=[1, 2, 3])

    # Predict
    preds = ensemble.predict(te)
    print(f"Predictions shape: {preds.shape}")  # (N_samples, output_size=5)
    print(f"First 3 predictions:\n{preds[:3]}")

    # Save models for later use
    # torch.save([m.state_dict() for m in ensemble.models_], "ensemble_models.pt")

    return preds


if __name__ == "__main__":
    # Run quickstart
    print("\n[1/3] Running v15 default quickstart (30 seeds, K=20)...")
    metrics_default = quickstart_default()

    # Budget-constrained example
    print("\n[2/3] Running budget-constrained quickstart (5 seeds, K=2)...")
    metrics_budget = quickstart_budget_constrained()

    # Predict only example
    print("\n[3/3] Running predict-only quickstart (3 seeds, K=2, tiny model)...")
    preds = quickstart_predict_only()

    print("\n=== Summary ===")
    print(f"  v15 default (30 seeds, K=20):    ensemble_mse = {metrics_default['ensemble_mse']:.4f}")
    print(f"  Budget-constrained (5 seeds, K=2): ensemble_mse = {metrics_budget['ensemble_mse']:.4f}")
    print(f"  Predict-only (3 seeds, K=2):        preds shape = {preds.shape}")
    print("\nFor the full 30-seed reproduction (~25 min), see")
    print("  scripts/probe_bicfc_30seed_reproduction.py")
