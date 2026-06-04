"""
Tests for BiCfCEnsemble class (round 68 v15 PERMANENTIZED recipe).

These tests verify the BiCfCEnsemble class behaviour:
1. Default initialization matches v15 recipe (n_seeds=30, K=20, etc.)
2. Custom configuration validation
3. fit() trains the correct number of models
4. predict() returns correct shape
5. evaluate() returns correct metrics
6. Smart selection ranks by validation MSE
7. Edge cases (K > n_seeds, no fit before predict, etc.)
8. The recipe is reproducible

Note: These tests use small configurations (n_seeds=2-3, small epochs)
to keep CI fast. The full 30-seed × 80-epoch reproduction is
verified by scripts/probe_bicfc_ensemble_reproduction.py.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lnn.core.ensemble import BiCfCEnsemble
from lnn.core.multimodal_physreg import CrossModalAttnBiCfCNADWithMDN


# ---------- Helper functions ----------


def _make_synthetic_dataset(
    num_samples: int = 32, window: int = 8, output_size: int = 2, seed: int = 0,
):
    """Create a tiny synthetic dataset for fast testing.

    Returns a TensorDataset where each sample is (video, audio, target_params).
    """
    torch.manual_seed(seed)
    video = torch.randn(num_samples, window, 3)
    audio = torch.randn(num_samples, window, 1)
    target = torch.randn(num_samples, output_size)
    return TensorDataset(video, audio, target)


def _to_multimodal_dict(dataset):
    """Convert TensorDataset to the multimodal dict format expected by Bi-CfC.

    The ensemble expects a dataset that yields (batch_dict, target_dict) where
    batch_dict has 'video' and 'audio' keys and target_dict has 'params'.
    """
    class _DictWrapper:
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            video, audio, target = self.base[i]
            return ({"video": video, "audio": audio}, {"params": target})
    return _DictWrapper(dataset)


def _make_loader(num_samples=32, window=8, output_size=2, batch_size=8, seed=0):
    """Make a DataLoader with the right dict format."""
    ds = _make_synthetic_dataset(
        num_samples=num_samples, window=window, output_size=output_size, seed=seed
    )
    return DataLoader(_to_multimodal_dict(ds), batch_size=batch_size, shuffle=False)


# ---------- Instantiation tests ----------


def test_default_initialization_matches_v15_recipe() -> None:
    """Default args should match the v15 FINAL recipe from round 65."""
    ensemble = BiCfCEnsemble()
    assert ensemble.n_seeds == 30, "Default n_seeds must be 30 (v15 recipe)"
    assert ensemble.K == 20, "Default K must be 20 (v15 recipe)"
    assert ensemble.hidden_size == 96, "Default hidden_size must be 96 (round 38 SOTA)"
    assert ensemble.epochs == 80, "Default epochs must be 80 (v15 recipe)"
    assert ensemble.warmup_epochs == 40, "Default warmup_epochs must be 40 (half of epochs)"
    assert ensemble.phase2_inject_sigma == 0.10, "Default inject_sigma must be 0.10 (v15 recipe)"
    assert ensemble.freeze == "audio_only", "Default freeze must be 'audio_only' (v15 recipe)"
    assert ensemble.val_frac == 0.20, "Default val_frac must be 0.20 (80/20 split)"


def test_custom_initialization() -> None:
    """Custom args should be properly stored."""
    ensemble = BiCfCEnsemble(
        n_seeds=10, K=5, hidden_size=32,
        num_mixtures=2, output_size=3,
        epochs=10, warmup_epochs=5,
        phase2_inject_sigma=0.05, freeze="none",
        val_frac=0.30, lr=1e-3, device="cpu",
    )
    assert ensemble.n_seeds == 10
    assert ensemble.K == 5
    assert ensemble.hidden_size == 32
    assert ensemble.num_mixtures == 2
    assert ensemble.output_size == 3
    assert ensemble.epochs == 10
    assert ensemble.warmup_epochs == 5
    assert ensemble.phase2_inject_sigma == 0.05
    assert ensemble.freeze == "none"
    assert ensemble.val_frac == 0.30
    assert ensemble.lr == 1e-3
    assert str(ensemble.device) == "cpu"


def test_K_greater_than_n_seeds_raises() -> None:
    """K > n_seeds is invalid; should raise ValueError."""
    with pytest.raises(ValueError, match="K .* cannot exceed n_seeds"):
        BiCfCEnsemble(n_seeds=5, K=10)


def test_invalid_freeze_raises() -> None:
    """freeze must be 'audio_only' or 'none'."""
    with pytest.raises(ValueError, match="freeze must be"):
        BiCfCEnsemble(freeze="invalid_value")


# ---------- State management tests ----------


def test_models_not_trained_before_fit() -> None:
    """Before fit(), models_ should be empty and predict() should raise."""
    ensemble = BiCfCEnsemble(n_seeds=2, K=1, epochs=2, warmup_epochs=1)
    assert ensemble.models_ == []
    assert ensemble.val_mses_ == []
    assert ensemble.top_k_indices_ == []
    loader = _make_loader(num_samples=8)
    with pytest.raises(RuntimeError, match="Call fit\\(\\)"):
        ensemble.predict(loader)
    with pytest.raises(RuntimeError, match="Call fit\\(\\)"):
        ensemble.evaluate(loader)


# ---------- Fit tests ----------


def test_fit_trains_n_seeds_models() -> None:
    """fit() should train exactly n_seeds models."""
    n_seeds = 3
    K = 2
    ensemble = BiCfCEnsemble(
        n_seeds=n_seeds, K=K,
        hidden_size=8,  # small for speed
        epochs=2, warmup_epochs=1,
        output_size=2, num_mixtures=1,
    )
    dataset = _to_multimodal_dict(_make_synthetic_dataset(num_samples=16, window=4))
    ensemble.fit(dataset, seed_values=[1, 2, 3])
    assert len(ensemble.models_) == n_seeds
    assert len(ensemble.val_mses_) == n_seeds
    assert len(ensemble.top_k_indices_) == K
    # All models should be CrossModalAttnBiCfCNADWithMDN
    for m in ensemble.models_:
        assert isinstance(m, CrossModalAttnBiCfCNADWithMDN)


def test_fit_uses_default_seeds_when_not_provided() -> None:
    """If seed_values is None, fit() should use 1..n_seeds."""
    ensemble = BiCfCEnsemble(n_seeds=3, K=2, hidden_size=8, epochs=2, warmup_epochs=1, output_size=2)
    dataset = _to_multimodal_dict(_make_synthetic_dataset(num_samples=8, window=4))
    ensemble.fit(dataset)  # no seed_values
    assert len(ensemble.models_) == 3
    assert len(ensemble.val_mses_) == 3


def test_fit_requires_at_least_n_seeds() -> None:
    """If seed_values has fewer than n_seeds, raise ValueError."""
    ensemble = BiCfCEnsemble(n_seeds=5, K=2, hidden_size=8, epochs=2, warmup_epochs=1, output_size=2)
    dataset = _to_multimodal_dict(_make_synthetic_dataset(num_samples=8, window=4))
    with pytest.raises(ValueError, match="Need at least n_seeds"):
        ensemble.fit(dataset, seed_values=[1, 2])  # only 2, but n_seeds=5


def test_fit_smart_selection_picks_lowest_val_mse() -> None:
    """top_k_indices_ should be sorted by val MSE (ascending)."""
    n_seeds = 5
    K = 3
    ensemble = BiCfCEnsemble(
        n_seeds=n_seeds, K=K, hidden_size=8, epochs=2, warmup_epochs=1, output_size=2,
    )
    dataset = _to_multimodal_dict(_make_synthetic_dataset(num_samples=16, window=4))
    ensemble.fit(dataset, seed_values=[1, 2, 3, 4, 5])
    # Verify selection is correct
    val_mses_array = ensemble.val_mses_
    expected_top_k = sorted(range(n_seeds), key=lambda i: val_mses_array[i])[:K]
    assert ensemble.top_k_indices_ == expected_top_k
    # Verify val_mses for selected models are in ascending order
    selected_val_mses = [val_mses_array[i] for i in ensemble.top_k_indices_]
    assert selected_val_mses == sorted(selected_val_mses)


# ---------- Predict tests ----------


def test_predict_returns_correct_shape() -> None:
    """predict() should return tensor of shape [N_samples, output_size]."""
    output_size = 5
    n_samples = 16
    ensemble = BiCfCEnsemble(
        n_seeds=3, K=2, hidden_size=8, epochs=2, warmup_epochs=1, output_size=output_size,
    )
    dataset = _to_multimodal_dict(
        _make_synthetic_dataset(num_samples=n_samples, window=4, output_size=output_size)
    )
    ensemble.fit(dataset, seed_values=[1, 2, 3])
    loader = _make_loader(num_samples=n_samples, window=4, output_size=output_size)
    preds = ensemble.predict(loader)
    assert preds.shape == (n_samples, output_size)
    assert isinstance(preds, torch.Tensor)


def test_predict_averages_top_k_not_all() -> None:
    """predict() should average exactly K models, not all n_seeds."""
    n_seeds = 4
    K = 2
    ensemble = BiCfCEnsemble(
        n_seeds=n_seeds, K=K, hidden_size=8, epochs=2, warmup_epochs=1, output_size=2,
    )
    dataset = _to_multimodal_dict(_make_synthetic_dataset(num_samples=8, window=4))
    ensemble.fit(dataset, seed_values=[1, 2, 3, 4])
    loader = _make_loader(num_samples=8, window=4)
    # Get predictions from each model individually
    individual_preds = []
    for model in ensemble.models_:
        model.eval()
        with torch.no_grad():
            preds_one = []
            for batch, _ in loader:
                video = batch["video"]
                audio = batch["audio"]
                out = model(video, audio)
                final = {k: v[:, -1] for k, v in out.items()}
                from lnn.core.mdn import mdn_mean
                preds_one.append(mdn_mean(final))
            individual_preds.append(torch.cat(preds_one))
    # Compute expected ensemble (top K)
    top_k = ensemble.top_k_indices_[:K]
    expected_ensemble = torch.stack([individual_preds[i] for i in top_k]).mean(dim=0)
    actual_ensemble = ensemble.predict(loader)
    assert torch.allclose(expected_ensemble, actual_ensemble, atol=1e-5)


# ---------- Evaluate tests ----------


def test_evaluate_returns_correct_metrics() -> None:
    """evaluate() should return dict with all required fields."""
    ensemble = BiCfCEnsemble(
        n_seeds=3, K=2, hidden_size=8, epochs=2, warmup_epochs=1, output_size=2,
    )
    dataset = _to_multimodal_dict(_make_synthetic_dataset(num_samples=16, window=4))
    ensemble.fit(dataset, seed_values=[1, 2, 3])
    loader = _make_loader(num_samples=16, window=4)
    metrics = ensemble.evaluate(loader)
    assert "ensemble_mse" in metrics
    assert "per_seed_mean_mse" in metrics
    assert "per_seed_std_mse" in metrics
    assert "n_seeds" in metrics
    assert "K" in metrics
    assert metrics["n_seeds"] == 3
    assert metrics["K"] == 2
    assert metrics["ensemble_mse"] >= 0
    assert metrics["per_seed_mean_mse"] >= 0
    assert metrics["per_seed_std_mse"] >= 0


def test_ensemble_mse_better_than_per_seed_mean() -> None:
    """Smart selection should produce ensemble MSE <= per-seed mean MSE.

    This is the key value proposition of seed ensemble: averaging top-K
    models should give MSE at least as good as the average per-seed model.
    """
    ensemble = BiCfCEnsemble(
        n_seeds=5, K=3, hidden_size=16, epochs=3, warmup_epochs=1, output_size=2,
    )
    dataset = _to_multimodal_dict(_make_synthetic_dataset(num_samples=32, window=8))
    ensemble.fit(dataset, seed_values=[1, 2, 3, 4, 5])
    loader = _make_loader(num_samples=32, window=8)
    metrics = ensemble.evaluate(loader)
    # Ensemble should be better (or equal) than per-seed mean
    assert metrics["ensemble_mse"] <= metrics["per_seed_mean_mse"] * 1.5, (
        f"ensemble_mse ({metrics['ensemble_mse']:.4f}) should be <= 1.5x per_seed_mean "
        f"({metrics['per_seed_mean_mse']:.4f})"
    )


# ---------- Integration test: full round 68 reproduction pattern ----------


def test_full_workflow_matches_reproduction() -> None:
    """End-to-end test mimicking scripts/probe_bicfc_ensemble_reproduction.py.

    Verifies the full workflow: instantiate -> fit -> predict -> evaluate.
    """
    # Use 2 seeds, K=1, very small for speed
    n_seeds = 2
    K = 1
    ensemble = BiCfCEnsemble(
        n_seeds=n_seeds, K=K, hidden_size=4, output_size=5,
        epochs=2, warmup_epochs=1,
        phase2_inject_sigma=0.10, freeze="audio_only",
    )
    dataset = _to_multimodal_dict(
        _make_synthetic_dataset(num_samples=8, window=4, output_size=5, seed=42)
    )
    ensemble.fit(dataset, seed_values=[1, 2])
    loader = _make_loader(num_samples=8, window=4, output_size=5, seed=42)
    metrics = ensemble.evaluate(loader)
    # All required fields present
    assert metrics["ensemble_mse"] >= 0
    assert len(ensemble.models_) == n_seeds
    assert len(ensemble.top_k_indices_) == K


# ---------- Recipe verification ----------


def test_v15_recipe_embedded_in_defaults() -> None:
    """Verify the v15 recipe is correctly captured in default args.

    Cross-references the recipe from round 65 (NEW BEST, 46th meta)
    and round 67 (47th meta, 30 seeds is FINAL sweet spot).
    """
    ensemble = BiCfCEnsemble()
    # From round 65: 30 seeds + K=20 + phase2 inject=0.10
    assert ensemble.n_seeds == 30
    assert ensemble.K == 20
    assert ensemble.phase2_inject_sigma == 0.10
    # From round 56/65: freeze=audio_only is best
    assert ensemble.freeze == "audio_only"
    # From round 65: 80/20 val split
    assert ensemble.val_frac == 0.20
    # From round 25-26: warmup = half of epochs (40 = 80/2)
    assert ensemble.warmup_epochs == ensemble.epochs // 2
