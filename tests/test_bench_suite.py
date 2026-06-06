"""Tests for lnn.core.bench_suite (round 73)."""

from __future__ import annotations

import pytest
import torch

from lnn.core.bench_suite import (
    BenchResult,
    DatasetBundle,
    list_backbones,
    list_datasets,
    load_dataset,
    run_suite,
)


# ---------------------------------------- 1. registry constants
def test_list_backbones():
    backbones = list_backbones()
    assert "cfc" in backbones
    assert "gru" in backbones
    assert "dss" in backbones
    assert "mamba" in backbones
    assert len(backbones) == 4


def test_list_datasets():
    datasets = list_datasets()
    assert "mackey_glass" in datasets
    assert "sine" in datasets
    assert "toy_class" in datasets
    assert len(datasets) == 3


# ---------------------------------------- 2. dataset loaders return bundles
def test_load_mackey_glass_returns_bundle():
    b = load_dataset("mackey_glass", seed=0)
    assert isinstance(b, DatasetBundle)
    assert b.task == "regression"
    assert b.num_classes == 1
    assert b.X_train.dim() == 3
    assert b.X_test.dim() == 3
    # y is 1-D
    assert b.y_train.dim() == 1


def test_load_sine_returns_bundle():
    b = load_dataset("sine", seed=0)
    assert b.task == "regression"
    assert b.seq_len == 50


def test_load_toy_class_returns_bundle():
    b = load_dataset("toy_class", seed=0)
    assert b.task == "classification"
    assert b.num_classes == 2
    # 1000 samples total, 80% train
    assert len(b.X_train) == 800
    assert len(b.X_test) == 200


def test_load_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_dataset("does_not_exist", seed=0)


# ---------------------------------------- 3. unknown backbone raises
def test_run_suite_unknown_backbone_raises():
    with pytest.raises(ValueError):
        run_suite("transformer", "mackey_glass", seed=0, hidden=8, epochs=1)


# ---------------------------------------- 4. all 4 backbones run on mackey_glass
@pytest.mark.parametrize("backbone", ["cfc", "gru", "dss", "mamba"])
def test_all_backbones_run_on_mackey_glass(backbone):
    r = run_suite(backbone, "mackey_glass", seed=0, hidden=8, epochs=1)
    assert isinstance(r, BenchResult)
    assert r.backbone == backbone
    assert r.dataset == "mackey_glass"
    assert r.test_metric_name == "mse"
    assert r.test_metric_value >= 0  # MSE non-negative
    assert r.n_params > 0
    assert r.wall_clock_s > 0


# ---------------------------------------- 5. classification dataset works
def test_classification_dataset_runs():
    r = run_suite("cfc", "toy_class", seed=0, hidden=8, epochs=1)
    assert r.test_metric_name == "acc"
    assert 0.0 <= r.test_metric_value <= 1.0


# ---------------------------------------- 6. reproducibility with same seed
def test_same_seed_reproducible():
    r1 = run_suite("cfc", "mackey_glass", seed=42, hidden=8, epochs=2)
    r2 = run_suite("cfc", "mackey_glass", seed=42, hidden=8, epochs=2)
    assert r1.test_metric_value == r2.test_metric_value
    assert r1.n_params == r2.n_params


# ---------------------------------------- 7. different seeds give different results
def test_different_seeds_diverge():
    r1 = run_suite("cfc", "mackey_glass", seed=0, hidden=8, epochs=2)
    r2 = run_suite("cfc", "mackey_glass", seed=1, hidden=8, epochs=2)
    # Not bit-identical (different initialisation / data shuffle).
    assert r1.test_metric_value != r2.test_metric_value


# ---------------------------------------- 8. BenchResult.to_dict is JSON-safe
def test_bench_result_to_dict_json_safe():
    import json

    r = run_suite("cfc", "mackey_glass", seed=0, hidden=8, epochs=1)
    d = r.to_dict()
    # Round-trip through json.dumps
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["backbone"] == "cfc"
    assert parsed["dataset"] == "mackey_glass"
    assert parsed["seed"] == 0
    assert parsed["hidden"] == 8


# ---------------------------------------- 9. n_params scales with hidden size
def test_n_params_grows_with_hidden():
    r_small = run_suite("cfc", "mackey_glass", seed=0, hidden=8, epochs=1)
    r_large = run_suite("cfc", "mackey_glass", seed=0, hidden=64, epochs=1)
    assert r_large.n_params > r_small.n_params


# ---------------------------------------- 10. notes field is populated
def test_notes_field_populated():
    r = run_suite("cfc", "mackey_glass", seed=0, hidden=8, epochs=1)
    assert "task=" in r.notes
    assert "T=" in r.notes
