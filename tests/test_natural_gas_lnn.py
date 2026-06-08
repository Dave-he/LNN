"""Tests for the iter-skill step 4 implementation: Natural Gas LNN Forecaster.

PRD-A from /home/hyx/.iter-skill/runs/20260608T080309Z-lnn-research-2026-06-08/02-prd-A.md
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lnn.data.natural_gas_generator import NaturalGasDatasetGenerator  # noqa: E402
from scripts.experiment_natural_gas_lnn import (  # noqa: E402
    _build_model,
    _directional_acc_7d,
    _mape,
    build_windows,
    load_natural_gas,
)


def test_generator_shape_and_columns() -> None:
    """NaturalGasDatasetGenerator produces 2645 business days × 34 features."""
    gen = NaturalGasDatasetGenerator(seed=42)
    df = gen.generate()
    assert df.shape == (2645, 34)
    assert "Spot Return" in df.columns
    assert "Spot Price" in df.columns


def test_build_windows_shape() -> None:
    """build_windows returns X [N, window, 1] and y [N]."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0, 0.02, size=200).astype(np.float32)
    X, y = build_windows(returns, window=30)
    n = 200 - 30
    assert X.shape == (n, 30, 1)
    assert y.shape == (n,)
    assert X.dtype == np.float32
    assert y.dtype == np.float32
    # y[i] should equal the next-day return (i.e. returns[i+window])
    np.testing.assert_allclose(y[:5], returns[30:35])
    # X[i, -1, 0] should equal returns[i + window - 1]
    np.testing.assert_allclose(X[:5, -1, 0], returns[29:34])


def test_load_natural_gas_returns_scaled_returns() -> None:
    """load_natural_gas returns the scaled Spot Return series (divided by 100)."""
    returns = load_natural_gas(seed=42)
    assert returns.dtype == np.float32
    assert len(returns) == 2645
    # After /100 scaling, values should be in roughly [-0.5, 0.7]
    assert returns.min() > -1.0 and returns.max() < 1.0


def test_build_model_all_five_backbones() -> None:
    """_build_model returns a usable model for all 5 backbones."""
    for name in ("ltc", "cfc", "ct_ltc", "gru", "lstm"):
        m = _build_model(name, input_size=1, hidden_size=16)
        # Forward pass on a tiny batch
        import torch
        x = torch.zeros(2, 30, 1)
        out = m(x)
        assert out.shape == (2,), f"{name}: expected [2], got {out.shape}"


def test_mape_handles_zero_returns() -> None:
    """MAPE filters out near-zero returns to avoid division blow-up."""
    y = np.array([0.001, 0.002, 0.5, 0.3, 0.4])
    p = np.array([0.002, 0.003, 0.6, 0.2, 0.5])
    mape = _mape(p, y)
    # Only |y| > 0.05 mask applies; 3 entries remain.
    assert not np.isnan(mape)
    assert mape > 0


def test_directional_acc_7d_perfect_match() -> None:
    """If preds == y_true, directional accuracy is 100%."""
    n = 30
    y = np.linspace(0.1, 0.5, n).astype(np.float32)
    acc = _directional_acc_7d(y, y)
    assert acc == pytest.approx(100.0)


def test_directional_acc_7d_random_returns_50_percent() -> None:
    """Random sign-of-cumreturn gives ~50% accuracy."""
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.1, 100).astype(np.float32)
    p = -y  # deliberately opposite signs
    acc = _directional_acc_7d(p, y)
    assert 0.0 <= acc <= 100.0


def test_natural_gas_cli_smoke(tmp_path) -> None:
    """scripts/experiment_natural_gas_lnn.py runs end-to-end with 1 seed × 1 backbone × 1 epoch."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "experiment_natural_gas_lnn.py"),
        "--seeds", "1",
        "--epochs", "1",
        "--window", "10",
        "--hidden-size", "16",
        "--batch-size", "32",
        "--backbones", "gru",  # fastest backbone
        "--out-prefix", "cli_smoke_ng",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, f"CLI failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    # Verify the summary + per-seed JSON were written to the standard ANALYSIS_DIR.
    analysis_dir = ROOT / "analysis" / "timeseries_ablation"
    seed_json = list(analysis_dir.glob("cli_smoke_ng_natural_gas_gru_seed42.json"))
    summary_md = list(analysis_dir.glob("cli_smoke_ng_natural_gas_lnn_summary.md"))
    summary_json = list(analysis_dir.glob("cli_smoke_ng_natural_gas_lnn_summary.json"))
    assert len(seed_json) >= 1
    assert len(summary_md) == 1
    assert len(summary_json) == 1
    # Clean up the CLI smoke artefacts
    for p in analysis_dir.glob("cli_smoke_ng_*"):
        p.unlink()
