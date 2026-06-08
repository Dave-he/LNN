"""Tests for iter#28 PDNA on synthetic Pathfinder (LRA-style long-range)."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lnn.data.pathfinder_synth import (  # noqa: E402
    PathfinderConfig,
    generate_pathfinder,
)


def test_pathfinder_default_config_shapes() -> None:
    """Default PathfinderConfig → 32x32 grid, 1024-length sequence."""
    cfg = PathfinderConfig()
    seqs, labels = generate_pathfinder(8, cfg=cfg, seed=42)
    assert seqs.shape == (8, 32 * 32)
    assert labels.shape == (8,)
    assert labels.dtype == torch.long
    assert set(labels.tolist()).issubset({0, 1})


def test_pathfinder_class_balance_close_to_50_50() -> None:
    """Large N should be roughly balanced (50/50) within ±15 pp."""
    seqs, labels = generate_pathfinder(400, cfg=PathfinderConfig(), seed=7)
    pos_rate = float(labels.float().mean())
    assert 0.35 < pos_rate < 0.65, f"class balance out of range: pos_rate={pos_rate}"


def test_pathfinder_deterministic_same_seed() -> None:
    """Same seed → identical sequences + labels (reproducibility)."""
    seqs_a, lab_a = generate_pathfinder(50, cfg=PathfinderConfig(), seed=123)
    seqs_b, lab_b = generate_pathfinder(50, cfg=PathfinderConfig(), seed=123)
    assert torch.equal(seqs_a, seqs_b)
    assert torch.equal(lab_a, lab_b)


def test_pathfinder_endpoint_markers_present() -> None:
    """Each image must contain at least 2 cells at endpoint_value (1.0)."""
    cfg = PathfinderConfig()
    seqs, _ = generate_pathfinder(16, cfg=cfg, seed=11)
    # Per image, count pixels at exactly 1.0
    for i in range(16):
        img = seqs[i].view(32, 32)
        n_endpoint_cells = int((img == cfg.endpoint_value).sum().item())
        # Two endpoints × (2*radius+1)² = 2 * 9 = 18 cells (radius=1 → 3x3 marker).
        assert n_endpoint_cells >= 6, f"image {i}: only {n_endpoint_cells} endpoint cells"


def test_pathfinder_different_seed_changes_data() -> None:
    """Different seeds must produce different data (sanity check)."""
    seqs_a, _ = generate_pathfinder(20, cfg=PathfinderConfig(), seed=1)
    seqs_b, _ = generate_pathfinder(20, cfg=PathfinderConfig(), seed=2)
    assert not torch.equal(seqs_a, seqs_b)


def test_pdna_lra_cli_runs_end_to_end(tmp_path) -> None:
    """experiment_pdna_lra.py runs end-to-end with 1 seed × 2 variants and writes JSON+MD."""
    output_dir = tmp_path / "pdna_lra_smoke"
    # We pass --train-samples / --test-samples small for fast CLI smoke.
    # Skip the auto-MD dump by relying on --out-prefix.
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "experiment_pdna_lra.py"),
        "--seeds", "1",
        "--epochs", "1",
        "--train-samples", "40",
        "--test-samples", "20",
        "--batch-size", "8",
        "--hidden-size", "16",
        "--variants", "baseline_cfc", "cfc_pulse",
        "--out-prefix", "cli_smoke",
        # We use the standard ANALYSIS_DIR for output, so don't override --output-dir.
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, f"CLI failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    analysis_dir = ROOT / "analysis" / "pdna_lra"
    seed_json = list(analysis_dir.glob("cli_smoke_pdna_lra_baseline_cfc_seed42.json"))
    seed_md = list(analysis_dir.glob("cli_smoke_pdna_lra_summary.md"))
    assert len(seed_json) >= 1
    assert len(seed_md) == 1
    # Clean up the CLI smoke artifacts so they don't pollute future runs.
    for p in analysis_dir.glob("cli_smoke_*"):
        p.unlink()
