"""Tests for the PRD §10 #3 ``--phase-d`` preset on the LNN-vs-LSTM
ablation runner.

These tests are CLI-level smoke tests: they import the runner, drive it via
``argparse`` with a tiny ``sys.argv`` patch, and assert the post-parse
``args`` reflect the phase-D scale-up preset (hidden=64, epochs=50,
samples=4000, seq_len=64, warmup_frac=0.1) *and* that explicit CLI overrides
beat the preset.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys
import types

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ablation_lnn_vs_lstm_timeseries.py"


def _load_runner() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("ablation_lnn_vs_lstm", SCRIPT)
    assert spec and spec.loader, f"failed to load spec for {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_main_with_argv(runner: types.ModuleType, argv: list[str]) -> int:
    saved = sys.argv
    sys.argv = ["ablation_lnn_vs_lstm"] + argv
    try:
        return runner.main()
    finally:
        sys.argv = saved


def test_phase_d_preset_applies_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "_run_one", lambda *a, **k: {"test_mse": 0.0, "test_mae": 0.0, "train_seconds": 0.0, "inference_samples_per_sec": 0.0, "parameters": 0})
    monkeypatch.setattr(runner, "_aggregate", lambda per_run: {})
    rc = _run_main_with_argv(
        runner,
        ["--phase-d", "--dataset", "mackey_glass", "--output-dir", str(tmp_path)],
    )
    assert rc == 0
    # Inspect the JSON we just wrote to learn the actual applied values.
    json_paths = list(tmp_path.glob("*_lnn_vs_lstm.json"))
    assert json_paths, "expected a phase-d run to produce a json artefact"
    latest = max(json_paths, key=lambda p: p.stat().st_mtime)
    import json

    payload = json.loads(latest.read_text(encoding="utf-8"))
    cfg = payload["config"]
    assert cfg["phase_d"] is True
    # Preset values applied.
    assert cfg["hidden_size"] == 64
    assert cfg["epochs"] == 50
    assert cfg["samples"] == 4000
    assert cfg["seq_len"] == 64
    assert cfg["warmup_frac"] == pytest.approx(0.1)
    # phase_d_applied should list every preset key.
    assert set(cfg["phase_d_applied"]) == {
        "hidden_size",
        "epochs",
        "samples",
        "seq_len",
        "warmup_frac",
    }


def test_phase_d_cli_overrides_win(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "_run_one", lambda *a, **k: {"test_mse": 0.0, "test_mae": 0.0, "train_seconds": 0.0, "inference_samples_per_sec": 0.0, "parameters": 0})
    monkeypatch.setattr(runner, "_aggregate", lambda per_run: {})
    rc = _run_main_with_argv(
        runner,
        [
            "--phase-d",
            "--hidden-size",
            "16",
            "--epochs",
            "5",
            "--samples",
            "100",
            "--dataset",
            "sine",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert rc == 0
    json_paths = list(tmp_path.glob("*_lnn_vs_lstm.json"))
    latest = max(json_paths, key=lambda p: p.stat().st_mtime)
    import json

    payload = json.loads(latest.read_text(encoding="utf-8"))
    cfg = payload["config"]
    assert cfg["phase_d"] is True
    # CLI wins.
    assert cfg["hidden_size"] == 16
    assert cfg["epochs"] == 5
    assert cfg["samples"] == 100
    # seq_len and warmup_frac still take the preset (no override).
    assert cfg["seq_len"] == 64
    assert cfg["warmup_frac"] == pytest.approx(0.1)


def test_phase_d_help_lists_flag() -> None:
    """The --phase-d help text should appear in the CLI help output."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert "--phase-d" in result.stdout
    assert "PRD §10 #3" in result.stdout
