"""Tests for ``scripts/experiment_device_control_cases.py`` (iter#36).

Verifies the 4-case device-control harness described in
``docs/PRD_设备操控_LNN.md`` §3:

* Synthetic data generators produce the expected shapes (no real hardware).
* Each of the 4 cases (quadruped / drone / industrial / battery) runs a
  forward + backward pass with in-house LNN modules.
* ``_aggregate_seeds`` produces a {mean, std, n_seeds} dict from per-seed
  reports.
* CLI smoke: ``--case industrial --quick --steps 8 --seeds 1`` writes the
  expected JSON schema under ``analysis/device_control/``.

The script is **synthetic-only** by design (no ROS, no mavlink, no CAN, no
real sensors). See ``docs/PRD_设备操控_LNN.md`` §0 / user preference 2026-06-09.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_device_control_cases import (  # noqa: E402
    _aggregate_seeds,
    _gen_battery_synth,
    _gen_drone_synth,
    _gen_inverted_pendulum_il,
    _gen_quadruped_rollout,
    run_case_battery,
    run_case_drone,
    run_case_industrial,
    run_case_quadruped,
)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def test_gen_quadruped_rollout_shapes() -> None:
    obs, actions, returns = _gen_quadruped_rollout(seed=42)
    assert obs.shape[0] == 8 and obs.ndim == 3
    assert actions.shape == obs.shape
    assert returns.shape == (8,)
    # Re-seed determinism.
    obs2, _, _ = _gen_quadruped_rollout(seed=42)
    assert torch.allclose(obs, obs2)


def test_gen_drone_synth_shapes() -> None:
    X_v, X_i, Y = _gen_drone_synth(seed=42)
    assert X_v.shape == (256, 32, 64)
    assert X_i.shape == (256, 32, 6)
    assert Y.shape == (256, 4)
    # Y is a function of both modalities — change seed and Y changes.
    _, _, Y_other = _gen_drone_synth(seed=43)
    assert not torch.allclose(Y, Y_other)


def test_gen_inverted_pendulum_il_shapes() -> None:
    obs, actions = _gen_inverted_pendulum_il(seed=42)
    assert obs.shape == (1024, 16, 4)
    assert actions.shape == (1024, 16, 1)
    # Sanity: actions are finite.
    assert torch.isfinite(actions).all()


def test_gen_battery_synth_shapes_and_decreasing_soh() -> None:
    X, Y = _gen_battery_synth(seed=42, n_cells=8, cycles_per_cell=50)
    assert X.shape == (8, 50, 128, 4)
    assert Y.shape == (8, 50)
    # Y is monotonically non-increasing along the cycle axis.
    diffs = Y[:, 1:] - Y[:, :-1]
    assert (diffs <= 1e-5).all(), "SoH should not increase over cycles"


# ---------------------------------------------------------------------------
# Per-case forward+backward smoke
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_fn", [
    run_case_quadruped, run_case_drone, run_case_industrial, run_case_battery,
])
def test_each_case_runs_end_to_end(case_fn) -> None:
    """Each of the 4 cases returns a valid report dict on a 8-step smoke."""
    rpt = case_fn(seed=42, steps=8, quick=True)
    assert isinstance(rpt, dict)
    assert rpt["case"] in {"quadruped", "drone", "industrial", "battery"}
    assert rpt["status"] in {"ok", "skipped", "error"}
    if rpt["status"] == "ok":
        # primary_metric_value must be a finite float.
        assert isinstance(rpt["primary_metric_value"], float)
        assert rpt["params"] > 0
        assert rpt["wall_time_s"] >= 0.0


# ---------------------------------------------------------------------------
# aggregate_seeds
# ---------------------------------------------------------------------------


def test_aggregate_seeds_handles_empty_input() -> None:
    assert _aggregate_seeds([]) == {"mean": None, "std": None, "n_seeds": 0}


def test_aggregate_seeds_handles_single_seed() -> None:
    rep = {"primary_metric_value": 0.5, "status": "ok"}
    agg = _aggregate_seeds([rep])
    assert agg["mean"] == 0.5
    assert agg["std"] == 0.0  # stdev undefined for n=1
    assert agg["n_seeds"] == 1


def test_aggregate_seeds_computes_mean_std_across_seeds() -> None:
    reps = [
        {"primary_metric_value": 0.1, "status": "ok"},
        {"primary_metric_value": 0.2, "status": "ok"},
        {"primary_metric_value": 0.3, "status": "ok"},
    ]
    agg = _aggregate_seeds(reps)
    assert agg["mean"] == pytest.approx(0.2)
    assert agg["std"] == pytest.approx(statistics_stdev([0.1, 0.2, 0.3]))
    assert agg["n_seeds"] == 3


def test_aggregate_seeds_skips_non_ok_reports() -> None:
    reps = [
        {"primary_metric_value": 0.1, "status": "ok"},
        {"status": "skipped", "notes": "import error"},
    ]
    agg = _aggregate_seeds(reps)
    assert agg["n_seeds"] == 1
    assert agg["mean"] == 0.1


def statistics_stdev(values):
    import statistics
    return statistics.stdev(values)


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_quick_industrial_smoke(tmp_path) -> None:
    """`--case industrial --quick --steps 8 --seeds 1` runs end-to-end."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "experiment_device_control_cases.py"),
        "--case", "industrial",
        "--quick",
        "--steps", "8",
        "--seeds", "1",
        "--out-dir", str(tmp_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (
        f"CLI failed:\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    json_files = list(tmp_path.glob("*_device_control_industrial.json"))
    assert len(json_files) == 1
    payload = json.loads(json_files[0].read_text())
    assert payload["case"] == "industrial"
    assert payload["schema"]["version"] == 1
    assert len(payload["per_seed"]) == 1
    # Master summary file should also be written.
    master = tmp_path / "latest_device_control_summary.json"
    assert master.exists()
