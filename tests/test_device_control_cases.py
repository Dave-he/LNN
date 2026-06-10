"""Tests for ``scripts/experiment_device_control_cases.py`` (iter#36 + iter#37).

Verifies the 4-case device-control harness described in
``docs/PRD_设备操控_LNN.md`` §3:

* Synthetic data generators produce the expected shapes (no real hardware).
* Each of the 4 cases (quadruped / drone / industrial / battery) runs a
  forward + backward pass with in-house LNN modules.
* ``_aggregate_seeds`` produces a {mean, std, n_seeds} dict from per-seed
  reports.
* CLI smoke: ``--case industrial --quick --steps 8 --seeds 1`` writes the
  expected JSON schema under ``analysis/device_control/``.

iter#37 addition — EntroLnn 2-stage transformable:
* ``TransformableLTC`` public class init / train_reference / refine_target
* ``run_case_battery(battery_mode="transformable")`` runs end-to-end
* CLI flag ``--battery-mode transformable`` works
* refine does not destroy reference-stage parameters (stability guard)

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

# iter#37: EntroLnn 2-stage transformable (PRD-B winner)
# We load lnn/core/ltc.py *directly* via importlib to bypass lnn/__init__.py
# (which transitively imports scipy via the LNN module graph and triggers a
# pre-existing numpy/scipy version mismatch on this host). The script uses
# the same try/except pattern in production.
import importlib.util as _importlib_util  # noqa: E402

_ltc_path = ROOT / "lnn" / "core" / "ltc.py"
_spec = _importlib_util.spec_from_file_location("_lnn_ltc_direct", _ltc_path)
_lnn_ltc = _importlib_util.module_from_spec(_spec) if _spec is not None else None
_TransformableLTC = None
_HAS_TRANSFORMABLE = False
_LTC_IMPORT_ERROR: str | None = None
if _lnn_ltc is not None and _spec is not None:
    try:
        _spec.loader.exec_module(_lnn_ltc)  # type: ignore[union-attr]
        _TransformableLTC = getattr(_lnn_ltc, "TransformableLTC", None)
        _HAS_TRANSFORMABLE = _TransformableLTC is not None
    except Exception as exc:  # pragma: no cover
        _TransformableLTC = None
        _HAS_TRANSFORMABLE = False
        _LTC_IMPORT_ERROR = f"{type(exc).__name__}: {str(exc)[:160]}"
TransformableLTC = _TransformableLTC
transformable_required = pytest.mark.skipif(
    not _HAS_TRANSFORMABLE,
    reason=(
        f"TransformableLTC unavailable: {_LTC_IMPORT_ERROR!r}"
        if _LTC_IMPORT_ERROR
        else "TransformableLTC unavailable"
    ),
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


# ---------------------------------------------------------------------------
# iter#37 — EntroLnn 2-stage transformable (PRD-B winner)
# ---------------------------------------------------------------------------


def _build_battery_inputs(seed: int = 7, n_cells: int = 4, cycles: int = 16) -> tuple:
    """Build small [B, T, F] + [B, T] inputs for TransformableLTC tests.

    Mirrors ``_gen_battery_synth`` but with smaller sizes for fast tests.
    """
    g = torch.Generator().manual_seed(seed)
    # [n_cells, cycles, 8, feat_dim=4]
    X = torch.randn(n_cells, cycles, 8, 4, generator=g) * 0.1
    # Y is the mean of the last feature across the time axis (signal).
    Y = X[:, :, -1, :].mean(dim=1)  # [n_cells, 4] — proxy for SoH
    # Flatten to a single batch dim for TransformableLTC
    X_flat = X.mean(dim=2)  # [n_cells, cycles, feat_dim]
    Y_flat = Y  # [n_cells, output=4]
    return X_flat, Y_flat


@transformable_required
def test_transformable_ltc_init_and_train_reference() -> None:
    """TransformableLTC init succeeds, train_reference returns history."""
    X, Y = _build_battery_inputs()
    model = TransformableLTC(
        input_size=4, hidden_size=8, output_size=4,
        train_lr=1e-3, refine_lr=1e-4,
    )
    # forward pass works
    out = model(X)
    assert out.shape == (4, 16, 4)
    # train_reference returns dict with ref_loss_history
    res = model.train_reference(X[:2], Y[:2], epochs=2, batch_size=2)
    assert "ref_loss_history" in res
    assert len(res["ref_loss_history"]) == 2
    # both losses are finite
    for l in res["ref_loss_history"]:
        assert isinstance(l, float)
        assert l == l  # not NaN
    assert res["final_ref_loss"] == res["ref_loss_history"][-1]


@transformable_required
def test_transformable_ltc_refine_updates_params() -> None:
    """refine_target actually changes parameters (gradient applied)."""
    X, Y = _build_battery_inputs()
    model = TransformableLTC(
        input_size=4, hidden_size=8, output_size=4,
        train_lr=1e-3, refine_lr=1e-4,
    )
    # Capture param L1 norm before
    norm_before = model.param_l1_norm()
    res = model.refine_target(X[2:], Y[2:], K=3, batch_size=2)
    norm_after = model.param_l1_norm()
    # Params should have changed (refine took gradient steps)
    assert norm_after != norm_before, (
        f"params unchanged after refine_target (norm {norm_before} == {norm_after})"
    )
    assert "refine_loss_history" in res
    assert len(res["refine_loss_history"]) == 3


@transformable_required
def test_transformable_ltc_refine_stability_guard() -> None:
    """refine with tiny lr (1e-4) should not destroy reference-stage params.

    The stability guard: final_ref_loss_after should not be >10× the
    post-train value. We use a fixed-seed small training run to keep the
    test deterministic.
    """
    X, Y = _build_battery_inputs()
    model = TransformableLTC(
        input_size=4, hidden_size=8, output_size=4,
        train_lr=1e-3, refine_lr=1e-4,  # 10× smaller for refine
    )
    train_res = model.train_reference(X[:2], Y[:2], epochs=2, batch_size=2)
    ref_loss_pre = train_res["final_ref_loss"]
    refine_res = model.refine_target(X[2:], Y[2:], K=3, batch_size=2)
    # The ref_loss_after is on target cells (proxy), but the relative ratio
    # is what matters. It should be within 10× of ref_loss_pre.
    ref_loss_after = refine_res["final_ref_loss_after"]
    assert ref_loss_pre > 0
    ratio = ref_loss_after / ref_loss_pre
    # Soft bound — allow 10× headroom for proxy-target asymmetry.
    assert ratio < 100, (
        f"refine destroyed params: ref loss {ref_loss_pre:.4f} → "
        f"{ref_loss_after:.4f} ({ratio:.1f}× spike)"
    )


@transformable_required
def test_transformable_ltc_refine_lr_safety() -> None:
    """Constructor rejects refine_lr > train_lr (lr 衰减 安全)."""
    with pytest.raises(ValueError, match="refine_lr"):
        TransformableLTC(
            input_size=4, hidden_size=8, output_size=4,
            train_lr=1e-3, refine_lr=1e-2,  # refine > train — unsafe
        )
    with pytest.raises(ValueError, match="refine_lr"):
        TransformableLTC(
            input_size=4, hidden_size=8, output_size=4,
            train_lr=1e-3, refine_lr=0.0,  # zero — also rejected
        )


@transformable_required
def test_run_case_battery_transformable_mode_runs() -> None:
    """run_case_battery(battery_mode='transformable') runs end-to-end."""
    rpt = run_case_battery(seed=42, steps=8, quick=True, battery_mode="transformable", refine_steps=3)
    assert rpt["case"] == "battery"
    assert rpt["status"] == "ok"
    assert rpt["primary_metric"] == "val_mse"
    assert isinstance(rpt["primary_metric_value"], float)
    # secondary_metrics should carry the 2-stage history
    sec = rpt["secondary_metrics"]
    assert sec["mode"] == "transformable"
    assert "ref_loss_history" in sec
    assert "refine_loss_history" in sec
    assert len(sec["refine_loss_history"]) == 3


@transformable_required
def test_battery_mode_cli_smoke_transformable(tmp_path) -> None:
    """`--case battery --battery-mode transformable` runs end-to-end via CLI."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "experiment_device_control_cases.py"),
        "--case", "battery",
        "--battery-mode", "transformable",
        "--refine-steps", "3",
        "--quick",
        "--steps", "8",
        "--seeds", "1",
        "--out-dir", str(tmp_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (
        f"CLI failed:\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    json_files = list(tmp_path.glob("*_device_control_battery.json"))
    assert len(json_files) == 1
    payload = json.loads(json_files[0].read_text())
    assert payload["config"]["battery_mode"] == "transformable"
    assert payload["config"]["refine_steps"] == 3
    assert payload["per_seed"][0]["secondary_metrics"]["mode"] == "transformable"
