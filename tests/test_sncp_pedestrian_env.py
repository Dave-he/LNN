"""Tests for the iter#27 pedestrian-aware env extension.

Verifies the new ``--n-pedestrians`` flag on ``PointMassNavLite``:
* obs_dim scales with n_pedestrians (BASE_OBS_DIM + 2*N)
* Pedestrians move on a deterministic seeded trajectory
* Backward compat: n_pedestrians=0 yields the iter#26 env (4-dim obs,
  2 static obstacles, no pedestrian entries in the obs)
* Invalid n_pedestrians is rejected
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_sncp_ppo_lite import PointMassNavLite  # noqa: E402


def test_obs_dim_constant_across_n_pedestrians() -> None:
    """obs_dim is constant (zero-padded) for all n_pedestrians in [0, MAX_PED_SLOTS]."""
    expected_obs_dim = PointMassNavLite.BASE_OBS_DIM + 2 * PointMassNavLite.MAX_PED_SLOTS
    for n in (0, 1, 2, 3, 5):
        env = PointMassNavLite(seed=42, n_pedestrians=n)
        assert env.obs_dim == expected_obs_dim, (
            f"n={n}: expected obs_dim={expected_obs_dim}, got {env.obs_dim}"
        )
        obs = env.reset(seed=42)
        assert obs.shape == (expected_obs_dim,)


def test_n_zero_backward_compat() -> None:
    """n_pedestrians=0 keeps 2 static obstacles and zero-pads all ped slots."""
    env = PointMassNavLite(seed=42, n_pedestrians=0)
    assert env.n_pedestrians == 0
    assert env.obstacles == [(-0.2, 0.3), (0.5, -0.3)]
    obs = env.reset(seed=42)
    # All 10 ped-slot dims are zero (sentinel).
    assert obs[4:].tolist() == [0.0] * (2 * PointMassNavLite.MAX_PED_SLOTS)
    # Step once and check obs shape still constant.
    step = env.step(torch.tensor([0.05, 0.0]))
    assert step.obs.shape == (env.obs_dim,)


def test_pedestrians_move_deterministically() -> None:
    """Same seed ⇒ same pedestrian trajectory across runs."""
    env1 = PointMassNavLite(seed=7, n_pedestrians=3)
    env2 = PointMassNavLite(seed=7, n_pedestrians=3)
    a = env1._ped_positions(0)
    b = env2._ped_positions(0)
    assert a == b, f"ped positions at t=0 differ: {a} vs {b}"
    a1 = env1._ped_positions(3)
    b1 = env2._ped_positions(3)
    assert a1 == b1
    assert a != a1, "pedestrians should move between t=0 and t=3"


def test_obs_includes_ped_relative_positions_and_pads() -> None:
    """Active ped slots are real relative positions; remaining slots are 0."""
    env = PointMassNavLite(seed=42, n_pedestrians=2)
    obs = env.reset(seed=42)
    # First 4 dims = base (agent at origin, so goal_delta = [1, 1]).
    assert obs[0].item() == pytest.approx(0.0)
    assert obs[1].item() == pytest.approx(0.0)
    assert obs[2].item() == pytest.approx(1.0)
    assert obs[3].item() == pytest.approx(1.0)
    # ped0 origin (-0.2, 0.3), radius 0.2, phase 0 → ped0 at (0.0, 0.3).
    # agent - ped0 = (0.0, -0.3).
    assert obs[4].item() == pytest.approx(0.0, abs=1e-5)
    assert obs[5].item() == pytest.approx(-0.3, abs=1e-5)
    # ped1 origin (0.5, -0.3), radius 0.2, phase 0 → ped1 at (0.7, -0.3).
    # agent - ped1 = (-0.7, 0.3).
    assert obs[6].item() == pytest.approx(-0.7, abs=1e-5)
    assert obs[7].item() == pytest.approx(0.3, abs=1e-5)
    # Remaining 3 ped slots are zero-padded.
    assert obs[8:].tolist() == [0.0] * (2 * (PointMassNavLite.MAX_PED_SLOTS - 2))


def test_invalid_n_pedestrians_rejected() -> None:
    """n_pedestrians < 0 or > MAX_PED_SLOTS must raise."""
    with pytest.raises(ValueError, match="n_pedestrians must be >= 0"):
        PointMassNavLite(seed=0, n_pedestrians=-1)
    with pytest.raises(ValueError, match="exceeds MAX_PED_SLOTS"):
        PointMassNavLite(seed=0, n_pedestrians=PointMassNavLite.MAX_PED_SLOTS + 1)


def test_curriculum_cli_runs_end_to_end(tmp_path) -> None:
    """`--curriculum` with 3 mini stages runs end-to-end and writes JSON+MD."""
    import subprocess  # local import — top-level would slow down plain unit tests
    output_dir = tmp_path / "sncp_curriculum_smoke"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "experiment_sncp_ppo_lite.py"),
        "--curriculum",
        "--ped-curriculum-list", "1,2,3",
        "--ppo-updates-per-stage", "2",
        "--episodes-per-update", "4",
        "--epochs", "1",
        "--ltc-hidden", "16",
        "--trunk-hidden", "16",
        "--seed", "7",
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"CLI failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    # One JSON and one MD file should be produced.
    json_files = list(output_dir.glob("*_sncp_ppo_lite.json"))
    md_files = list(output_dir.glob("*_sncp_ppo_lite.md"))
    assert len(json_files) == 1, f"expected 1 JSON, got {len(json_files)} in {output_dir}"
    assert len(md_files) == 1, f"expected 1 MD, got {len(md_files)} in {output_dir}"
    import json as _json
    payload = _json.loads(json_files[0].read_text())
    assert payload["stages"] == [1, 2, 3]
    assert len(payload["stage_history"]) == 3
    # 2 updates * 3 stages = 6 rollout records.
    assert len(payload["rollout_history"]) == 6
    # Each stage has its own n_pedestrians tagged in rollout_history.
    n_peds_seen = sorted({r["n_pedestrians"] for r in payload["rollout_history"]})
    assert n_peds_seen == [1, 2, 3]
