"""Tests for local-only deployment simulation.

The simulator must remain a rehearsal layer over synthetic reports. It should
write auditable manifests and must not claim or attempt real device access.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.local_deployment_sim import (  # noqa: E402
    FORBIDDEN_INTERFACES,
    TARGETS,
    build_manifest,
    budget_check,
    run_local_deployment_sim,
)


def _fake_source_report(case: str = "industrial") -> dict:
    return {
        "case": case,
        "per_seed": [
            {
                "case": case,
                "seed": 42,
                "status": "ok",
                "primary_metric_value": 0.01,
                "inference_ms": 0.5,
                "params": 128,
            }
        ],
        "summary": {
            "ok_seeds": 1,
            "total_seeds": 1,
            "mean_primary_metric_value": 0.01,
            "mean_inference_ms": 0.5,
            "mean_params": 128.0,
        },
    }


def test_build_manifest_declares_local_simulation_safety() -> None:
    manifest = build_manifest("industrial", TARGETS["local_cpu_smoke"], _fake_source_report())
    assert manifest["mode"] == "local_simulation"
    assert manifest["artifact"]["uri"].startswith("sim://")
    assert manifest["safety"]["real_device_access"] is False
    assert manifest["safety"]["submit_allowed"] is False
    for iface in FORBIDDEN_INTERFACES:
        assert iface in manifest["safety"]["forbidden_interfaces"]


def test_budget_check_marks_unsupported_case_as_fail() -> None:
    result = budget_check(_fake_source_report(case="drone"), TARGETS["mcu_tiny"])
    assert result["status"] == "fail"
    assert result["checks"]["case_supported"] is False


def test_run_local_deployment_sim_industrial_quick() -> None:
    payload = run_local_deployment_sim(
        "industrial",
        target_name="local_cpu_smoke",
        seeds=1,
        steps=8,
        quick=True,
    )
    assert payload["schema"]["version"] == 1
    assert payload["case"] == "industrial"
    assert payload["manifest"]["safety"]["real_device_access"] is False
    assert payload["source_report"]["summary"]["total_seeds"] == 1
    assert payload["status"] in {"pass", "fail", "blocked"}
    assert payload["audit"], "audit trail should not be empty"
    for event in payload["audit"]:
        assert event["real_device_access"] is False
        assert event["interfaces_used"] == ["python", "filesystem", "sim://"]


def test_cli_writes_payload_and_latest_summary(tmp_path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "local_deployment_sim.py"),
        "--case",
        "industrial",
        "--target",
        "local_cpu_smoke",
        "--quick",
        "--steps",
        "8",
        "--seeds",
        "1",
        "--out-dir",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (
        f"CLI failed:\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    payloads = list(tmp_path.glob("*_local_deployment_sim_local_cpu_smoke_industrial.json"))
    assert len(payloads) == 1
    payload = json.loads(payloads[0].read_text())
    assert payload["case"] == "industrial"
    assert payload["manifest"]["artifact"]["uri"].startswith("sim://")
    latest = tmp_path / "latest_local_deployment_simulation.json"
    assert latest.exists()
    latest_payload = json.loads(latest.read_text())
    assert latest_payload["runs"][0]["case"] == "industrial"
