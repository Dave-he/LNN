"""Local deployment simulation for LNN device-control cases.

This script turns the synthetic-only device-control harness into a local
deployment rehearsal. It packages one case report into a ``sim://`` artifact,
replays a mock deploy/load/inference flow, checks target budgets, and writes an
auditable JSON record under ``analysis/local_deployment_sim/``.

It never talks to real hardware. The safety contract is explicit in every
manifest and audit record: no adb, devicectl, ROS, MAVLink, CAN, serial, or
network device calls are made.

Usage
-----
::

    python scripts/local_deployment_sim.py --case industrial --quick --steps 8
    python scripts/local_deployment_sim.py --case all --target jetson_orin_cpu --quick --steps 8
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
import pathlib
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOCAL_DEPLOYMENT_SIM_SCHEMA: Dict[str, Any] = {
    "version": 1,
    "fields": [
        "case",
        "target",
        "source_report",
        "manifest",
        "audit",
        "budget_check",
        "status",
    ],
}

CASE_NAMES = ["quadruped", "drone", "industrial", "battery"]
_CASE_FUNCS_CACHE: Optional[Dict[str, Any]] = None

FORBIDDEN_INTERFACES = [
    "adb",
    "xcrun devicectl",
    "ros2",
    "mavlink",
    "can",
    "serial",
    "bluetooth",
    "payment",
    "sms",
]


@dataclass(frozen=True)
class SimTarget:
    """A local mock target profile used for deployment budget checks."""

    name: str
    arch: str
    accelerator: str
    latency_budget_ms: float
    memory_budget_kb: int
    supported_cases: List[str]
    notes: str


TARGETS: Dict[str, SimTarget] = {
    "local_cpu_smoke": SimTarget(
        name="local_cpu_smoke",
        arch="x86_64/aarch64-host",
        accelerator="cpu",
        latency_budget_ms=250.0,
        memory_budget_kb=128 * 1024,
        supported_cases=["quadruped", "drone", "industrial", "battery"],
        notes="Generous local smoke target; proves packaging and local inference path.",
    ),
    "jetson_orin_cpu": SimTarget(
        name="jetson_orin_cpu",
        arch="aarch64",
        accelerator="cpu",
        latency_budget_ms=10.0,
        memory_budget_kb=8 * 1024 * 1024,
        supported_cases=["quadruped", "drone", "industrial", "battery"],
        notes="Jetson-like CPU budget without touching a Jetson or CUDA driver.",
    ),
    "mcu_tiny": SimTarget(
        name="mcu_tiny",
        arch="cortex-m4",
        accelerator="none",
        latency_budget_ms=1.0,
        memory_budget_kb=64,
        supported_cases=["industrial", "battery"],
        notes="Strict MCU-style budget check; expected to fail PyTorch smoke artifacts.",
    ),
}


def _preload_optional_cudss() -> None:
    """Best-effort preload for host PyTorch wheels that depend on libcudss.

    Some Jetson/CUDA PyTorch wheels have a hard dynamic dependency on
    ``libcudss.so.0``. The repository already documents the local install path
    in ``scripts/jetson_cuda_env.sh``; preloading the library here lets the
    simulator import the synthetic harness on hosts where the library exists
    but ``LD_LIBRARY_PATH`` was not exported before Python started.
    """
    candidates = []
    env_home = os.environ.get("CUDSS_HOME")
    if env_home:
        candidates.append(pathlib.Path(env_home) / "lib" / "libcudss.so.0")
    candidates.append(
        pathlib.Path.home()
        / ".local"
        / "opt"
        / "libcudss-linux-aarch64-0.8.0.10_cuda12-archive"
        / "lib"
        / "libcudss.so.0"
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                ctypes.CDLL(str(candidate), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            except OSError:
                pass
            return


def _case_funcs() -> Dict[str, Any]:
    """Import the source harness lazily so pure manifest tests need no torch."""
    global _CASE_FUNCS_CACHE
    if _CASE_FUNCS_CACHE is None:
        _preload_optional_cudss()
        from scripts.experiment_device_control_cases import CASE_FUNCS

        _CASE_FUNCS_CACHE = CASE_FUNCS
    return _CASE_FUNCS_CACHE


def _utc_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _param_memory_kb(params: Optional[int]) -> Optional[float]:
    if params is None:
        return None
    return params * 4 / 1024.0


def _mean_ok_values(reports: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    values = [
        float(r[key])
        for r in reports
        if r.get("status") == "ok" and key in r and r[key] is not None
    ]
    if not values:
        return None
    return statistics.fmean(values)


def _run_source_case(
    case: str,
    *,
    seeds: int,
    steps: int,
    quick: bool,
    battery_mode: str,
    refine_steps: int,
) -> Dict[str, Any]:
    """Run the synthetic source harness and return per-seed reports."""
    func = _case_funcs()[case]
    per_seed: List[Dict[str, Any]] = []
    for idx in range(seeds):
        seed = 42 + idx * 1000
        steps_per_seed = max(8, steps // max(1, seeds))
        try:
            if case == "battery":
                report = func(
                    seed=seed,
                    steps=steps_per_seed,
                    quick=quick,
                    battery_mode=battery_mode,
                    refine_steps=refine_steps,
                )
            else:
                report = func(seed=seed, steps=steps_per_seed, quick=quick)
        except Exception as exc:  # pragma: no cover - CLI safety net
            report = {
                "case": case,
                "seed": seed,
                "steps": steps_per_seed,
                "status": "error",
                "notes": f"{type(exc).__name__}: {exc}",
            }
        per_seed.append(report)

    ok_reports = [r for r in per_seed if r.get("status") == "ok"]
    return {
        "case": case,
        "per_seed": per_seed,
        "summary": {
            "ok_seeds": len(ok_reports),
            "total_seeds": len(per_seed),
            "mean_primary_metric_value": _mean_ok_values(per_seed, "primary_metric_value"),
            "mean_inference_ms": _mean_ok_values(per_seed, "inference_ms"),
            "mean_params": _mean_ok_values(per_seed, "params"),
        },
    }


def build_manifest(case: str, target: SimTarget, source_report: Dict[str, Any]) -> Dict[str, Any]:
    """Build a deployment manifest for a local ``sim://`` artifact."""
    summary = source_report["summary"]
    params = summary.get("mean_params")
    params_int = int(params) if params is not None else None
    return {
        "schema_version": 1,
        "generated_utc": _utc_now(),
        "mode": "local_simulation",
        "artifact": {
            "uri": f"sim://lnn/{case}/{target.name}",
            "format": "torch-in-memory-smoke",
            "case": case,
            "parameter_count": params_int,
            "estimated_fp32_memory_kb": _param_memory_kb(params_int),
        },
        "target": asdict(target),
        "safety": {
            "real_device_access": False,
            "submit_allowed": False,
            "network_device_calls": False,
            "forbidden_interfaces": FORBIDDEN_INTERFACES,
            "note": "This manifest is a local rehearsal only; it is not a hardware deployment ticket.",
        },
    }


def budget_check(source_report: Dict[str, Any], target: SimTarget) -> Dict[str, Any]:
    """Compare source smoke metrics against a mock target profile."""
    summary = source_report["summary"]
    mean_latency = summary.get("mean_inference_ms")
    mean_params = summary.get("mean_params")
    memory_kb = _param_memory_kb(int(mean_params)) if mean_params is not None else None
    case = source_report["case"]

    checks = {
        "case_supported": case in target.supported_cases,
        "latency_ms": {
            "observed": mean_latency,
            "budget": target.latency_budget_ms,
            "pass": None if mean_latency is None else mean_latency <= target.latency_budget_ms,
        },
        "memory_kb": {
            "observed": memory_kb,
            "budget": target.memory_budget_kb,
            "pass": None if memory_kb is None else memory_kb <= target.memory_budget_kb,
        },
    }
    hard_bools = [checks["case_supported"]]
    for item in ("latency_ms", "memory_kb"):
        value = checks[item]["pass"]
        if value is not None:
            hard_bools.append(bool(value))
    status = "pass" if all(hard_bools) else "fail"
    if summary.get("ok_seeds", 0) == 0:
        status = "blocked"
    return {"status": status, "checks": checks}


def simulate_deploy(case: str, manifest: Dict[str, Any], source_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Replay a local-only deployment lifecycle and return audit events."""
    artifact_uri = manifest["artifact"]["uri"]
    audit: List[Dict[str, Any]] = []
    stages = [
        ("package", f"sealed {artifact_uri} manifest"),
        ("transfer", "mock transfer to local sandbox namespace"),
        ("load", "mock runtime loaded in current Python process"),
        ("warmup", "single synthetic warmup request replayed"),
        ("inference_loop", "synthetic inference metrics read from source report"),
        ("rollback_check", "mock rollback available; no persistent state changed"),
    ]
    for stage, detail in stages:
        audit.append(
            {
                "ts_utc": _utc_now(),
                "stage": stage,
                "status": "ok",
                "detail": detail,
                "real_device_access": False,
                "interfaces_used": ["python", "filesystem", "sim://"],
            }
        )
        time.sleep(0.001)
    if source_report["summary"].get("ok_seeds", 0) == 0:
        audit.append(
            {
                "ts_utc": _utc_now(),
                "stage": "source_report",
                "status": "blocked",
                "detail": f"{case} produced no ok source reports; deployment was not promoted.",
                "real_device_access": False,
                "interfaces_used": ["python", "filesystem", "sim://"],
            }
        )
    return audit


def run_local_deployment_sim(
    case: str,
    *,
    target_name: str,
    seeds: int,
    steps: int,
    quick: bool,
    battery_mode: str = "single",
    refine_steps: int = 10,
) -> Dict[str, Any]:
    """Run one case through the local deployment simulator."""
    if target_name not in TARGETS:
        raise ValueError(f"unknown target {target_name!r}; choose one of {sorted(TARGETS)}")
    if case not in CASE_NAMES:
        raise ValueError(f"unknown case {case!r}; choose one of {CASE_NAMES}")

    target = TARGETS[target_name]
    source_report = _run_source_case(
        case,
        seeds=seeds,
        steps=steps,
        quick=quick,
        battery_mode=battery_mode,
        refine_steps=refine_steps,
    )
    manifest = build_manifest(case, target, source_report)
    audit = simulate_deploy(case, manifest, source_report)
    budgets = budget_check(source_report, target)
    status = budgets["status"]
    if source_report["summary"].get("ok_seeds", 0) == 0:
        status = "blocked"
    return {
        "schema": LOCAL_DEPLOYMENT_SIM_SCHEMA,
        "case": case,
        "target": target.name,
        "source_report": source_report,
        "manifest": manifest,
        "audit": audit,
        "budget_check": budgets,
        "status": status,
    }


def _write_payload(out_dir: pathlib.Path, payload: Dict[str, Any]) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    out_path = out_dir / f"{stamp}_local_deployment_sim_{payload['target']}_{payload['case']}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--case", choices=CASE_NAMES + ["all"], default="industrial")
    parser.add_argument("--target", choices=sorted(TARGETS), default="local_cpu_smoke")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--battery-mode", choices=["single", "transformable"], default="single")
    parser.add_argument("--refine-steps", type=int, default=10)
    parser.add_argument("--out-dir", default="analysis/local_deployment_sim")
    args = parser.parse_args()

    cases = CASE_NAMES if args.case == "all" else [args.case]
    out_dir = pathlib.Path(args.out_dir)
    summaries: List[Dict[str, Any]] = []
    for case in cases:
        payload = run_local_deployment_sim(
            case,
            target_name=args.target,
            seeds=args.seeds,
            steps=args.steps,
            quick=args.quick,
            battery_mode=args.battery_mode,
            refine_steps=args.refine_steps,
        )
        out_path = _write_payload(out_dir, payload)
        budgets = payload["budget_check"]
        summaries.append(
            {
                "case": case,
                "target": args.target,
                "status": payload["status"],
                "budget_status": budgets["status"],
                "path": str(out_path),
            }
        )
        print(f"[{case}] status={payload['status']} budget={budgets['status']} -> {out_path}")

    master = out_dir / "latest_local_deployment_simulation.json"
    with open(master, "w") as f:
        json.dump({"generated_utc": _utc_now(), "runs": summaries}, f, indent=2)
    print(f"Summary -> {master}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
