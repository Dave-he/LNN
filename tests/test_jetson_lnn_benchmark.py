"""Unit tests for scripts/jetson_lnn_benchmark.py.

These tests run in two modes:

1. **torch present** (the LNN dev env): the script runs end-to-end in
   ``--quick`` mode and produces a real Pareto JSON with both CfC-style
   and GRU rows. We assert the JSON shape and that at least one
   ``pareto_front: true`` row exists.

2. **torch missing** (a stock Jetson without the PyTorch wheel): the
   script must report ``status: "skipped"`` and not crash.

The script also exposes the lower-level ``mark_pareto_front`` helper,
which we test directly against synthetic result dicts so the
Pareto-dominance logic is covered even when torch is unavailable.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "jetson_lnn_benchmark.py"


# ---------------------------------------------------------------------------
# Imports from the script (do not require torch)
# ---------------------------------------------------------------------------

def _import_helpers():
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("jetson_lnn_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_mark_pareto_front_keeps_non_dominated() -> None:
    """The non-dominated row (lowest MSE, highest throughput) survives."""
    module = _import_helpers()
    results = [
        {"name": "CfC", "test_mse": 0.10, "parameters": 1000,
         "train_seconds": 5.0, "inference_steps_per_sec": 1000.0},
        {"name": "CfC", "test_mse": 0.05, "parameters": 1000,
         "train_seconds": 5.0, "inference_steps_per_sec": 1000.0},
        {"name": "GRU", "test_mse": 0.20, "parameters": 800,
         "train_seconds": 3.0, "inference_steps_per_sec": 5000.0},
    ]
    marked = module.mark_pareto_front(results)
    pareto_names = {row["name"] for row in marked if row.get("pareto_front")}
    # The 0.05 MSE CfC is strictly better than the 0.10 MSE CfC on MSE only;
    # the GRU is strictly better on throughput and train time. Both must
    # survive; the dominated 0.10 MSE CfC must be marked False.
    assert "CfC" in pareto_names
    assert "GRU" in pareto_names
    by_mse = {row["test_mse"]: row for row in marked if row["name"] == "CfC"}
    assert by_mse[0.10]["pareto_front"] is False
    assert by_mse[0.05]["pareto_front"] is True


def test_mark_pareto_front_handles_single_row() -> None:
    """A single row is always Pareto-optimal."""
    module = _import_helpers()
    results = [
        {"name": "LTC", "test_mse": 0.5, "parameters": 200,
         "train_seconds": 1.0, "inference_steps_per_sec": 100.0},
    ]
    marked = module.mark_pareto_front(results)
    assert len(marked) == 1
    assert marked[0]["pareto_front"] is True


def test_parse_int_list_default_and_override() -> None:
    """Empty / None inputs use the default; comma-separated parses."""
    module = _import_helpers()
    assert module.parse_int_list(None, [8, 16]) == [8, 16]
    assert module.parse_int_list("", [8, 16]) == [8, 16]
    assert module.parse_int_list("8, 16, 32", [99]) == [8, 16, 32]
    with pytest.raises(ValueError):
        module.parse_int_list(" , , ", [8, 16])


def test_dominates_requires_strictly_better() -> None:
    """Equal-on-everything does NOT count as domination."""
    module = _import_helpers()
    a = {"test_mse": 0.1, "parameters": 100, "train_seconds": 1.0,
         "inference_steps_per_sec": 1000.0}
    b = dict(a)  # identical
    assert module.dominates(a, b) is False
    # Now strictly better on one axis:
    a2 = dict(a, test_mse=0.05)
    assert module.dominates(a2, a) is True
    # Worse on one axis:
    a3 = dict(a, test_mse=0.2)
    assert module.dominates(a3, a) is False


def test_aggregate_seeds_groups_by_model_hidden_seq() -> None:
    """iter#35: 6 per-seed rows collapse to 2 aggregated rows."""
    module = _import_helpers()
    per_seed = [
        {"name": "CfC", "hidden_size": 8, "seq_len": 16, "seed": 42,
         "parameters": 329, "test_mse": 0.50, "inference_steps_per_sec": 40000.0,
         "train_seconds": 1.0},
        {"name": "CfC", "hidden_size": 8, "seq_len": 16, "seed": 123,
         "parameters": 329, "test_mse": 0.60, "inference_steps_per_sec": 42000.0,
         "train_seconds": 1.2},
        {"name": "CfC", "hidden_size": 8, "seq_len": 16, "seed": 7,
         "parameters": 329, "test_mse": 0.55, "inference_steps_per_sec": 38000.0,
         "train_seconds": 1.1},
        {"name": "GRU", "hidden_size": 8, "seq_len": 16, "seed": 42,
         "parameters": 273, "test_mse": 0.60, "inference_steps_per_sec": 200000.0,
         "train_seconds": 0.3},
        {"name": "GRU", "hidden_size": 8, "seq_len": 16, "seed": 123,
         "parameters": 273, "test_mse": 0.65, "inference_steps_per_sec": 210000.0,
         "train_seconds": 0.4},
        {"name": "GRU", "hidden_size": 8, "seq_len": 16, "seed": 7,
         "parameters": 273, "test_mse": 0.62, "inference_steps_per_sec": 195000.0,
         "train_seconds": 0.35},
    ]
    aggregated = module.aggregate_seeds(per_seed)
    assert len(aggregated) == 2  # 2 (name, hidden, seq) groups
    by_name = {row["name"]: row for row in aggregated}
    # CfC group
    cfc = by_name["CfC"]
    assert cfc["hidden_size"] == 8 and cfc["seq_len"] == 16
    assert cfc["parameters"] == 329
    assert sorted(cfc["seeds"]) == [7, 42, 123]
    # Mean of 0.50, 0.60, 0.55 = 0.55; std ≈ 0.05
    assert abs(cfc["test_mse"]["mean"] - 0.55) < 1e-9
    assert cfc["test_mse"]["std"] > 0.04 and cfc["test_mse"]["std"] < 0.06
    assert cfc["test_mse"]["n_seeds"] == 3
    # GRU group: steps/sec mean ~201667, std should be > 0
    gru = by_name["GRU"]
    assert gru["inference_steps_per_sec"]["n_seeds"] == 3
    assert gru["inference_steps_per_sec"]["std"] > 0
    # Single-seed case: std must be 0.0 (not crash on stdev of 1)
    single = per_seed[:1]
    agg_single = module.aggregate_seeds(single)
    assert len(agg_single) == 1
    assert agg_single[0]["test_mse"]["std"] == 0.0
    assert agg_single[0]["test_mse"]["n_seeds"] == 1


def test_aggregate_seeds_then_mark_pareto_front_uses_mean() -> None:
    """iter#35: aggregated dominance check uses .mean sub-fields."""
    module = _import_helpers()
    # Three aggregated rows; PDNA h=8 has mean 0.45 (better) but high std.
    # CfC h=8 has mean 0.50 (worse) but low std. They both go on Pareto
    # because PDNA doesn't strictly dominate CfC (mean 0.45 < 0.50 is
    # strictly better, but parameters 418 > 329). Verify the math.
    aggregated = [
        {"name": "PDNA", "hidden_size": 8, "seq_len": 16, "parameters": 418,
         "test_mse": {"mean": 0.45, "std": 0.10, "n_seeds": 3},
         "train_seconds": {"mean": 1.0, "std": 0.0, "n_seeds": 3},
         "inference_steps_per_sec": {"mean": 50000.0, "std": 0.0, "n_seeds": 3}},
        {"name": "CfC", "hidden_size": 8, "seq_len": 16, "parameters": 329,
         "test_mse": {"mean": 0.50, "std": 0.01, "n_seeds": 3},
         "train_seconds": {"mean": 1.0, "std": 0.0, "n_seeds": 3},
         "inference_steps_per_sec": {"mean": 40000.0, "std": 0.0, "n_seeds": 3}},
        {"name": "GRU", "hidden_size": 8, "seq_len": 16, "parameters": 273,
         "test_mse": {"mean": 0.60, "std": 0.02, "n_seeds": 3},
         "train_seconds": {"mean": 0.3, "std": 0.0, "n_seeds": 3},
         "inference_steps_per_sec": {"mean": 200000.0, "std": 0.0, "n_seeds": 3}},
    ]
    marked = module.mark_pareto_front_aggregated(aggregated)
    pareto_names = {row["name"] for row in marked if row.get("pareto_front")}
    # All three are Pareto: PDNA has best MSE but more params, CfC middle,
    # GRU has best throughput. None strictly dominates another on all 4 axes.
    assert pareto_names == {"PDNA", "CfC", "GRU"}


def test_looks_like_cuda_runtime_error_detection() -> None:
    """CUDA OOM and cublas asserts are detected; plain Python errors are not.

    The script requires BOTH a 'cuda' substring in the lowercased text AND one
    of the marker keywords (cublas / cudacachingallocator / nvml / etc.). A
    bare 'cuDNN' message without 'cuda' is intentionally NOT detected because
    the fallback handler also wants to confirm it's CUDA-side, not a
    cuDNN-only assertion in CPU code.
    """
    module = _import_helpers()
    assert module.looks_like_cuda_runtime_error(
        RuntimeError("CUDA out of memory. Tried to allocate 1.5 GiB")
    ) is True
    # cuDNN with "cuda" in the surrounding message IS detected (cublas marker).
    assert module.looks_like_cuda_runtime_error(
        RuntimeError("CUDA runtime error: cublas not initialized")
    ) is True
    # A pure cuDNN exception without "cuda" in the message is NOT detected.
    assert module.looks_like_cuda_runtime_error(
        RuntimeError("cuDNN error: CUDNN_STATUS_NOT_SUPPORTED")
    ) is False
    # Plain Python error: not detected.
    assert module.looks_like_cuda_runtime_error(
        ValueError("shape mismatch in linear layer")
    ) is False
    # 'accelerator' marker with 'cuda' prefix IS detected.
    assert module.looks_like_cuda_runtime_error(
        RuntimeError("CUDA accelerator not available, falling back to CPU")
    ) is True


# ---------------------------------------------------------------------------
# End-to-end CLI smoke — only when torch is importable
# ---------------------------------------------------------------------------

def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_cli_quick_pareto_produces_valid_json(tmp_path: pathlib.Path) -> None:
    """Run --quick --cpu --pareto end-to-end and validate the JSON shape."""
    run_date = "2026-06-09_test_quick"
    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SCRIPT),
        "--quick", "--cpu", "--pareto",
        "--date", run_date,
    ]
    result = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"benchmark failed (rc={result.returncode}):\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    json_path = out_dir / f"{run_date}_lnn_benchmark.json"
    assert json_path.exists(), f"missing {json_path}"
    payload = json.loads(json_path.read_text())
    assert payload["status"] in {"ok", "ok_cpu_fallback"}
    assert payload["experiment"] == "jetson_lnn_pareto_sweep"
    assert payload["device"] == "cpu"
    assert "environment" in payload
    assert "nv_tegra_release" in payload["environment"]
    assert len(payload["results"]) >= 2  # at least 2 (CfC + GRU) per config
    # At least one Pareto winner must exist.
    pareto = [row for row in payload["results"] if row.get("pareto_front")]
    assert len(pareto) >= 1
    # All result rows have the 4 mandatory numeric fields.
    for row in payload["results"]:
        for key in ("parameters", "test_mse", "train_seconds",
                    "inference_steps_per_sec"):
            assert isinstance(row[key], (int, float)), (
                f"row {row.get('name')} missing numeric {key}"
            )


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_cli_quick_single_run_no_pareto(tmp_path: pathlib.Path) -> None:
    """The non-Pareto (single-config) CLI also produces a valid JSON."""
    run_date = "2026-06-09_test_single"
    out_dir = ROOT / "analysis" / "jetson"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SCRIPT),
        "--quick", "--cpu",
        "--date", run_date,
    ]
    result = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"benchmark failed (rc={result.returncode}):\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    json_path = out_dir / f"{run_date}_lnn_benchmark.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["status"] in {"ok", "ok_cpu_fallback"}
    # iter#34: non-Pareto now has 4 rows (CfCStyle + LTC + PDNAPulse + GRU),
    # no hidden_size / seq_len / pareto_front fields.
    assert len(payload["results"]) == 4
    names = {row["name"] for row in payload["results"]}
    assert "CfCStyle" in names
    assert "LTC" in names
    assert "PDNAPulse" in names
    assert "GRU" in names
    for row in payload["results"]:
        assert "pareto_front" not in row
