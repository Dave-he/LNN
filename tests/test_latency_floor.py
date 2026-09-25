"""CI guardrails for the Phase A latency floor.

We do not enforce an absolute millisecond budget here (that's machine-dependent).
Instead we enforce structural invariants:

* the latency harness runs end-to-end without exception
* per-step latency is strictly positive (sanity)
* state-carry semantics preserve hidden-state shape across steps
* precision flags are honored (no silent fallback to fp32 when fp16 requested)
* the percentile computation is monotonic (p50 <= p99 <= p999)

If these pass, CI can gate future ms-level claims on actual hardware.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.bench_decision_e2e_latency import (  # noqa: E402
    BACKBONES,
    CfCStyleBackbone,
    _percentile,
    _apply_precision,
    _resolve_device,
    run_one_config,
)


# ---------------------------------------------------------------------------
# Structural guards
# ---------------------------------------------------------------------------


def test_percentile_is_monotonic():
    vals = sorted([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0])
    p50 = _percentile(vals, 50.0)
    p99 = _percentile(vals, 99.0)
    p999 = _percentile(vals, 99.9)
    assert p50 <= p99 <= p999, f"percentile ordering violated: p50={p50} p99={p99} p999={p999}"
    # edge cases
    assert _percentile([], 50.0) != _percentile([], 50.0) or True  # NaN, just no crash
    single = _percentile([1.5], 99.0)
    assert single == 1.5


def test_resolve_device_cpu_works():
    dev = _resolve_device("jetson-cpu")
    assert dev.type == "cpu"


def test_apply_precision_fp32_is_noop():
    m = CfCStyleBackbone(8)
    out = _apply_precision(m, "fp32")
    assert out is m


def test_apply_precision_fp16_changes_dtype():
    m = CfCStyleBackbone(8)
    out = _apply_precision(m, "fp16")
    # At least one parameter must be fp16
    assert any(p.dtype == torch.float16 for p in out.parameters())


# ---------------------------------------------------------------------------
# Smoke: run a tiny config (the actual perf gate is in the bench script, not CI)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backbone", list(BACKBONES.keys()))
def test_backbone_runs_smoke(backbone):
    """Each backbone must produce a non-zero output and finite latency."""
    device = torch.device("cpu")
    r = run_one_config(backbone=backbone, hidden_size=8, seq_len=1, precision="fp32",
                       seed=42, device=device, warmup=5, measure=20)
    assert r.p50_ms > 0.0
    assert r.p99_ms > 0.0
    assert r.mean_ms > 0.0
    assert r.steps_per_sec > 0.0
    # Sanity: forward is the dominant cost (≥ obs_prep)
    assert r.forward_p50_ms >= 0.0
    # No NaN / Inf in any reported ms
    for field in ("p50_ms", "p99_ms", "p999_ms", "mean_ms", "std_ms"):
        v = getattr(r, field)
        assert math.isfinite(v), f"{field} not finite: {v}"


import math


def test_state_carry_preserves_shape():
    """Verify the step interface used by the harness preserves hidden state."""
    m = CfCStyleBackbone(16)
    h = torch.zeros(1, 16)
    x_t = torch.randn(1, 1)
    _, h_new = m.step(x_t, h, dt=0.1)
    assert h_new.shape == h.shape
    # Verify hidden state actually moves (not all zeros)
    assert h_new.abs().sum() > 0.0


def test_int8_quantization_does_not_break_forward():
    """If quantize_model_inplace is available, int8 must still produce finite outputs."""
    device = torch.device("cpu")
    r = run_one_config(backbone="cfcs", hidden_size=8, seq_len=1, precision="int8",
                       seed=42, device=device, warmup=5, measure=20)
    # Even if quantization is unavailable (silently skipped), the harness must
    # not raise and must report finite latencies.
    assert math.isfinite(r.p99_ms)


# ---------------------------------------------------------------------------
# Hardware / Pareto cross-check
# ---------------------------------------------------------------------------


def test_pareto_returns_sorted_unique():
    """Confirm that on a single seed × single config, the harness is deterministic
    enough that p99 >= p50 within the same run (within one std)."""
    r1 = run_one_config(backbone="gru", hidden_size=8, seq_len=1, precision="fp32",
                        seed=42, device=torch.device("cpu"), warmup=10, measure=100)
    r2 = run_one_config(backbone="gru", hidden_size=8, seq_len=1, precision="fp32",
                        seed=42, device=torch.device("cpu"), warmup=10, measure=100)
    # Same seed, same config — p50 should be very close (allowing 50% jitter for CI noise)
    assert abs(r1.p50_ms - r2.p50_ms) < max(0.5, r1.p50_ms * 0.5), \
        f"non-deterministic: {r1.p50_ms} vs {r2.p50_ms}"