"""Round 15 (N19) — tests for hybrid_gate teacher distillation.

Validates that :class:`DualStageDistiller` correctly handles
``teacher_retention_kind='hybrid_gate'`` and produces a compressible student.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.distillation import (
    ActivationAlignedHybridGateCfCNetwork,
    DistillConfig,
    DualStageDistiller,
)


def _seed(seed: int = 42):
    torch.manual_seed(seed)


def _toy_data(n_samples=64, seq_len=16, n_feat=4, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n_samples, seq_len, n_feat)
    y = torch.randn(n_samples, seq_len, 1)
    return x, y


# ---------------------------------------------------------------------------
# ActivationAlignedHybridGateCfCNetwork
# ---------------------------------------------------------------------------


def test_activation_aligned_hybrid_gate_forward_shape():
    _seed()
    net = ActivationAlignedHybridGateCfCNetwork(input_size=4, hidden_size=8, output_size=1)
    x = torch.randn(5, 12, 4)
    y_seq, h_seq = net(x)
    assert y_seq.shape == (5, 12, 1)
    assert h_seq.shape == (5, 12, 8)


def test_activation_aligned_hybrid_gate_has_input_dep_alpha():
    """Confirm hybrid_gate teacher retains gate_mlps for input-dep alpha."""
    _seed()
    net = ActivationAlignedHybridGateCfCNetwork(input_size=4, hidden_size=8, output_size=1)
    assert net.cell.gate_mlps is not None
    assert net.cell.retention_kind == "hybrid_gate"


# ---------------------------------------------------------------------------
# DistillConfig with hybrid_gate teacher
# ---------------------------------------------------------------------------


def test_distill_config_hybrid_gate_default():
    """Default DistillConfig should still use cfc teacher (backward compat)."""
    cfg = DistillConfig(input_size=4, output_size=1)
    assert cfg.teacher_retention_kind == "cfc"


def test_distill_config_hybrid_gate_explicit():
    cfg = DistillConfig(
        input_size=4, output_size=1,
        teacher_retention_kind="hybrid_gate",
    )
    assert cfg.teacher_retention_kind == "hybrid_gate"


# ---------------------------------------------------------------------------
# DualStageDistiller with hybrid_gate teacher
# ---------------------------------------------------------------------------


def test_distiller_creates_hybrid_gate_teacher():
    """DualStageDistiller should accept hybrid_gate teacher_retention_kind."""
    _seed()
    cfg = DistillConfig(
        input_size=4, output_size=1, teacher_hidden=8,
        teacher_retention_kind="hybrid_gate",
    )
    d = DualStageDistiller(cfg)
    assert d.teacher_retention_kind == "hybrid_gate"
    assert d.teacher.cell.retention_kind == "hybrid_gate"


def test_distiller_rejects_unknown_teacher_kind():
    from lnn.core.distillation import DualStageDistiller
    cfg = DistillConfig(
        input_size=4, output_size=1, teacher_hidden=8,
        teacher_retention_kind="bogus",
    )
    try:
        DualStageDistiller(cfg)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for unknown teacher_retention_kind")


def test_pareto_sweep_with_hybrid_gate_teacher():
    """End-to-end Pareto sweep with hybrid_gate teacher should run cleanly
    and produce (N+1) ParetoPoints, with student strictly smaller than teacher."""
    _seed()
    cfg = DistillConfig(
        input_size=4, output_size=1, teacher_hidden=12,
        student_hiddens=(4, 6, 8), epochs=3, batch=8, lr=1e-2,
        teacher_retention_kind="hybrid_gate",
    )
    d = DualStageDistiller(cfg)
    x_tr, y_tr = _toy_data(n_samples=64, seq_len=12, seed=0)
    x_te, y_te = _toy_data(n_samples=32, seq_len=12, seed=1)
    results = d.run_pareto_sweep(x_tr, y_tr, x_te, y_te)
    assert len(results) == 1 + len(cfg.student_hiddens)
    teacher_params = results[0].params
    student_params = [r.params for r in results[1:]]
    assert all(p < teacher_params for p in student_params)
    # All MSEs finite
    import math
    for r in results:
        assert math.isfinite(r.test_mse)
