"""Round 21 (N21) — tests for hybrid_gate student in dual-stage distiller.

Validates that DualStageDistiller correctly supports hybrid_gate students
(N21 round-trip distillation: hybrid_gate teacher → hybrid_gate student).
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.distillation import (
    ActivationAlignedCfCNetwork,
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
# DistillConfig
# ---------------------------------------------------------------------------


def test_distill_config_student_default_is_cfc():
    """Backward compat: default student_retention_kind is 'cfc'."""
    cfg = DistillConfig(input_size=4, output_size=1)
    assert cfg.student_retention_kind == "cfc"


def test_distill_config_student_hybrid_gate_explicit():
    cfg = DistillConfig(
        input_size=4, output_size=1,
        student_retention_kind="hybrid_gate",
    )
    assert cfg.student_retention_kind == "hybrid_gate"


# ---------------------------------------------------------------------------
# DualStageDistiller
# ---------------------------------------------------------------------------


def test_distiller_rejects_unknown_student_kind():
    _seed()
    cfg = DistillConfig(
        input_size=4, output_size=1, teacher_hidden=8,
        student_retention_kind="bogus",
    )
    try:
        DualStageDistiller(cfg)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for unknown student_retention_kind")


def test_make_student_uses_hybrid_gate_when_configured():
    _seed()
    cfg = DistillConfig(
        input_size=4, output_size=1, teacher_hidden=8,
        student_retention_kind="hybrid_gate",
    )
    d = DualStageDistiller(cfg)
    student, _ = d._make_student(hidden=4)
    assert isinstance(student, ActivationAlignedHybridGateCfCNetwork)
    assert student.cell.retention_kind == "hybrid_gate"


def test_make_student_uses_cfc_when_configured():
    _seed()
    cfg = DistillConfig(
        input_size=4, output_size=1, teacher_hidden=8,
        student_retention_kind="cfc",
    )
    d = DualStageDistiller(cfg)
    student, _ = d._make_student(hidden=4)
    assert isinstance(student, ActivationAlignedCfCNetwork)


def test_pareto_sweep_with_hybrid_gate_student():
    """End-to-end: hybrid_gate teacher -> hybrid_gate student, h in {4, 8}."""
    _seed()
    cfg = DistillConfig(
        input_size=4, output_size=1, teacher_hidden=12,
        student_hiddens=(4, 6, 8), epochs=3, batch=8, lr=1e-2,
        teacher_retention_kind="hybrid_gate",
        student_retention_kind="hybrid_gate",
    )
    d = DualStageDistiller(cfg)
    x_tr, y_tr = _toy_data(n_samples=64, seq_len=12, seed=0)
    x_te, y_te = _toy_data(n_samples=32, seq_len=12, seed=1)
    results = d.run_pareto_sweep(x_tr, y_tr, x_te, y_te)
    assert len(results) == 1 + len(cfg.student_hiddens)
    for p in results[1:]:
        # Verify students are actually hybrid_gate
        student, _ = d.students[p.student_hidden]
        assert student.cell.retention_kind == "hybrid_gate"
        # All student params < teacher params
        assert p.params < results[0].params
    import math
    for r in results:
        assert math.isfinite(r.test_mse)


def test_round_trip_hybrid_gate_to_hybrid_gate_loss_decreases():
    """Stage 1 distillation should drive loss down for round-trip config."""
    _seed()
    cfg = DistillConfig(
        input_size=4, output_size=1, teacher_hidden=8,
        student_hiddens=(4,), epochs=8, batch=8, lr=1e-2,
        teacher_retention_kind="hybrid_gate",
        student_retention_kind="hybrid_gate",
    )
    d = DualStageDistiller(cfg)
    x, y = _toy_data(n_samples=64, seq_len=12, seed=0)
    # Pre-train teacher briefly
    opt = torch.optim.Adam(d.teacher.parameters(), lr=1e-2)
    for _ in range(4):
        for b in range(0, 64, 8):
            xb, yb = x[b:b + 8], y[b:b + 8]
            opt.zero_grad()
            y_t, _ = d.teacher(xb)
            torch.nn.functional.mse_loss(y_t, yb).backward()
            opt.step()
    # Snapshot loss
    import torch.nn.functional as F
    student, proj = d._make_student(4)
    with torch.no_grad():
        _, h_t = d.teacher(x)
        y_s, h_s = student(x)
        loss0 = F.mse_loss(y_s, y) + 0.5 * F.mse_loss(h_s, proj(h_t))
    d.train_stage1(student, proj, x, y)
    with torch.no_grad():
        y_s, h_s = student(x)
        loss1 = F.mse_loss(y_s, y) + 0.5 * F.mse_loss(h_s, proj(h_t))
    assert loss1 < loss0, f"loss did not decrease: {loss0:.4f} -> {loss1:.4f}"
