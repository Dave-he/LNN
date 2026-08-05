"""Round 284 — tests for 'hybrid_gate' (input-dependent α) in MemoryFusionCfCCell.

Validates that the new hybrid_gate mode provides *true* conditional gating
by making α a function of (x_t, dt) via a per-branch MLP, not a static parameter.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.memory_fusion_cfc import MemoryFusionCfCCell, _VALID_RETENTION


def _seed(seed: int = 42):
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


def test_hybrid_gate_in_valid_set():
    assert "hybrid_gate" in _VALID_RETENTION


def test_init_hybrid_gate_creates_gate_mlps():
    _seed()
    cell = MemoryFusionCfCCell(input_size=4, hidden_size=8, retention_kind="hybrid_gate")
    assert cell.gate_mlps is not None
    assert len(cell.gate_mlps) == 1  # n_tau=1
    # gate_mlp input dim = input_size + 1
    assert cell.gate_mlps[0][0].in_features == 5  # 4 + 1
    assert cell.gate_mlps[0][0].out_features == 8  # branch_dim


def test_init_hybrid_gate_alpha_is_none():
    """hybrid_gate uses gate_mlps (input-dep), not alpha (static) like hybrid."""
    _seed()
    cell = MemoryFusionCfCCell(input_size=4, hidden_size=8, retention_kind="hybrid_gate")
    assert cell.alpha is None


# ---------------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------------


def test_forward_shape_hybrid_gate():
    _seed()
    cell = MemoryFusionCfCCell(input_size=3, hidden_size=8, retention_kind="hybrid_gate")
    x = torch.randn(5, 3)
    h = torch.randn(5, 8)
    dt = torch.ones(5, 1)
    out = cell(x, h, dt=dt)
    assert out.shape == (5, 8)


def test_forward_shape_multi_tau_hybrid_gate():
    _seed()
    cell = MemoryFusionCfCCell(input_size=3, hidden_size=10, retention_kind="hybrid_gate", n_tau=3)
    x = torch.randn(4, 3)
    h = torch.randn(4, 10)
    dt = torch.ones(4, 1)
    out = cell(x, h, dt=dt)
    assert out.shape == (4, 10)


# ---------------------------------------------------------------------------
# Conditional gating: α varies with input and dt
# ---------------------------------------------------------------------------


def test_alpha_depends_on_x():
    """After init, varying x_t should produce different α outputs (via MLP)."""
    _seed()
    cell = MemoryFusionCfCCell(input_size=4, hidden_size=8, retention_kind="hybrid_gate")
    with torch.no_grad():
        a_x1 = cell.gate_mlps[0](torch.cat([torch.zeros(1, 4), torch.ones(1, 1)], dim=-1))
        a_x2 = cell.gate_mlps[0](torch.cat([torch.ones(1, 4) * 5, torch.ones(1, 1)], dim=-1))
    diff = (a_x1 - a_x2).abs().mean().item()
    # After init the spread is small but non-zero (gain=0.1 init)
    assert diff > 1e-4, f"α should depend on x; spread={diff}"


def test_alpha_depends_on_dt():
    """After init, varying dt should produce different α outputs."""
    _seed()
    cell = MemoryFusionCfCCell(input_size=4, hidden_size=8, retention_kind="hybrid_gate")
    with torch.no_grad():
        a_dt1 = cell.gate_mlps[0](torch.cat([torch.zeros(1, 4), torch.tensor([[1.0]])], dim=-1))
        a_dt2 = cell.gate_mlps[0](torch.cat([torch.zeros(1, 4), torch.tensor([[5.0]])], dim=-1))
    diff = (a_dt1 - a_dt2).abs().mean().item()
    assert diff > 1e-4, f"α should depend on dt; spread={diff}"


def test_alpha_learns_conditional_gating_after_training():
    """After 20 training steps with irregular dt, α should clearly vary with both x and dt."""
    _seed()
    cell = MemoryFusionCfCCell(input_size=4, hidden_size=8, retention_kind="hybrid_gate")
    opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
    x_train = torch.randn(32, 4)
    h_train = torch.randn(32, 8)
    target = torch.randn(32, 8)
    for _ in range(20):
        opt.zero_grad()
        dt = torch.exp(torch.randn(32, 1) * 0.5 - 0.5 * 0.5**2)
        h_new = cell(x_train, h_train, dt=dt)
        loss = torch.nn.functional.mse_loss(h_new, target)
        loss.backward()
        opt.step()
    with torch.no_grad():
        a_x1 = cell.gate_mlps[0](torch.cat([torch.zeros(1, 4), torch.ones(1, 1)], dim=-1))
        a_x2 = cell.gate_mlps[0](torch.cat([torch.ones(1, 4) * 5, torch.ones(1, 1)], dim=-1))
        a_dt1 = cell.gate_mlps[0](torch.cat([torch.zeros(1, 4), torch.tensor([[1.0]])], dim=-1))
        a_dt2 = cell.gate_mlps[0](torch.cat([torch.zeros(1, 4), torch.tensor([[5.0]])], dim=-1))
    spread_x = (a_x1 - a_x2).abs().mean().item()
    spread_dt = (a_dt1 - a_dt2).abs().mean().item()
    # Should be substantially larger than init-time spread (init was ~0.0001)
    assert spread_x > 0.05, f"x-driven α spread too small after training: {spread_x}"
    assert spread_dt > 0.01, f"dt-driven α spread too small after training: {spread_dt}"


# ---------------------------------------------------------------------------
# Math sanity
# ---------------------------------------------------------------------------


def test_hybrid_gate_dt_zero_recovers_h_prev():
    """With dt → 0, k_tfp → 1 (CfC path killed by sigmoid(0)≈0.5 + exp(-0)=1 — but α is input-dep).

    Note: α may not → 1 because it's input-dependent. We just check that the
    forward is well-defined and finite.
    """
    _seed()
    cell = MemoryFusionCfCCell(input_size=2, hidden_size=4, retention_kind="hybrid_gate")
    x = torch.randn(2, 2)
    h_prev = torch.randn(2, 4)
    dt = torch.full((2, 1), 1e-6)
    out = cell(x, h_prev, dt=dt)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow_hybrid_gate():
    _seed()
    cell = MemoryFusionCfCCell(input_size=3, hidden_size=8, retention_kind="hybrid_gate")
    x = torch.randn(4, 3, requires_grad=True)
    h = torch.randn(4, 8)
    dt = torch.ones(4, 1)
    out = cell(x, h, dt=dt)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # gate_mlp gradient should also exist
    for mlp in cell.gate_mlps:
        for lin in mlp:
            if hasattr(lin, "weight"):
                assert lin.weight.grad is not None


def test_end_to_end_training_step_hybrid_gate():
    """5-step training should decrease loss."""
    _seed()
    cell = MemoryFusionCfCCell(input_size=3, hidden_size=8, retention_kind="hybrid_gate")
    opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
    x = torch.randn(8, 3)
    h_prev = torch.randn(8, 8)
    h_target = torch.randn(8, 8)
    dt = torch.ones(8, 1)
    loss0 = torch.nn.functional.mse_loss(cell(x, h_prev, dt=dt), h_target).item()
    for _ in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(cell(x, h_prev, dt=dt), h_target)
        loss.backward()
        opt.step()
    loss1 = torch.nn.functional.mse_loss(cell(x, h_prev, dt=dt), h_target).item()
    assert loss1 < loss0, f"loss did not decrease ({loss0} → {loss1})"
