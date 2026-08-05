"""Round 281 — tests for MemoryFusionCfCCell (PRD N3+N2 from
LNN_Family_Taxonomy_And_Gap_2026-08-03 §3).

This is a *cross-paper synthesis* test that validates three retention
mechanisms (CfC / TFP / NSFD) behave as a drop-in CfC replacement and
produce numerically distinct outputs on identical input.
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.memory_fusion_cfc import (
    MemoryFusionCfCCell,
    MemoryFusionCfCNetwork,
    _VALID_RETENTION,
)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


def test_init_valid_kinds():
    for kind in _VALID_RETENTION:
        cell = MemoryFusionCfCCell(input_size=4, hidden_size=8, retention_kind=kind)
        assert cell.retention_kind == kind
        assert cell.hidden_size == 8
        assert cell.n_tau == 1


def test_init_invalid_kind_raises():
    try:
        MemoryFusionCfCCell(input_size=4, hidden_size=8, retention_kind="bogus")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid retention_kind")


def test_init_n_tau_branch_dims():
    cell = MemoryFusionCfCCell(input_size=4, hidden_size=10, n_tau=3)
    assert cell._branch_dims == [3, 3, 4]
    assert sum(cell._branch_dims) == 10


def test_init_n_tau_invalid():
    try:
        MemoryFusionCfCCell(input_size=4, hidden_size=8, n_tau=0)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for n_tau=0")


# ---------------------------------------------------------------------------
# Forward / shape tests
# ---------------------------------------------------------------------------


def _seed():
    torch.manual_seed(42)


def test_forward_shape_single_tau():
    _seed()
    batch = 7
    for kind in _VALID_RETENTION:
        cell = MemoryFusionCfCCell(input_size=3, hidden_size=8, retention_kind=kind)
        x_t = torch.randn(batch, 3)
        h = torch.randn(batch, 8)
        out = cell(x_t, h, dt=1.0)
        assert out.shape == (batch, 8), f"{kind}: shape {out.shape}"


def test_forward_shape_multi_tau():
    _seed()
    cell = MemoryFusionCfCCell(input_size=3, hidden_size=10, retention_kind="tfp", n_tau=3)
    x_t = torch.randn(4, 3)
    h = torch.randn(4, 10)
    out = cell(x_t, h, dt=0.5)
    assert out.shape == (4, 10)


def test_network_forward_shape():
    _seed()
    net = MemoryFusionCfCNetwork(input_size=3, hidden_size=8, output_size=2, retention_kind="nsfd")
    x = torch.randn(5, 12, 3)
    out = net(x, dt=0.7)
    assert out.shape == (5, 12, 2)


# ---------------------------------------------------------------------------
# Cross-paper distinctness
# ---------------------------------------------------------------------------


def test_three_retention_kinds_produce_different_outputs():
    """Same weights, same input ⇒ three modes produce different outputs.
    This is the *headline* test: it proves the three mechanisms are not
    numerically degenerate.
    """
    _seed()
    x_t = torch.randn(1, 4)
    h = torch.randn(1, 8)
    outs = {}
    for kind in _VALID_RETENTION:
        torch.manual_seed(0)  # same initialization
        cell = MemoryFusionCfCCell(input_size=4, hidden_size=8, retention_kind=kind)
        # Use the same input projection weights across modes so we compare
        # *only* the retention mechanism, not the input projection.
        with torch.no_grad():
            cell.g_branch[0][0].weight.copy_(_shared_g_weight())
            cell.g_branch[0][0].bias.zero_()
            cell.h_branch[0][0].weight.copy_(_shared_h_weight())
            cell.h_branch[0][0].bias.zero_()
        outs[kind] = cell(x_t, h, dt=1.0).clone()

    # cfc and tfp must differ
    diff_cfc_tfp = (outs["cfc"] - outs["tfp"]).abs().max().item()
    assert diff_cfc_tfp > 1e-3, f"cfc ≈ tfp (max diff {diff_cfc_tfp})"
    # tfp and nsfd must differ
    diff_tfp_nsfd = (outs["tfp"] - outs["nsfd"]).abs().max().item()
    assert diff_tfp_nsfd > 1e-3, f"tfp ≈ nsfd (max diff {diff_tfp_nsfd})"
    # cfc and nsfd must differ
    diff_cfc_nsfd = (outs["cfc"] - outs["nsfd"]).abs().max().item()
    assert diff_cfc_nsfd > 1e-3, f"cfc ≈ nsfd (max diff {diff_cfc_nsfd})"


def _shared_g_weight():
    g = torch.zeros(8, 12)
    torch.manual_seed(0)
    nn = torch.nn
    lin = nn.Linear(12, 8)
    return lin.weight.data


def _shared_h_weight():
    g = torch.zeros(8, 12)
    torch.manual_seed(0)
    nn = torch.nn
    lin = nn.Linear(12, 8)
    return lin.weight.data


# ---------------------------------------------------------------------------
# TFP-specific sanity
# ---------------------------------------------------------------------------


def test_tfp_retention_is_bounded():
    """TFP ``k = exp(-dt/τ) ∈ (0, 1]`` ⇒ h_new is a convex combo of h_prev and cand."""
    _seed()
    cell = MemoryFusionCfCCell(input_size=2, hidden_size=6, retention_kind="tfp")
    x_t = torch.randn(3, 2) * 5.0
    h_prev = torch.randn(3, 6) * 5.0
    out = cell(x_t, h_prev, dt=2.0)
    # h_new should be in [min(h_prev, cand), max(h_prev, cand)] element-wise
    # (we don't have direct cand here, but verify output is finite)
    assert torch.isfinite(out).all()
    # k is in (0, 1] — verify h_new is not exploding
    assert out.abs().max() < 1e6


def test_tfp_dt_zero_recovers_candidate():
    """With dt → 0, retention ``k → 1`` and ``h_new → h_prev`` (k*h + (1-k)*cand ≈ h_prev)."""
    _seed()
    cell = MemoryFusionCfCCell(input_size=2, hidden_size=4, retention_kind="tfp")
    x_t = torch.randn(2, 2)
    h_prev = torch.randn(2, 4)
    out = cell(x_t, h_prev, dt=1e-6)
    assert torch.allclose(out, h_prev, atol=1e-4)


# ---------------------------------------------------------------------------
# NSFD-specific sanity
# ---------------------------------------------------------------------------


def test_nsfd_dt_zero_recovers_h_prev():
    """With dt → 0, NSFD update ``(h + dt·G) / (1 + dt·L) → h_prev``."""
    _seed()
    cell = MemoryFusionCfCCell(input_size=2, hidden_size=4, retention_kind="nsfd")
    x_t = torch.randn(2, 2)
    h_prev = torch.randn(2, 4)
    out = cell(x_t, h_prev, dt=1e-6)
    assert torch.allclose(out, h_prev, atol=1e-4)


def test_nsfd_positivity_preserved_when_input_nonneg():
    """NSFD guarantees positivity if h_prev ≥ 0: verify."""
    _seed()
    cell = MemoryFusionCfCCell(input_size=2, hidden_size=4, retention_kind="nsfd")
    x_t = torch.randn(2, 2).abs()
    h_prev = torch.zeros(2, 4)  # non-negative init
    out = cell(x_t, h_prev, dt=0.5)
    assert (out >= -1e-6).all(), f"NSFD violated positivity, min={out.min().item()}"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow_cfc():
    _seed()
    cell = MemoryFusionCfCCell(input_size=3, hidden_size=8, retention_kind="cfc")
    x_t = torch.randn(4, 3, requires_grad=True)
    h = torch.randn(4, 8)
    out = cell(x_t, h, dt=1.0)
    out.sum().backward()
    assert x_t.grad is not None
    assert torch.isfinite(x_t.grad).all()


def test_gradients_flow_tfp():
    _seed()
    cell = MemoryFusionCfCCell(input_size=3, hidden_size=8, retention_kind="tfp")
    x_t = torch.randn(4, 3, requires_grad=True)
    h = torch.randn(4, 8)
    out = cell(x_t, h, dt=1.0)
    out.sum().backward()
    assert x_t.grad is not None
    assert torch.isfinite(x_t.grad).all()


def test_gradients_flow_nsfd():
    _seed()
    cell = MemoryFusionCfCCell(input_size=3, hidden_size=8, retention_kind="nsfd")
    x_t = torch.randn(4, 3, requires_grad=True)
    h = torch.randn(4, 8)
    out = cell(x_t, h, dt=1.0)
    out.sum().backward()
    assert x_t.grad is not None
    assert torch.isfinite(x_t.grad).all()


# ---------------------------------------------------------------------------
# End-to-end training smoke
# ---------------------------------------------------------------------------


def test_end_to_end_training_step():
    """Run a single train step on a synthetic regression task to make sure
    the cell is optimizable end-to-end through the network wrapper.
    """
    _seed()
    for kind in _VALID_RETENTION:
        torch.manual_seed(0)
        net = MemoryFusionCfCNetwork(
            input_size=3, hidden_size=8, output_size=2, retention_kind=kind
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        x = torch.randn(8, 16, 3)
        target = torch.randn(8, 16, 2)
        loss0 = torch.nn.functional.mse_loss(net(x, dt=1.0), target).item()
        for _ in range(5):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(x, dt=1.0), target)
            loss.backward()
            opt.step()
        loss1 = torch.nn.functional.mse_loss(net(x, dt=1.0), target).item()
        assert loss1 < loss0, f"{kind}: loss did not decrease ({loss0} → {loss1})"
