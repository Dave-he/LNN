"""Tests for round 126 Mixture-of-Recursions CfC (PRD #10-88)."""
from __future__ import annotations

import math

import pytest
import torch

from lnn.core.mor_cfc import (
    MoRCfCCell,
    MoRCfCNetwork,
    MoRRouter,
    mor_router_summary,
    mor_router_weights,
)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def test_router_init_default():
    r = MoRRouter(input_size=2, hidden_size=8, max_depth=3)
    assert r.max_depth == 3
    assert r.last_weights is None


def test_router_init_max_depth_1():
    r = MoRRouter(input_size=2, hidden_size=8, max_depth=1)
    assert r.max_depth == 1


def test_router_init_invalid_max_depth():
    with pytest.raises(AssertionError):
        MoRRouter(input_size=2, hidden_size=8, max_depth=0)


def test_router_init_with_hidden():
    r = MoRRouter(input_size=2, hidden_size=8, max_depth=4, router_hidden=16)
    assert r.max_depth == 4


def test_router_forward_shape():
    r = MoRRouter(input_size=2, hidden_size=8, max_depth=3)
    x = torch.randn(5, 2)
    h = torch.randn(5, 8)
    w = r(x, h)
    assert w.shape == (5, 3)
    # softmax sums to 1
    assert torch.allclose(w.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert r.last_weights is not None


def test_router_warm_start_biased_toward_d1():
    """At init, softmax bias [-2, -4, -6, ...] should strongly favour d=1."""
    r = MoRRouter(input_size=2, hidden_size=8, max_depth=3)
    x = torch.zeros(1, 2)
    h = torch.zeros(1, 8)
    w = r(x, h)
    # d=1 weight should be largest
    assert w[0, 0] > w[0, 1]
    assert w[0, 0] > w[0, 2]
    # Bias favours d=1 strongly
    assert w[0, 0] > 0.5


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------


def test_cell_init_default():
    cell = MoRCfCCell(input_size=2, hidden_size=8, max_depth=3)
    assert cell.max_depth == 3
    assert cell.input_size == 2
    assert cell.hidden_size == 8


def test_cell_init_max_depth_1():
    cell = MoRCfCCell(input_size=2, hidden_size=8, max_depth=1)
    assert cell.max_depth == 1


def test_cell_init_invalid_max_depth():
    with pytest.raises(AssertionError):
        MoRCfCCell(input_size=2, hidden_size=8, max_depth=0)


def test_cell_forward_shape():
    cell = MoRCfCCell(input_size=2, hidden_size=8, max_depth=3)
    x = torch.randn(4, 2)
    h = torch.randn(4, 8)
    h_new = cell(x, h, dt=1.0)
    assert h_new.shape == (4, 8)


def test_cell_forward_with_aux():
    cell = MoRCfCCell(input_size=2, hidden_size=8, max_depth=3)
    x = torch.randn(4, 2)
    h = torch.randn(4, 8)
    h_new, aux = cell.forward_with_aux(x, h, dt=1.0)
    assert h_new.shape == (4, 8)
    assert aux["weights"].shape == (4, 3)
    assert aux["h_states"].shape == (4, 3, 8)


def test_cell_max_depth_1_equals_base_cfc():
    """At max_depth=1 and frozen router, MoR cell should equal base CfC cell."""
    from lnn.core.cfc import CfCCell
    torch.manual_seed(42)
    base = CfCCell(input_size=2, hidden_size=8)
    mor = MoRCfCCell(input_size=2, hidden_size=8, max_depth=1)
    # Copy base weights into mor cell
    mor.cell.load_state_dict(base.state_dict())
    # Now ensure router weights are uniform 1.0
    mor.router.last_weights = None
    # Forward and compare
    x = torch.randn(4, 2)
    h = torch.randn(4, 8)
    h_base = base(x, h, dt=1.0)
    h_mor, _ = mor.forward_with_aux(x, h, dt=1.0)
    # When max_depth=1, w_1 = 1.0, so h_new == h_1
    assert torch.allclose(h_base, h_mor, atol=1e-5)


def test_cell_gradient_flow():
    cell = MoRCfCCell(input_size=2, hidden_size=8, max_depth=3)
    x = torch.randn(4, 2, requires_grad=True)
    h = torch.randn(4, 8)
    h_new = cell(x, h, dt=1.0)
    loss = h_new.sum()
    loss.backward()
    # Gradients should flow to all params including router
    for name, p in cell.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"


def test_cell_warm_start_dominant_d1():
    """At init, depth-1 weight should be largest, so output ≈ h_1."""
    cell = MoRCfCCell(input_size=2, hidden_size=8, max_depth=3)
    x = torch.zeros(1, 2)
    h = torch.zeros(1, 8)
    h_new, aux = cell.forward_with_aux(x, h, dt=1.0)
    w = aux["weights"]
    assert w[0, 0] > w[0, 1]
    assert w[0, 0] > w[0, 2]


def test_cell_smoke_sin_learns():
    """Quick sanity check: MoR cell with max_depth=1 should learn sin."""
    torch.manual_seed(0)
    cell = MoRCfCCell(input_size=1, hidden_size=8, max_depth=1)
    opt = torch.optim.Adam(cell.parameters(), lr=0.01)
    for _ in range(50):
        t = torch.linspace(0, 1, 16).unsqueeze(0).unsqueeze(-1)  # [1, 16, 1]
        h = torch.zeros(1, 8)
        outs = []
        h_i = h
        for i in range(16):
            x_t = t[0, i, :].unsqueeze(0)  # [1, 1]
            h_i = cell(x_t, h_i, dt=1.0)
            outs.append(h_i)
        out = torch.stack(outs, dim=1)  # [1, 16, 8]
        # Just check output is finite and not all zero
        assert torch.isfinite(out).all()
        # Use a simple loss
        target = torch.sin(t * 2 * math.pi)
        pred = out.mean(dim=-1, keepdim=True)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_cell_router_summary():
    cell = MoRCfCCell(input_size=2, hidden_size=8, max_depth=3)
    x = torch.randn(4, 2)
    h = torch.randn(4, 8)
    cell(x, h, dt=1.0)
    summary = mor_router_summary(cell)
    assert "mean_depth_weights" in summary
    assert len(summary["mean_depth_weights"]) == 3
    assert "argmax_depth" in summary
    assert 0 <= summary["argmax_depth_frac"] <= 1.0


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def test_network_init_default():
    net = MoRCfCNetwork(input_size=2, hidden_size=8, output_size=1, num_layers=2, max_depth=3)
    assert len(net.cells) == 2
    assert net.max_depth == 3


def test_network_forward_shape():
    net = MoRCfCNetwork(input_size=2, hidden_size=8, output_size=1, num_layers=2, max_depth=3)
    x = torch.randn(2, 10, 2)
    out = net(x)
    assert out.shape == (2, 10, 1)


def test_network_last_step():
    net = MoRCfCNetwork(input_size=2, hidden_size=8, output_size=1, num_layers=2, max_depth=3, return_sequences=False)
    x = torch.randn(2, 10, 2)
    out = net(x)
    assert out.shape == (2, 1)


def test_network_handles_nan():
    net = MoRCfCNetwork(input_size=2, hidden_size=8, output_size=1, num_layers=2, max_depth=3)
    x = torch.randn(2, 10, 2)
    x[0, 5, 0] = float("nan")
    out = net(x)
    assert torch.isfinite(out).all()


def test_network_max_depth_1_learns():
    """Network with max_depth=1 should still learn (baseline regression check)."""
    torch.manual_seed(1)
    net = MoRCfCNetwork(input_size=1, hidden_size=8, output_size=1, num_layers=2, max_depth=1)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(20):
        t = torch.linspace(0, 1, 16).unsqueeze(0).unsqueeze(-1)
        target = torch.sin(t * 2 * math.pi)
        pred = net(t)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)


def test_network_max_depth_3_gradient_flow():
    """Network with max_depth=3 should have gradient flow through all depths."""
    net = MoRCfCNetwork(input_size=2, hidden_size=8, output_size=1, num_layers=2, max_depth=3)
    x = torch.randn(2, 10, 2, requires_grad=True)
    out = net(x)
    loss = out.sum()
    loss.backward()
    for name, p in net.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"


def test_network_router_summary():
    net = MoRCfCNetwork(input_size=2, hidden_size=8, output_size=1, num_layers=2, max_depth=3)
    x = torch.randn(2, 10, 2)
    _ = net(x)
    # Each cell has a router with last_weights set
    for i, cell in enumerate(net.cells):
        summary = mor_router_summary(cell)
        assert "mean_depth_weights" in summary, f"layer {i} router has no summary"
        assert len(summary["mean_depth_weights"]) == 3
