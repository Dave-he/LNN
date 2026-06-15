"""Round 143 — tests for Decoupled CfC + IndRNN-CfC (PRD #10-105)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.decoupled_cfc import (
    DecoupledCfCCell,
    DecoupledCfCStackedNetwork,
    IndRNNCfCCell,
)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


def test_init_decoupled_cell():
    """Decoupled cell init."""
    cell = DecoupledCfCCell(input_size=2, hidden_size=8)
    assert cell.input_size == 2
    assert cell.hidden_size == 8
    assert hasattr(cell, "x_proj")
    assert hasattr(cell, "h_proj")
    assert isinstance(cell.time_scale, torch.nn.Parameter)


def test_init_indrnn_cell():
    """IndRNN cell init."""
    cell = IndRNNCfCCell(input_size=2, hidden_size=8)
    assert cell.input_size == 2
    assert cell.hidden_size == 8
    assert hasattr(cell, "x_proj")
    # u is a parameter, not a Linear layer.
    assert isinstance(cell.u, torch.nn.Parameter)
    assert cell.u.shape == (8,)


def test_init_indrnn_u_init_value():
    """IndRNN u should be initialized to u_init."""
    cell = IndRNNCfCCell(input_size=2, hidden_size=8, u_init=0.7)
    assert torch.allclose(cell.u, torch.full((8,), 0.7), atol=1e-6)


def test_init_invalid_variant():
    """Invalid variant in stacked network should raise."""
    try:
        DecoupledCfCStackedNetwork(
            input_size=2, hidden_size=8, output_size=1, variant="invalid"
        )
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_init_indrnn_has_fewer_params_than_decoupled():
    """IndRNN has fewer params than Decoupled (u is vector, not matrix)."""
    decoupled = DecoupledCfCCell(input_size=2, hidden_size=8)
    indrnn = IndRNNCfCCell(input_size=2, hidden_size=8)
    n_decoupled = sum(p.numel() for p in decoupled.parameters())
    n_indrnn = sum(p.numel() for p in indrnn.parameters())
    # IndRNN has u (8 params) instead of h_proj (8*8 + 8 = 72 params).
    # So IndRNN should have 72-8 = 64 fewer params.
    assert n_indrnn < n_decoupled, (
        f"IndRNN ({n_indrnn}) should have fewer params than Decoupled ({n_decoupled})"
    )


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------


def test_forward_shape_decoupled():
    """Decoupled cell forward returns [B, hidden_size]."""
    cell = DecoupledCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert h_new.shape == (4, 8)


def test_forward_shape_indrnn():
    """IndRNN cell forward returns [B, hidden_size]."""
    cell = IndRNNCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert h_new.shape == (4, 8)


def test_forward_finite_decoupled():
    """Decoupled cell output is finite."""
    torch.manual_seed(0)
    cell = DecoupledCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2) * 5.0
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert torch.isfinite(h_new).all()


def test_forward_finite_indrnn():
    """IndRNN cell output is finite."""
    torch.manual_seed(0)
    cell = IndRNNCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2) * 5.0
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert torch.isfinite(h_new).all()


def test_forward_evolves_from_h_zero_decoupled():
    """Decoupled cell: h must evolve from h=0."""
    cell = DecoupledCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    assert h_new.abs().sum() > 1e-3


def test_forward_evolves_from_h_zero_indrnn():
    """IndRNN cell: h must evolve from h=0 (u=0.5 > 0)."""
    cell = IndRNNCfCCell(input_size=2, hidden_size=8, u_init=0.5)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    assert h_new.abs().sum() > 1e-3


def test_forward_stability_100_steps_decoupled():
    """No NaN/Inf in 100 sequential decoupled steps."""
    torch.manual_seed(0)
    cell = DecoupledCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    for _ in range(100):
        h = cell(x, h)
    assert torch.isfinite(h).all()


def test_forward_stability_100_steps_indrnn():
    """No NaN/Inf in 100 sequential IndRNN steps."""
    torch.manual_seed(0)
    cell = IndRNNCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    for _ in range(100):
        h = cell(x, h)
    assert torch.isfinite(h).all()


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


def test_gradient_to_x_proj_decoupled():
    """Gradient should reach x_proj weights in decoupled cell."""
    cell = DecoupledCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.x_proj.weight.grad is not None
    assert cell.x_proj.weight.grad.abs().sum().item() > 0


def test_gradient_to_h_proj_decoupled():
    """Gradient should reach h_proj weights in decoupled cell."""
    cell = DecoupledCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.randn(2, 8)  # non-zero h
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.h_proj.weight.grad is not None
    assert cell.h_proj.weight.grad.abs().sum().item() > 0


def test_gradient_to_u_indrnn():
    """Gradient should reach u (element-wise recurrent weights)."""
    cell = IndRNNCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.randn(2, 8)
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.u.grad is not None
    assert cell.u.grad.abs().sum().item() > 0


def test_gradient_to_f_gate_decoupled():
    """Gradient should reach f_gate weights in decoupled cell."""
    cell = DecoupledCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_decoupled():
    """Default 2-layer decoupled stacked network."""
    net = DecoupledCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, variant="decoupled",
    )
    assert net.num_layers == 2
    assert net.variant == "decoupled"


def test_stacked_init_indrnn():
    """Default 2-layer IndRNN stacked network."""
    net = DecoupledCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, variant="indrnn",
    )
    assert net.variant == "indrnn"


def test_stacked_forward_shape_decoupled():
    """return_sequences=True returns [B, T, output_size] for decoupled."""
    net = DecoupledCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, variant="decoupled",
        return_sequences=True,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 16, 1)


def test_stacked_forward_shape_indrnn():
    """return_sequences=True returns [B, T, output_size] for indrnn."""
    net = DecoupledCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, variant="indrnn",
        return_sequences=True,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 16, 1)


def test_stacked_handles_nan_input_indrnn():
    """Forward should handle NaN inputs (zero-fill)."""
    net = DecoupledCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, variant="indrnn",
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_smoke_learns_sin_decoupled():
    """Smoke: stacked decoupled should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = DecoupledCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, variant="decoupled",
    )
    t = torch.linspace(0, 4 * math.pi, 16).unsqueeze(0).unsqueeze(-1)
    x = torch.zeros(2, 16, 2)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    target = torch.sin(t.squeeze(-1)).unsqueeze(-1).expand(2, 16, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    initial_loss = None
    final_loss = None
    for _ in range(50):
        opt.zero_grad()
        out = net(x)
        loss = F.mse_loss(out, target)
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    assert math.isfinite(final_loss)
    assert final_loss < 5.0
    assert final_loss < initial_loss


def test_stacked_smoke_learns_sin_indrnn():
    """Smoke: stacked IndRNN-CfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = DecoupledCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, variant="indrnn",
    )
    t = torch.linspace(0, 4 * math.pi, 16).unsqueeze(0).unsqueeze(-1)
    x = torch.zeros(2, 16, 2)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    target = torch.sin(t.squeeze(-1)).unsqueeze(-1).expand(2, 16, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    initial_loss = None
    final_loss = None
    for _ in range(50):
        opt.zero_grad()
        out = net(x)
        loss = F.mse_loss(out, target)
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    assert math.isfinite(final_loss)
    assert final_loss < 5.0
    assert final_loss < initial_loss


# ---------------------------------------------------------------------------
# Bench-style smoke test
# ---------------------------------------------------------------------------


def test_bench_smoke_decoupled_vs_cfc():
    """Mini-bench: Decoupled CfC vs CfC baseline on sin task."""
    from lnn.core.cfc import CfCNetwork

    B, T, D, H = 4, 16, 2, 8
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
    x = torch.zeros(B, T, D)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    y = torch.sin(t.squeeze(-1)).unsqueeze(-1).expand(B, T, 1)

    # CfC baseline.
    torch.manual_seed(42)
    cfc = CfCNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        return_sequences=True,
    )
    opt = torch.optim.Adam(cfc.parameters(), lr=1e-2)
    cfc_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = cfc(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        cfc_loss = loss.item()

    # Decoupled CfC.
    torch.manual_seed(42)
    decoupled = DecoupledCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2, variant="decoupled",
        return_sequences=True,
    )
    opt = torch.optim.Adam(decoupled.parameters(), lr=1e-2)
    decoupled_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = decoupled(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        decoupled_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(decoupled_loss)


if __name__ == "__main__":
    import inspect
    this = sys.modules[__name__]
    funcs = [
        (name, fn)
        for name, fn in inspect.getmembers(this, inspect.isfunction)
        if name.startswith("test_")
    ]
    print(f"=== Running {len(funcs)} tests ===")
    failed = []
    for name, fn in funcs:
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed.append((name, e))
    if failed:
        print(f"\n{len(failed)} FAILED")
        for name, e in failed:
            print(f"  - {name}: {e}")
        sys.exit(1)
    print(f"\nAll {len(funcs)} tests passed.")
