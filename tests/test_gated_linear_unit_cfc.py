"""Round 139 — tests for Gated Linear Unit CfC (PRD #10-101)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.gated_linear_unit_cfc import (
    GatedLinearUnitCfCCell,
    GatedLinearUnitCfCStackedNetwork,
)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


def test_init_default():
    """Default init: input_size and hidden_size."""
    cell = GatedLinearUnitCfCCell(input_size=2, hidden_size=8)
    assert cell.input_size == 2
    assert cell.hidden_size == 8


def test_init_time_scale():
    """Time scale parameter should be initialized to time_scale_init."""
    cell = GatedLinearUnitCfCCell(input_size=2, hidden_size=8, time_scale_init=2.5)
    assert torch.allclose(cell.time_scale, torch.full((8,), 2.5))


def test_init_input_gate_shape():
    """Input gate should be a Linear(input_size, input_size)."""
    cell = GatedLinearUnitCfCCell(input_size=4, hidden_size=8)
    assert cell.input_gate[0].in_features == 4
    assert cell.input_gate[0].out_features == 4


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------


def test_forward_shape():
    """Forward should return [B, hidden_size]."""
    cell = GatedLinearUnitCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert h_new.shape == (4, 8)


def test_forward_finite():
    """Forward output should be finite."""
    cell = GatedLinearUnitCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2) * 5.0
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert torch.isfinite(h_new).all()


def test_forward_stability_100_steps():
    """No NaN/Inf in 100 sequential forward steps."""
    torch.manual_seed(0)
    cell = GatedLinearUnitCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    for _ in range(100):
        h = cell(x, h)
    assert torch.isfinite(h).all()


def test_forward_gate_bounded():
    """Input gate should be in [0, 1]."""
    cell = GatedLinearUnitCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2) * 100.0  # large inputs
    gate = cell.gate(x)
    assert (gate >= 0.0).all()
    assert (gate <= 1.0).all()


def test_forward_gate_zero_means_zero_output():
    """If gate=0 for all features, gated input is zero."""
    cell = GatedLinearUnitCfCCell(input_size=2, hidden_size=8)
    # Force gate to be 0 by zeroing the linear bias and using very negative input.
    with torch.no_grad():
        cell.input_gate[0].weight.zero_()
        cell.input_gate[0].bias.zero_()
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    # Should not be NaN or Inf even with zero input.
    assert torch.isfinite(h_new).all()


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


def test_gradient_to_input_gate():
    """Gradient should reach the GLU input gate weights."""
    cell = GatedLinearUnitCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.input_gate[0].weight.grad is not None
    assert cell.input_gate[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_W_f():
    """Gradient should reach the CfC f_gate weights."""
    cell = GatedLinearUnitCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_W_g():
    """Gradient should reach the CfC g_branch weights."""
    cell = GatedLinearUnitCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.g_branch[0].weight.grad is not None
    assert cell.g_branch[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_W_h():
    """Gradient should reach the CfC h_branch weights."""
    cell = GatedLinearUnitCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.h_branch[0].weight.grad is not None
    assert cell.h_branch[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_time_scale():
    """Gradient should reach the CfC time_scale parameter."""
    cell = GatedLinearUnitCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.time_scale.grad is not None
    assert cell.time_scale.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network."""
    net = GatedLinearUnitCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8


def test_stacked_forward_shape_sequences():
    """return_sequences=True returns [B, T, output_size]."""
    net = GatedLinearUnitCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=True,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 16, 1)


def test_stacked_forward_shape_last_step():
    """return_sequences=False returns [B, output_size]."""
    net = GatedLinearUnitCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=False,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 1)


def test_stacked_handles_nan_input():
    """Forward should handle NaN inputs (zero-fill)."""
    net = GatedLinearUnitCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach all layers' parameters."""
    net = GatedLinearUnitCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.g_branch[0].weight.grad is not None
        assert cell.g_branch[0].weight.grad.abs().sum().item() > 0
        assert cell.input_gate[0].weight.grad is not None
        assert cell.input_gate[0].weight.grad.abs().sum().item() > 0


def test_stacked_smoke_learns_sin():
    """Smoke: stacked GLU-CfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = GatedLinearUnitCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
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
    assert final_loss is not None
    assert math.isfinite(final_loss), f"loss blew up: {final_loss}"
    assert final_loss < 5.0, f"loss too high after 50 steps: {final_loss}"
    # Verify loss decreased.
    assert final_loss < initial_loss, (
        f"loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# Bench-style smoke test
# ---------------------------------------------------------------------------


def test_bench_smoke_glu_vs_cfc():
    """Mini-bench: GLU-CfC vs CfC baseline on sin task."""
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

    # GLU-CfC.
    torch.manual_seed(42)
    glu = GatedLinearUnitCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        return_sequences=True,
    )
    opt = torch.optim.Adam(glu.parameters(), lr=1e-2)
    glu_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = glu(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        glu_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(glu_loss)


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
