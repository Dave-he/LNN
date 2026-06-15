"""Round 151 — tests for Multi-Scale Dilated Conv CfC (MSDC-CfC) (PRD #10-113)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.msdc_cfc import (
    MultiScaleDilatedConvCfCCell,
    MultiScaleDilatedConvCfCStackedNetwork,
)


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------


def test_cell_init_default():
    """Default sum combine, dilations 1/2/4."""
    cell = MultiScaleDilatedConvCfCCell(input_size=2, hidden_size=8, combine="sum")
    assert cell.input_size == 2
    assert cell.hidden_size == 8
    assert cell.dilations == [1, 2, 4]
    assert cell.combine == "sum"
    assert cell.context_dim == 2
    assert len(cell.convs) == 3


def test_cell_init_concat_combine():
    """Concat combine init."""
    cell = MultiScaleDilatedConvCfCCell(input_size=2, hidden_size=8, combine="concat")
    assert cell.combine == "concat"
    assert cell.context_dim == 6  # 3 dilations * 2 input_size


def test_cell_init_custom_dilations():
    """Custom dilations list."""
    cell = MultiScaleDilatedConvCfCCell(
        input_size=2, hidden_size=8, dilations=[1, 3], combine="sum"
    )
    assert cell.dilations == [1, 3]
    assert len(cell.convs) == 2
    assert cell.context_dim == 2


def test_cell_init_invalid_combine():
    """Invalid combine should raise."""
    try:
        MultiScaleDilatedConvCfCCell(input_size=2, hidden_size=8, combine="invalid")
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_forward_shape_sum():
    """Sum combine returns [B, T, hidden_size]."""
    cell = MultiScaleDilatedConvCfCCell(input_size=2, hidden_size=8, combine="sum")
    x = torch.randn(2, 16, 2)
    out = cell(x)
    assert out.shape == (2, 16, 8)


def test_cell_forward_shape_concat():
    """Concat combine returns [B, T, hidden_size]."""
    cell = MultiScaleDilatedConvCfCCell(input_size=2, hidden_size=8, combine="concat")
    x = torch.randn(2, 16, 2)
    out = cell(x)
    assert out.shape == (2, 16, 8)


def test_cell_forward_finite():
    """Forward output is finite."""
    cell = MultiScaleDilatedConvCfCCell(input_size=2, hidden_size=8, combine="sum")
    x = torch.randn(2, 16, 2) * 5.0
    out = cell(x)
    assert torch.isfinite(out).all()


def test_cell_forward_handles_nan():
    """NaN input handled."""
    cell = MultiScaleDilatedConvCfCCell(input_size=2, hidden_size=8, combine="sum")
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    out = cell(x)
    assert torch.isfinite(out).all()


def test_cell_gradient_flows_to_convs():
    """Gradient should reach all conv weights."""
    cell = MultiScaleDilatedConvCfCCell(input_size=2, hidden_size=8, combine="sum")
    x = torch.randn(2, 16, 2)
    out = cell(x)
    out.sum().backward()
    for conv in cell.convs:
        assert conv.weight.grad is not None
        assert conv.weight.grad.abs().sum().item() > 0


def test_cell_gradient_flows_to_cfc():
    """Gradient should reach CfC weights."""
    cell = MultiScaleDilatedConvCfCCell(input_size=2, hidden_size=8, combine="sum")
    x = torch.randn(2, 16, 2)
    out = cell(x)
    out.sum().backward()
    assert cell.cfc.f_gate[0].weight.grad is not None
    assert cell.cfc.f_gate[0].weight.grad.abs().sum().item() > 0


def test_cell_causality():
    """Causal: changing future input should not affect past output."""
    cell = MultiScaleDilatedConvCfCCell(input_size=2, hidden_size=8, combine="sum")
    cell.eval()
    with torch.no_grad():
        x1 = torch.randn(1, 16, 2)
        x2 = x1.clone()
        x2[0, 10:, :] = torch.randn(6, 2)  # change future
        out1 = cell(x1)
        out2 = cell(x2)
        # Output at t < 10 should be identical (causal).
        for t in range(10):
            assert torch.allclose(out1[0, t, :], out2[0, t, :], atol=1e-5), f"Non-causal at t={t}"


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network."""
    net = MultiScaleDilatedConvCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, combine="sum",
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8
    assert net.combine == "sum"
    assert net.dilations == [1, 2, 4]


def test_stacked_forward_shape():
    """Forward returns [B, T, output_size]."""
    net = MultiScaleDilatedConvCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, combine="sum",
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = MultiScaleDilatedConvCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, combine="sum",
        return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_stacked_forward_finite_with_nan():
    """Forward handles NaN inputs."""
    net = MultiScaleDilatedConvCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, combine="sum",
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach all layers' conv and CfC weights."""
    net = MultiScaleDilatedConvCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, combine="sum",
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for li, cell in enumerate(net.cells):
        for conv in cell.convs:
            assert conv.weight.grad is not None
            assert conv.weight.grad.abs().sum().item() > 0
        assert cell.cfc.f_gate[0].weight.grad is not None
        assert cell.cfc.f_gate[0].weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_learns_sin():
    """Smoke: MSDC-CfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = MultiScaleDilatedConvCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, combine="sum",
    )
    t = torch.linspace(0, 4 * math.pi, 16).unsqueeze(0).unsqueeze(-1).expand(2, 16, 1)
    x = torch.zeros(2, 16, 2)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    target = torch.sin(t.squeeze(-1)).unsqueeze(-1)
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


def test_smoke_learns_structured():
    """Smoke: MSDC-CfC should reduce loss on structured task."""
    torch.manual_seed(0)
    net = MultiScaleDilatedConvCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, combine="sum",
    )
    t = torch.linspace(0, 4 * math.pi, 16).unsqueeze(0).unsqueeze(-1).expand(2, 16, 1)
    x = torch.zeros(2, 16, 2)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    target = torch.sin(2 * t.squeeze(-1)).unsqueeze(-1)
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


def test_bench_smoke_msdc_vs_cfc():
    """Mini-bench: MSDC-CfC vs CfC baseline on sin task."""
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
        input_size=D, hidden_size=H, output_size=1, num_layers=2, return_sequences=True,
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

    # MSDC-CfC.
    torch.manual_seed(42)
    msdc = MultiScaleDilatedConvCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2, combine="sum",
        return_sequences=True,
    )
    opt = torch.optim.Adam(msdc.parameters(), lr=1e-2)
    msdc_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = msdc(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        msdc_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(msdc_loss)


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
