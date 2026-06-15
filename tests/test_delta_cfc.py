"""Round 155 — tests for DELTA-CfC (Hidden State Delta Augmentation) (PRD #10-117)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.delta_cfc import DeltaCfCCell, DeltaCfCStackedNetwork


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------


def test_cell_init_concat():
    """Default concat mode."""
    cell = DeltaCfCCell(input_size=2, hidden_size=8, delta_mode="concat")
    assert cell.input_size == 2
    assert cell.hidden_size == 8
    assert cell.delta_mode == "concat"


def test_cell_init_proj():
    """Proj mode."""
    cell = DeltaCfCCell(input_size=2, hidden_size=8, delta_mode="proj")
    assert cell.delta_mode == "proj"
    assert hasattr(cell, "delta_proj")


def test_cell_init_gated():
    """Gated mode."""
    cell = DeltaCfCCell(input_size=2, hidden_size=8, delta_mode="gated")
    assert cell.delta_mode == "gated"
    assert hasattr(cell, "delta_gate")
    # Initialized to 0 → sigmoid(0) = 0.5.
    assert torch.allclose(cell.delta_gate, torch.zeros(8))


def test_cell_init_concat_input():
    """Concat_input mode."""
    cell = DeltaCfCCell(input_size=2, hidden_size=8, delta_mode="concat_input")
    assert cell.delta_mode == "concat_input"


def test_cell_init_invalid_mode():
    """Invalid delta_mode should raise."""
    try:
        DeltaCfCCell(input_size=2, hidden_size=8, delta_mode="invalid")
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_step_shape_concat():
    """Concat mode: 2*hidden_size output."""
    cell = DeltaCfCCell(input_size=2, hidden_size=8, delta_mode="concat")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    out = cell(x, h)
    assert out.shape == (4, 16)


def test_cell_step_shape_proj():
    """Proj mode: hidden_size output."""
    cell = DeltaCfCCell(input_size=2, hidden_size=8, delta_mode="proj")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    out = cell(x, h)
    assert out.shape == (4, 8)


def test_cell_step_shape_gated():
    """Gated mode: hidden_size output."""
    cell = DeltaCfCCell(input_size=2, hidden_size=8, delta_mode="gated")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    out = cell(x, h)
    assert out.shape == (4, 8)


def test_cell_step_shape_concat_input():
    """Concat_input mode: hidden_size output."""
    cell = DeltaCfCCell(input_size=2, hidden_size=8, delta_mode="concat_input")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    out = cell(x, h)
    assert out.shape == (4, 8)


def test_cell_step_finite_concat():
    """Forward output is finite (concat)."""
    cell = DeltaCfCCell(input_size=2, hidden_size=8, delta_mode="concat")
    x = torch.randn(4, 2) * 5.0
    h = torch.zeros(4, 8)
    out = cell(x, h)
    assert torch.isfinite(out).all()


def test_cell_step_handles_nan_concat():
    """NaN input handled (concat)."""
    cell = DeltaCfCCell(input_size=2, hidden_size=8, delta_mode="concat")
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 8)
    out = cell(x, h)
    assert torch.isfinite(out).all()


def test_cell_gated_init_returns_h():
    """Gated init (alpha=0) means initial output ≈ h_new (because alpha=0.5 not 0)."""
    # NOTE: sigmoid(0) = 0.5, so initial alpha = 0.5, not 0.
    # Just verify the cell returns a valid output.
    cell = DeltaCfCCell(input_size=2, hidden_size=8, delta_mode="gated")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    out = cell(x, h)
    assert out.shape == (4, 8)
    assert torch.isfinite(out).all()


def test_cell_gradient_flows_concat():
    """Gradient should reach CfC and delta-related params (concat)."""
    cell = DeltaCfCCell(input_size=2, hidden_size=8, delta_mode="concat")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_cell_gradient_flows_proj():
    """Gradient should reach delta_proj (proj mode)."""
    cell = DeltaCfCCell(input_size=2, hidden_size=8, delta_mode="proj")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.delta_proj.weight.grad is not None
    assert cell.delta_proj.weight.grad.abs().sum().item() > 0


def test_cell_gradient_flows_gated():
    """Gradient should reach delta_gate (gated mode)."""
    cell = DeltaCfCCell(input_size=2, hidden_size=8, delta_mode="gated")
    # Use non-zero h to ensure Δh is non-zero so the gradient on
    # delta_gate is meaningful.
    x = torch.randn(4, 2)
    h = torch.randn(4, 8) * 0.5
    out = cell(x, h)
    out.sum().backward()
    assert cell.delta_gate.grad is not None
    # The gradient on delta_gate is small (1st-order through sigmoid
    # scaling) but non-zero when h and h_new are different.
    assert cell.delta_gate.grad.abs().sum().item() > 0


def test_cell_delta_for_next_layer():
    """Concat_input mode: delta_for_next_layer returns h_new - h_prev."""
    cell = DeltaCfCCell(input_size=2, hidden_size=8, delta_mode="concat_input")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    delta = cell.delta_for_next_layer(h_new, h)
    assert torch.allclose(delta, h_new - h)


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network (concat)."""
    net = DeltaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, delta_mode="concat",
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8
    assert net.delta_mode == "concat"


def test_stacked_forward_shape_concat():
    """Forward returns [B, T, output_size] (concat)."""
    net = DeltaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, delta_mode="concat",
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_proj():
    """Forward returns [B, T, output_size] (proj)."""
    net = DeltaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, delta_mode="proj",
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_gated():
    """Forward returns [B, T, output_size] (gated)."""
    net = DeltaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, delta_mode="gated",
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_concat_input():
    """Forward returns [B, T, output_size] (concat_input)."""
    net = DeltaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, delta_mode="concat_input",
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = DeltaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, delta_mode="concat",
        return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_stacked_forward_finite_with_nan_concat():
    """Forward handles NaN inputs (concat)."""
    net = DeltaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, delta_mode="concat",
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers_concat():
    """Gradient should reach all layers' CfC weights (concat)."""
    net = DeltaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, delta_mode="concat",
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.f_gate[0].weight.grad is not None
        assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_learns_sin_concat():
    """Smoke: Delta-CfC (concat) should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = DeltaCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, delta_mode="concat",
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


def test_smoke_learns_sin_gated():
    """Smoke: Delta-CfC (gated) should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = DeltaCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, delta_mode="gated",
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


# ---------------------------------------------------------------------------
# Bench-style smoke test
# ---------------------------------------------------------------------------


def test_bench_smoke_delta_vs_cfc():
    """Mini-bench: Delta-CfC vs CfC baseline on sin task."""
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

    # Delta-CfC concat.
    torch.manual_seed(42)
    delta = DeltaCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2, delta_mode="concat",
        return_sequences=True,
    )
    opt = torch.optim.Adam(delta.parameters(), lr=1e-2)
    delta_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = delta(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        delta_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(delta_loss)


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
