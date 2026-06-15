"""Round 154 — tests for MONO-CfC (Monotonic Activation CfC) (PRD #10-116)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.mono_cfc import MonoCfCCell, MonoCfCStackedNetwork


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------


def test_cell_init_g_only():
    """Default g_only mode."""
    cell = MonoCfCCell(input_size=2, hidden_size=8, mono_mode="g_only")
    assert cell.input_size == 2
    assert cell.hidden_size == 8
    assert cell.mono_mode == "g_only"
    # g_branch uses Softplus.
    assert isinstance(cell.g_branch[1], torch.nn.Softplus)
    # h_branch uses Tanh.
    assert isinstance(cell.h_branch[1], torch.nn.Tanh)


def test_cell_init_h_only():
    """h_only mode: g=Tanh, h=Softplus."""
    cell = MonoCfCCell(input_size=2, hidden_size=8, mono_mode="h_only")
    assert isinstance(cell.g_branch[1], torch.nn.Tanh)
    assert isinstance(cell.h_branch[1], torch.nn.Softplus)


def test_cell_init_both():
    """both mode: g=Softplus, h=Softplus."""
    cell = MonoCfCCell(input_size=2, hidden_size=8, mono_mode="both")
    assert isinstance(cell.g_branch[1], torch.nn.Softplus)
    assert isinstance(cell.h_branch[1], torch.nn.Softplus)


def test_cell_init_sigmoid():
    """sigmoid mode: g=Sigmoid, h=Sigmoid."""
    cell = MonoCfCCell(input_size=2, hidden_size=8, mono_mode="sigmoid")
    assert isinstance(cell.g_branch[1], torch.nn.Sigmoid)
    assert isinstance(cell.h_branch[1], torch.nn.Sigmoid)


def test_cell_init_invalid_mode():
    """Invalid mono_mode should raise."""
    try:
        MonoCfCCell(input_size=2, hidden_size=8, mono_mode="invalid")
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_step_shape_g_only():
    """One-step forward shape (g_only)."""
    cell = MonoCfCCell(input_size=2, hidden_size=8, mono_mode="g_only")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    out = cell(x, h)
    assert out.shape == (4, 8)


def test_cell_step_shape_h_only():
    """One-step forward shape (h_only)."""
    cell = MonoCfCCell(input_size=2, hidden_size=8, mono_mode="h_only")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    out = cell(x, h)
    assert out.shape == (4, 8)


def test_cell_step_shape_both():
    """One-step forward shape (both)."""
    cell = MonoCfCCell(input_size=2, hidden_size=8, mono_mode="both")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    out = cell(x, h)
    assert out.shape == (4, 8)


def test_cell_step_shape_sigmoid():
    """One-step forward shape (sigmoid)."""
    cell = MonoCfCCell(input_size=2, hidden_size=8, mono_mode="sigmoid")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    out = cell(x, h)
    assert out.shape == (4, 8)


def test_cell_step_finite_g_only():
    """Forward output is finite (g_only)."""
    cell = MonoCfCCell(input_size=2, hidden_size=8, mono_mode="g_only")
    x = torch.randn(4, 2) * 5.0
    h = torch.zeros(4, 8)
    out = cell(x, h)
    assert torch.isfinite(out).all()


def test_cell_step_handles_nan_g_only():
    """NaN input handled (g_only)."""
    cell = MonoCfCCell(input_size=2, hidden_size=8, mono_mode="g_only")
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 8)
    out = cell(x, h)
    assert torch.isfinite(out).all()


def test_cell_softplus_positive_g_only():
    """Softplus in g_branch means g is non-negative."""
    cell = MonoCfCCell(input_size=2, hidden_size=8, mono_mode="g_only")
    cell.eval()
    with torch.no_grad():
        x = torch.randn(4, 2)
        h = torch.zeros(4, 8)
        z = torch.cat([x, h], dim=-1)
        g = cell.g_branch(z)
        assert (g >= 0).all()


def test_cell_tanh_bounded_h_only():
    """Tanh in h_branch means h_branch in [-1, 1]."""
    cell = MonoCfCCell(input_size=2, hidden_size=8, mono_mode="h_only")
    cell.eval()
    with torch.no_grad():
        x = torch.randn(4, 2)
        h = torch.zeros(4, 8)
        z = torch.cat([x, h], dim=-1)
        h_branch = cell.h_branch(z)
        assert (h_branch >= -1.0).all()
        assert (h_branch <= 1.0).all()


def test_cell_gradient_flows_g_only():
    """Gradient should reach g_branch and h_branch (g_only)."""
    cell = MonoCfCCell(input_size=2, hidden_size=8, mono_mode="g_only")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.g_branch[0].weight.grad is not None
    assert cell.g_branch[0].weight.grad.abs().sum().item() > 0
    assert cell.h_branch[0].weight.grad is not None
    assert cell.h_branch[0].weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network (g_only)."""
    net = MonoCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, mono_mode="g_only",
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8
    assert net.mono_mode == "g_only"


def test_stacked_forward_shape_g_only():
    """Forward returns [B, T, output_size] (g_only)."""
    net = MonoCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, mono_mode="g_only",
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_h_only():
    """Forward returns [B, T, output_size] (h_only)."""
    net = MonoCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, mono_mode="h_only",
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_both():
    """Forward returns [B, T, output_size] (both)."""
    net = MonoCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, mono_mode="both",
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_sigmoid():
    """Forward returns [B, T, output_size] (sigmoid)."""
    net = MonoCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, mono_mode="sigmoid",
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = MonoCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, mono_mode="g_only",
        return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_stacked_forward_finite_with_nan():
    """Forward handles NaN inputs (g_only)."""
    net = MonoCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, mono_mode="g_only",
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers_g_only():
    """Gradient should reach all layers' g_branch and h_branch (g_only)."""
    net = MonoCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, mono_mode="g_only",
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.g_branch[0].weight.grad is not None
        assert cell.g_branch[0].weight.grad.abs().sum().item() > 0
        assert cell.h_branch[0].weight.grad is not None
        assert cell.h_branch[0].weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_learns_sin_g_only():
    """Smoke: Mono-CfC (g_only) should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = MonoCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, mono_mode="g_only",
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


def test_smoke_learns_sin_both():
    """Smoke: Mono-CfC (both) should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = MonoCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, mono_mode="both",
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


def test_bench_smoke_mono_vs_cfc():
    """Mini-bench: Mono-CfC vs CfC baseline on sin task."""
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

    # Mono-CfC g_only.
    torch.manual_seed(42)
    mono = MonoCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2, mono_mode="g_only",
        return_sequences=True,
    )
    opt = torch.optim.Adam(mono.parameters(), lr=1e-2)
    mono_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = mono(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        mono_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(mono_loss)


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
