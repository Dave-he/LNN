"""Round 147 — tests for Clockwork CfC (PRD #10-109)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.clockwork_cfc import ClockworkCfCCell, ClockworkCfCStackedNetwork


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------


def test_cell_init_default():
    """Default K=3 modules with equal split."""
    cell = ClockworkCfCCell(input_size=2, hidden_size=12, num_modules=3)
    assert cell.input_size == 2
    assert cell.hidden_size == 12
    assert cell.num_modules == 3
    assert cell.module_sizes == [4, 4, 4]
    assert cell.periods == [1, 2, 4]


def test_cell_init_k4():
    """K=4 modules with hidden=16 should be [4,4,4,4]."""
    cell = ClockworkCfCCell(input_size=2, hidden_size=16, num_modules=4)
    assert cell.module_sizes == [4, 4, 4, 4]
    assert cell.periods == [1, 2, 4, 8]


def test_cell_init_k2():
    """K=2 modules with hidden=10 should be [5, 5]."""
    cell = ClockworkCfCCell(input_size=2, hidden_size=10, num_modules=2)
    assert cell.module_sizes == [5, 5]
    assert cell.periods == [1, 2]


def test_cell_init_uneven_hidden():
    """K=3 with hidden=10 → [3, 3, 4] (last absorbs remainder)."""
    cell = ClockworkCfCCell(input_size=2, hidden_size=10, num_modules=3)
    assert cell.module_sizes == [3, 3, 4]


def test_cell_init_custom_module_sizes():
    """Custom module sizes."""
    cell = ClockworkCfCCell(
        input_size=2, hidden_size=10, num_modules=3, module_sizes=[2, 3, 5],
    )
    assert cell.module_sizes == [2, 3, 5]


def test_cell_init_module_sizes_mismatch():
    """module_sizes not summing to hidden should raise."""
    try:
        ClockworkCfCCell(input_size=2, hidden_size=10, num_modules=3, module_sizes=[2, 3, 4])
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_invalid_num_modules():
    """num_modules < 1 should raise."""
    try:
        ClockworkCfCCell(input_size=2, hidden_size=8, num_modules=0)
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_forward_shape():
    """Forward returns [B, T, hidden]."""
    cell = ClockworkCfCCell(input_size=2, hidden_size=12, num_modules=3)
    x = torch.randn(2, 16, 2)
    out = cell(x)
    assert out.shape == (2, 16, 12)


def test_cell_forward_finite():
    """Forward output is finite."""
    cell = ClockworkCfCCell(input_size=2, hidden_size=12, num_modules=3)
    x = torch.randn(2, 16, 2) * 5.0
    out = cell(x)
    assert torch.isfinite(out).all()


def test_cell_forward_handles_nan():
    """NaN input handled."""
    cell = ClockworkCfCCell(input_size=2, hidden_size=12, num_modules=3)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    out = cell(x)
    assert torch.isfinite(out).all()


def test_cell_module_0_updates_every_step():
    """Module 0 (period 1) should update at every step."""
    torch.manual_seed(0)
    cell = ClockworkCfCCell(input_size=2, hidden_size=4, num_modules=1)
    # Replace its f_gate with deterministic weights.
    with torch.no_grad():
        cell.cells[0].f_gate[0].weight.fill_(0.1)
        cell.cells[0].f_gate[0].bias.fill_(0.0)
    x = torch.randn(1, 4, 2) * 0.5
    out = cell(x)
    # Each timestep's h_0 should be different (proves module 0 updates every step).
    h0 = out[0, :, :]  # [T, hidden]
    # Diff between consecutive timesteps should be non-zero.
    diffs = (h0[1:] - h0[:-1]).abs().sum(dim=-1)
    assert (diffs > 0).all()


def test_cell_module_1_updates_every_2_steps():
    """Module 1 (period 2) should NOT update at t=1, t=3, etc."""
    torch.manual_seed(0)
    cell = ClockworkCfCCell(
        input_size=2, hidden_size=8, num_modules=2, module_sizes=[4, 4],
    )
    with torch.no_grad():
        for c in cell.cells:
            c.f_gate[0].weight.fill_(0.1)
            c.f_gate[0].bias.fill_(0.0)
    x = torch.randn(1, 8, 2) * 0.5
    out = cell(x)
    # h_1 is the second half (modules [4:8]).
    h1 = out[0, :, 4:8]  # [T, 4]
    # At t=0 → t=1: h_1 should be unchanged (carried forward).
    assert torch.allclose(h1[1], h1[0], atol=1e-6)
    # At t=1 → t=2: h_1 should change (updated).
    assert not torch.allclose(h1[2], h1[1], atol=1e-6)


def test_cell_gradient_flows_to_all_modules():
    """Gradient should reach all modules' cells."""
    cell = ClockworkCfCCell(input_size=2, hidden_size=12, num_modules=3)
    x = torch.randn(2, 16, 2)
    out = cell(x)
    out.sum().backward()
    for k, c in enumerate(cell.cells):
        assert c.f_gate[0].weight.grad is not None
        assert c.f_gate[0].weight.grad.abs().sum().item() > 0, f"Module {k} has zero grad"


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network."""
    net = ClockworkCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, num_modules=3,
    )
    assert net.num_layers == 2
    assert net.hidden_size == 12
    assert net.num_modules == 3


def test_stacked_init_k4():
    """K=4 modules per layer."""
    net = ClockworkCfCStackedNetwork(
        input_size=2, hidden_size=16, output_size=1, num_layers=2, num_modules=4,
    )
    for cell in net.cells:
        assert cell.periods == [1, 2, 4, 8]


def test_stacked_forward_shape():
    """return_sequences=True returns [B, T, output_size]."""
    net = ClockworkCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, num_modules=3,
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_finite_with_nan():
    """Forward handles NaN inputs."""
    net = ClockworkCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, num_modules=3,
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach all layers' cells."""
    net = ClockworkCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, num_modules=3,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for li, cell in enumerate(net.cells):
        for k, c in enumerate(cell.cells):
            assert c.f_gate[0].weight.grad is not None, f"Layer {li} module {k} no grad"
            assert c.f_gate[0].weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_learns_sin_k3():
    """Smoke: Clockwork CfC K=3 should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = ClockworkCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, num_modules=3,
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


def test_smoke_learns_structured_k4():
    """Smoke: Clockwork CfC K=4 should reduce loss on toy structured."""
    torch.manual_seed(0)
    net = ClockworkCfCStackedNetwork(
        input_size=2, hidden_size=16, output_size=1, num_layers=2, num_modules=4,
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


def test_bench_smoke_clockwork_vs_cfc():
    """Mini-bench: Clockwork CfC vs CfC baseline on sin task."""
    from lnn.core.cfc import CfCNetwork

    B, T, D, H = 4, 16, 2, 8
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
    x = torch.zeros(B, T, D)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    y = torch.sin(t.squeeze(-1)).unsqueeze(-1).expand(B, T, 1)

    # CfC baseline (H=12 to match clockwork hidden).
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

    # Clockwork CfC K=3.
    torch.manual_seed(42)
    cw = ClockworkCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2, num_modules=3,
        return_sequences=True,
    )
    opt = torch.optim.Adam(cw.parameters(), lr=1e-2)
    cw_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = cw(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        cw_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(cw_loss)


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
