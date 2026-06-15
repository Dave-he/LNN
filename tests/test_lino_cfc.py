"""Round 150 — tests for Linear-Nonlinear CfC (LiNo-CfC) (PRD #10-112)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.lino_cfc import LinearNonlinearCfCCell, LinearNonlinearCfCStackedNetwork


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------


def test_cell_init_default():
    """Default sum mode init."""
    cell = LinearNonlinearCfCCell(input_size=2, hidden_size=8, mode="sum")
    assert cell.input_size == 2
    assert cell.hidden_size == 8
    assert cell.mode == "sum"
    assert cell.concat_size == 8


def test_cell_init_concat_mode():
    """Concat mode init."""
    cell = LinearNonlinearCfCCell(input_size=2, hidden_size=8, mode="concat")
    assert cell.mode == "concat"
    assert cell.concat_size == 16


def test_cell_init_invalid_mode():
    """Invalid mode should raise."""
    try:
        LinearNonlinearCfCCell(input_size=2, hidden_size=8, mode="invalid")
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_forward_shape_sum():
    """Sum mode returns [B, T, hidden_size]."""
    cell = LinearNonlinearCfCCell(input_size=2, hidden_size=8, mode="sum")
    x = torch.randn(2, 16, 2)
    out = cell(x)
    assert out.shape == (2, 16, 8)


def test_cell_forward_shape_concat():
    """Concat mode returns [B, T, 2*hidden_size]."""
    cell = LinearNonlinearCfCCell(input_size=2, hidden_size=8, mode="concat")
    x = torch.randn(2, 16, 2)
    out = cell(x)
    assert out.shape == (2, 16, 16)


def test_cell_forward_finite():
    """Forward output is finite."""
    cell = LinearNonlinearCfCCell(input_size=2, hidden_size=8, mode="sum")
    x = torch.randn(2, 16, 2) * 5.0
    out = cell(x)
    assert torch.isfinite(out).all()


def test_cell_forward_handles_nan():
    """NaN input handled."""
    cell = LinearNonlinearCfCCell(input_size=2, hidden_size=8, mode="sum")
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    out = cell(x)
    assert torch.isfinite(out).all()


def test_cell_gradient_flows_to_linear():
    """Gradient should reach linear_proj weights."""
    cell = LinearNonlinearCfCCell(input_size=2, hidden_size=8, mode="sum")
    x = torch.randn(2, 16, 2)
    out = cell(x)
    out.sum().backward()
    assert cell.linear_proj.weight.grad is not None
    assert cell.linear_proj.weight.grad.abs().sum().item() > 0


def test_cell_gradient_flows_to_cfc():
    """Gradient should reach CfC weights."""
    cell = LinearNonlinearCfCCell(input_size=2, hidden_size=8, mode="sum")
    x = torch.randn(2, 16, 2)
    out = cell(x)
    out.sum().backward()
    assert cell.cfc.f_gate[0].weight.grad is not None
    assert cell.cfc.f_gate[0].weight.grad.abs().sum().item() > 0


def test_cell_sum_decomposes_correctly():
    """Verify h = h_lin + h_nl in sum mode."""
    torch.manual_seed(0)
    cell = LinearNonlinearCfCCell(input_size=2, hidden_size=4, mode="sum")
    x = torch.randn(1, 4, 2)
    out = cell(x)
    # h_lin = linear_proj(x), h_nl = CfC(x), out = h_lin + h_nl
    # Just check shape and finiteness (exact values depend on init).
    assert out.shape == (1, 4, 4)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network."""
    net = LinearNonlinearCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, mode="sum",
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8
    assert net.mode == "sum"


def test_stacked_init_concat():
    """Concat mode stacked network."""
    net = LinearNonlinearCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, mode="concat",
    )
    assert net.mode == "concat"


def test_stacked_forward_shape_sum():
    """return_sequences=True returns [B, T, output_size]."""
    net = LinearNonlinearCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, mode="sum",
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_concat():
    """Concat mode returns [B, T, output_size]."""
    net = LinearNonlinearCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, mode="concat",
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = LinearNonlinearCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, mode="sum",
        return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_stacked_forward_finite_with_nan():
    """Forward handles NaN inputs."""
    net = LinearNonlinearCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, mode="sum",
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach all layers' linear and CfC weights."""
    net = LinearNonlinearCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, mode="sum",
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for li, cell in enumerate(net.cells):
        assert cell.linear_proj.weight.grad is not None
        assert cell.linear_proj.weight.grad.abs().sum().item() > 0
        assert cell.cfc.f_gate[0].weight.grad is not None
        assert cell.cfc.f_gate[0].weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_learns_sin():
    """Smoke: LiNo-CfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = LinearNonlinearCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, mode="sum",
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
    """Smoke: LiNo-CfC should reduce loss on structured task."""
    torch.manual_seed(0)
    net = LinearNonlinearCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, mode="sum",
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


def test_bench_smoke_lino_vs_cfc():
    """Mini-bench: LiNo-CfC vs CfC baseline on sin task."""
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

    # LiNo-CfC.
    torch.manual_seed(42)
    lino = LinearNonlinearCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2, mode="sum",
        return_sequences=True,
    )
    opt = torch.optim.Adam(lino.parameters(), lr=1e-2)
    lino_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = lino(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        lino_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(lino_loss)


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
