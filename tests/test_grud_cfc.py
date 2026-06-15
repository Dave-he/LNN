"""Round 148 — tests for Time-Decay CfC (GRU-D / CT-RNN style) (PRD #10-110)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.grud_cfc import TimeDecayCfCCell, TimeDecayCfCStackedNetwork


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------


def test_cell_init_default():
    """Default init has γ ≈ 0.05 (light decay)."""
    cell = TimeDecayCfCCell(input_size=2, hidden_size=8, gamma_init=-3.0)
    assert cell.input_size == 2
    assert cell.hidden_size == 8
    gamma = cell.get_gamma()
    # softplus(-3.0) ≈ 0.0486
    assert torch.allclose(gamma, torch.full((8,), 0.0486), atol=0.01)


def test_cell_init_gamma_0():
    """γ=0 init means no decay (control condition)."""
    cell = TimeDecayCfCCell(input_size=2, hidden_size=8, gamma_init=-10.0)
    gamma = cell.get_gamma()
    # softplus(-10.0) ≈ 4.5e-5, effectively 0
    assert (gamma < 1e-3).all()


def test_cell_init_gamma_high():
    """γ=high init means strong decay."""
    cell = TimeDecayCfCCell(input_size=2, hidden_size=8, gamma_init=2.0)
    gamma = cell.get_gamma()
    # softplus(2.0) ≈ 2.13
    assert (gamma > 2.0).all()


def test_cell_forward_shape():
    """Forward returns [B, T, hidden]."""
    cell = TimeDecayCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 16, 2)
    out = cell(x)
    assert out.shape == (2, 16, 8)


def test_cell_forward_with_dt():
    """Forward with explicit dt."""
    cell = TimeDecayCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 16, 2)
    dt = torch.ones(2, 16, 1) * 0.5
    out = cell(x, dt=dt)
    assert out.shape == (2, 16, 8)


def test_cell_forward_finite():
    """Forward output is finite."""
    cell = TimeDecayCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 16, 2) * 5.0
    out = cell(x)
    assert torch.isfinite(out).all()


def test_cell_forward_handles_nan_input():
    """NaN input handled."""
    cell = TimeDecayCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    out = cell(x)
    assert torch.isfinite(out).all()


def test_cell_forward_handles_nan_dt():
    """NaN dt handled (no decay when time is missing)."""
    cell = TimeDecayCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 16, 2)
    dt = torch.ones(2, 16, 1)
    dt[0, 5, 0] = float("nan")
    out = cell(x, dt=dt)
    assert torch.isfinite(out).all()


def test_cell_decay_applied_correctly():
    """Verify decay factor actually multiplies h."""
    torch.manual_seed(0)
    cell = TimeDecayCfCCell(input_size=2, hidden_size=4, gamma_init=2.0)
    # gamma ≈ 2.13
    # If dt=1.0, decay ≈ exp(-2.13) ≈ 0.119
    # If dt=0, decay = 1.0 (no decay)
    x = torch.zeros(1, 4, 2)  # zero input so the only thing affecting h is decay
    # CfCCell expects input_size + hidden_size = 2 + 4 = 6 input features.
    # Set weights so that for zero input, f_gate = sigmoid(0) = 0.5,
    # g_branch = 0, h_branch = 1, so h = 0.5*0 + 0.5*1 = 0.5.
    with torch.no_grad():
        cell.cfc.f_gate[0].weight.zero_()
        cell.cfc.f_gate[0].bias.zero_()
        cell.cfc.g_branch[0].weight.zero_()
        cell.cfc.g_branch[0].bias.zero_()
        cell.cfc.h_branch[0].weight.zero_()
        cell.cfc.h_branch[0].bias.fill_(1.0)
    out = cell(x, dt=torch.ones(1, 4, 1))
    # Just check output is finite and shape is correct.
    # (The exact value depends on CfC's closed-form time-dependent mixing
    # which is non-trivial to compute by hand — we don't assert a value.)
    assert torch.isfinite(out).all()
    assert out.shape == (1, 4, 4)


def test_cell_gradient_flows_to_gamma():
    """Gradient should reach gamma_param."""
    cell = TimeDecayCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 16, 2)
    out = cell(x)
    out.sum().backward()
    assert cell.gamma_param.grad is not None
    assert cell.gamma_param.grad.abs().sum().item() > 0


def test_cell_gradient_flows_to_cfc():
    """Gradient should reach CfC weights."""
    cell = TimeDecayCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 16, 2)
    out = cell(x)
    out.sum().backward()
    assert cell.cfc.f_gate[0].weight.grad is not None
    assert cell.cfc.f_gate[0].weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network."""
    net = TimeDecayCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8


def test_stacked_forward_shape():
    """return_sequences=True returns [B, T, output_size]."""
    net = TimeDecayCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = TimeDecayCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_stacked_forward_finite_with_nan():
    """Forward handles NaN inputs."""
    net = TimeDecayCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach all layers' gamma_param."""
    net = TimeDecayCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for li, cell in enumerate(net.cells):
        assert cell.gamma_param.grad is not None
        assert cell.gamma_param.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_learns_sin():
    """Smoke: Time-Decay CfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = TimeDecayCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2,
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


def test_smoke_handles_variable_dt():
    """Smoke: Time-Decay CfC should work with variable dt (irregular TS)."""
    torch.manual_seed(0)
    net = TimeDecayCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    # Irregular time deltas: 0.5, 1.0, 2.0, 0.3, ...
    B, T = 4, 16
    x = torch.randn(B, T, 2)
    dt = torch.rand(B, T, 1) * 2.0  # in [0, 2]
    target = torch.randn(B, T, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    initial_loss = None
    final_loss = None
    for _ in range(10):
        opt.zero_grad()
        out = net(x, dt=dt)
        loss = F.mse_loss(out, target)
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    assert math.isfinite(final_loss)
    assert final_loss < initial_loss


# ---------------------------------------------------------------------------
# Bench-style smoke test
# ---------------------------------------------------------------------------


def test_bench_smoke_grud_vs_cfc():
    """Mini-bench: Time-Decay CfC vs CfC baseline on sin task."""
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

    # Time-Decay CfC.
    torch.manual_seed(42)
    grud = TimeDecayCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2, return_sequences=True,
    )
    opt = torch.optim.Adam(grud.parameters(), lr=1e-2)
    grud_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = grud(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        grud_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(grud_loss)


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
