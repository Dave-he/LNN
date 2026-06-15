"""Round 141 — tests for Adaptive Time-Constant CfC (PRD #10-103)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.adaptive_time_constant_cfc import (
    AdaptiveTimeConstantCfCCell,
    AdaptiveTimeConstantCfCStackedNetwork,
)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


def test_init_default_concat():
    """Default init: mode='concat'."""
    cell = AdaptiveTimeConstantCfCCell(input_size=2, hidden_size=8)
    assert cell.mode == "concat"
    assert cell.input_size == 2
    assert cell.hidden_size == 8


def test_init_modes():
    """Test all ATC modes."""
    for mode in ("concat", "input"):
        cell = AdaptiveTimeConstantCfCCell(input_size=2, hidden_size=8, mode=mode)
        assert cell.mode == mode


def test_init_invalid_mode():
    """Invalid mode should raise."""
    try:
        AdaptiveTimeConstantCfCCell(input_size=2, hidden_size=8, mode="invalid")
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_init_tau_at_time_scale_init():
    """Tau should equal time_scale_init at init (when x=0, h=0)."""
    cell = AdaptiveTimeConstantCfCCell(input_size=2, hidden_size=8, time_scale_init=2.0)
    x = torch.zeros(1, 2)
    h = torch.zeros(1, 8)
    tau = cell.tau(x, h)
    # softplus(b) + 1 = 2.0 → b = softplus_inv(1.0) = log(exp(1) - 1)
    assert torch.allclose(tau, torch.full((1, 8), 2.0), atol=1e-4)


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------


def test_forward_shape():
    """Forward should return [B, hidden_size]."""
    cell = AdaptiveTimeConstantCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert h_new.shape == (4, 8)


def test_forward_finite():
    """Forward output should be finite."""
    cell = AdaptiveTimeConstantCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2) * 5.0
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert torch.isfinite(h_new).all()


def test_forward_stability_100_steps():
    """No NaN/Inf in 100 sequential forward steps."""
    torch.manual_seed(0)
    cell = AdaptiveTimeConstantCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    for _ in range(100):
        h = cell(x, h)
    assert torch.isfinite(h).all()


def test_forward_tau_positive():
    """Tau should always be >= 1.0."""
    cell = AdaptiveTimeConstantCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2) * 100.0
    h = torch.randn(4, 8) * 100.0
    tau = cell.tau(x, h)
    assert (tau >= 1.0).all()


def test_forward_tau_input_conditional():
    """Different x should give different tau (when h same)."""
    cell = AdaptiveTimeConstantCfCCell(input_size=2, hidden_size=8)
    h = torch.zeros(1, 8)
    x1 = torch.tensor([[1.0, 0.0]])
    x2 = torch.tensor([[0.0, 1.0]])
    # At init with weights=0, tau = softplus(bias) + 1 = constant.
    # To make this test meaningful, we use random weights.
    torch.manual_seed(0)
    cell2 = AdaptiveTimeConstantCfCCell(input_size=2, hidden_size=8)
    with torch.no_grad():
        cell2.tau_net.weight.normal_(std=0.1)
    tau1 = cell2.tau(x1, h)
    tau2 = cell2.tau(x2, h)
    assert not torch.allclose(tau1, tau2, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


def test_gradient_to_tau_net():
    """Gradient should reach the tau_net weights."""
    cell = AdaptiveTimeConstantCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.tau_net.weight.grad is not None
    assert cell.tau_net.weight.grad.abs().sum().item() > 0


def test_gradient_to_W_f():
    """Gradient should reach the CfC f_gate weights."""
    cell = AdaptiveTimeConstantCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_W_g():
    """Gradient should reach the CfC g_branch weights."""
    cell = AdaptiveTimeConstantCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.g_branch[0].weight.grad is not None
    assert cell.g_branch[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_W_h():
    """Gradient should reach the CfC h_branch weights."""
    cell = AdaptiveTimeConstantCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.h_branch[0].weight.grad is not None
    assert cell.h_branch[0].weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network."""
    net = AdaptiveTimeConstantCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8


def test_stacked_forward_shape_sequences():
    """return_sequences=True returns [B, T, output_size]."""
    net = AdaptiveTimeConstantCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=True,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 16, 1)


def test_stacked_forward_shape_last_step():
    """return_sequences=False returns [B, output_size]."""
    net = AdaptiveTimeConstantCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=False,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 1)


def test_stacked_handles_nan_input():
    """Forward should handle NaN inputs (zero-fill)."""
    net = AdaptiveTimeConstantCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach all layers' parameters."""
    net = AdaptiveTimeConstantCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.g_branch[0].weight.grad is not None
        assert cell.g_branch[0].weight.grad.abs().sum().item() > 0
        assert cell.tau_net.weight.grad is not None
        assert cell.tau_net.weight.grad.abs().sum().item() > 0


def test_stacked_smoke_learns_sin():
    """Smoke: stacked ATC-CfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = AdaptiveTimeConstantCfCStackedNetwork(
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


def test_bench_smoke_atc_vs_cfc():
    """Mini-bench: ATC-CfC vs CfC baseline on sin task."""
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

    # ATC-CfC.
    torch.manual_seed(42)
    atc = AdaptiveTimeConstantCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        return_sequences=True,
    )
    opt = torch.optim.Adam(atc.parameters(), lr=1e-2)
    atc_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = atc(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        atc_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(atc_loss)


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
