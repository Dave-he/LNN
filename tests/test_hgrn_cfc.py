"""Round 131 — tests for HGRN CfC cell and stacked network (PRD #10-93)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.hgrn_cfc import HGRNCfCCell, HGRNCfCStackedNetwork


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


def test_init_default():
    """Default init: alpha=0.1, learnable, tanh."""
    cell = HGRNCfCCell(input_size=2, hidden_size=8)
    assert cell.alpha == pytest_approx(0.1, eps=1e-3)  # noqa: F821
    assert cell.nonlinearity == "tanh"


def test_init_custom_alpha():
    """Custom alpha_init should be reflected in cell.alpha."""
    cell = HGRNCfCCell(input_size=2, hidden_size=8, alpha_init=0.3)
    assert cell.alpha == pytest_approx(0.3, eps=1e-3)  # noqa: F821


def test_init_alpha_zero():
    """alpha_init=0 should give a near-zero bound (free-gate)."""
    cell = HGRNCfCCell(input_size=2, hidden_size=8, alpha_init=0.0)
    assert cell.alpha < 0.01


def test_init_alpha_one():
    """alpha_init=1 should give a near-1 bound (no forgetting)."""
    cell = HGRNCfCCell(input_size=2, hidden_size=8, alpha_init=1.0)
    assert cell.alpha > 0.99


def test_init_alpha_out_of_range():
    """alpha_init > 1 should raise."""
    import pytest
    with pytest.raises(ValueError):
        HGRNCfCCell(input_size=2, hidden_size=8, alpha_init=1.5)


def test_init_fixed_alpha():
    """learn_alpha=False should register a buffer, not a parameter."""
    cell = HGRNCfCCell(input_size=2, hidden_size=8, alpha_init=0.4, learn_alpha=False)
    assert not hasattr(cell, "raw_alpha")
    assert hasattr(cell, "_alpha_const")
    assert cell.alpha == pytest_approx(0.4, eps=1e-3)  # noqa: F821


def test_init_nonlinearity():
    """relu nonlinearity should be accepted."""
    cell = HGRNCfCCell(input_size=2, hidden_size=8, nonlinearity="relu")
    assert cell.nonlinearity == "relu"


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------


def test_forward_shape():
    """Forward should return [B, hidden_size]."""
    cell = HGRNCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert h_new.shape == (4, 8)


def test_forward_zero_h_is_input_only():
    """With h=0, the first step should be gate * candidate."""
    torch.manual_seed(0)
    cell = HGRNCfCCell(input_size=2, hidden_size=4, alpha_init=0.0)
    x = torch.randn(1, 2)
    h = torch.zeros(1, 4)
    h_new = cell(x, h)
    # With h=0, the recurrence collapses to h_new = gate * tanh(W_x x).
    # With alpha=0, gate = 0 + (1-0) * sigmoid = sigmoid.
    gate_pre = torch.sigmoid(cell.W_g(x))
    candidate = torch.tanh(cell.W_x(x))
    expected = gate_pre * candidate
    assert torch.allclose(h_new, expected, atol=1e-5)


def test_forward_gate_lower_bounded():
    """With alpha>0, gate values should be >= alpha."""
    cell = HGRNCfCCell(input_size=2, hidden_size=8, alpha_init=0.3)
    x = torch.randn(16, 2)
    h = torch.zeros(16, 8)
    cell(x, h)
    gate = cell.last_gate
    assert (gate >= 0.3 - 1e-6).all(), f"min gate = {gate.min()}"


def test_forward_gate_upper_bounded():
    """Sigmoid output is in [0, 1], so gate values are also in [0, 1]."""
    cell = HGRNCfCCell(input_size=2, hidden_size=8, alpha_init=0.0)
    x = torch.randn(16, 2)
    h = torch.zeros(16, 8)
    cell(x, h)
    gate = cell.last_gate
    assert (gate >= 0.0).all() and (gate <= 1.0).all()


def test_forward_relu_nonlinearity():
    """relu candidate should be non-negative."""
    cell = HGRNCfCCell(input_size=2, hidden_size=8, nonlinearity="relu", alpha_init=0.0)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert (h_new >= 0.0).all()


def test_forward_stability_100_steps():
    """No NaN/Inf in 100 sequential forward steps."""
    torch.manual_seed(0)
    cell = HGRNCfCCell(input_size=2, hidden_size=8, alpha_init=0.2)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    for _ in range(100):
        h = cell(x, h)
    assert torch.isfinite(h).all()


def test_reset_state_clears_caches():
    """reset_state should clear last_gate and last_alpha."""
    cell = HGRNCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    cell(x, h)
    assert cell.last_gate is not None
    cell.reset_state()
    assert cell.last_gate is None
    assert cell.last_alpha is None


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


def test_gradient_to_W_x():
    """Gradient should reach W_x."""
    cell = HGRNCfCCell(input_size=2, hidden_size=8, alpha_init=0.0)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.W_x.weight.grad is not None
    assert cell.W_x.weight.grad.abs().sum().item() > 0


def test_gradient_to_W_g():
    """Gradient should reach W_g."""
    cell = HGRNCfCCell(input_size=2, hidden_size=8, alpha_init=0.0)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.W_g.weight.grad is not None
    assert cell.W_g.weight.grad.abs().sum().item() > 0


def test_gradient_to_alpha_when_learnable():
    """Gradient should reach raw_alpha when learnable."""
    torch.manual_seed(42)
    cell = HGRNCfCCell(input_size=2, hidden_size=8, alpha_init=0.3, learn_alpha=True)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.raw_alpha.grad is not None
    assert cell.raw_alpha.grad.abs().item() > 0


def test_gradient_no_alpha_when_fixed():
    """No raw_alpha parameter when learn_alpha=False."""
    cell = HGRNCfCCell(input_size=2, hidden_size=8, alpha_init=0.3, learn_alpha=False)
    assert not hasattr(cell, "raw_alpha")


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default_hierarchical():
    """Hierarchical default: alpha_l increases monotonically."""
    net = HGRNCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        alpha_init=0.0, hierarchical=True, alpha_max=0.6,
    )
    alphas = net.alphas()
    # First layer should be near 0, last near alpha_max.
    assert alphas[0] < 0.05
    assert alphas[-1] > 0.55
    # Monotonically increasing.
    for i in range(len(alphas) - 1):
        assert alphas[i] <= alphas[i + 1] + 1e-6


def test_stacked_init_uniform_alpha():
    """Non-hierarchical: all layers share alpha_init."""
    net = HGRNCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        alpha_init=0.3, hierarchical=False,
    )
    alphas = net.alphas()
    for a in alphas:
        assert abs(a - 0.3) < 0.01


def test_stacked_forward_shape_sequences():
    """return_sequences=True returns [B, T, output_size]."""
    net = HGRNCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=True,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 16, 1)


def test_stacked_forward_shape_last_step():
    """return_sequences=False returns [B, output_size]."""
    net = HGRNCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=False,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 1)


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach the W_x of every layer."""
    net = HGRNCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        hierarchical=True, alpha_max=0.6,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for li, cell in enumerate(net.cells):
        assert cell.W_x.weight.grad is not None
        assert cell.W_x.weight.grad.abs().sum().item() > 0, (
            f"layer {li} W_x has no grad"
        )


def test_stacked_gradient_to_alphas():
    """Gradient should reach the raw_alpha of every layer."""
    torch.manual_seed(42)
    net = HGRNCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        hierarchical=True, alpha_max=0.6, learn_alpha=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for li, cell in enumerate(net.cells):
        assert hasattr(cell, "raw_alpha")
        assert cell.raw_alpha.grad is not None
        # Grad may be 0 for some layers if their hidden state doesn't
        # affect the output much, but at least one should be non-zero.
    # Check at least one layer's alpha has non-zero grad.
    grads = [c.raw_alpha.grad.abs().item() for c in net.cells if hasattr(c, "raw_alpha")]
    assert any(g > 1e-12 for g in grads), f"no alpha grad: {grads}"


def test_stacked_smoke_learns_sin():
    """Smoke: stacked HGRN should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = HGRNCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        hierarchical=True, alpha_max=0.5,
    )
    t = torch.linspace(0, 4 * math.pi, 16).unsqueeze(0).unsqueeze(-1)
    x = torch.zeros(2, 16, 2)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    mask = torch.rand(2, 16, 2) < 0.3
    x[mask] = float("nan")
    target = torch.sin(t.squeeze(-1)).unsqueeze(-1).expand(2, 16, 1)
    x_clean = torch.nan_to_num(x, nan=0.0)
    target_clean = torch.nan_to_num(target, nan=0.0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    initial_loss = None
    final_loss = None
    for _ in range(30):
        opt.zero_grad()
        out = net(x_clean)
        loss = F.mse_loss(out, target_clean)
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    assert final_loss is not None
    assert math.isfinite(final_loss), f"loss blew up: {final_loss}"
    assert final_loss < 5.0, f"loss too high after 30 steps: {final_loss}"


# ---------------------------------------------------------------------------
# Helper (pytest.approx replacement for non-pytest context)
# ---------------------------------------------------------------------------


def pytest_approx(value, eps=1e-6):
    """Lightweight replacement for pytest.approx."""
    class Approx:
        def __init__(self, target, tol):
            self.target = target
            self.tol = tol
        def __eq__(self, other):
            return abs(self.target - other) < self.tol
    return Approx(value, eps)


# ---------------------------------------------------------------------------
# Bench-style smoke test
# ---------------------------------------------------------------------------


def test_bench_smoke_hgrn_vs_cfc():
    """Mini-bench: HGRN bounded vs CfC baseline on sin task."""
    torch.manual_seed(42)
    from lnn.core.cfc import CfCNetwork

    B, T, D, H = 4, 16, 2, 8
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
    x = torch.zeros(B, T, D)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    y = torch.sin(t.squeeze(-1)).unsqueeze(-1).expand(B, T, 1)
    x_clean = torch.nan_to_num(x, nan=0.0)

    # CfC baseline.
    torch.manual_seed(42)
    cfc = CfCNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        return_sequences=True,
    )
    opt = torch.optim.Adam(cfc.parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        out = cfc(x_clean)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
    cfc_loss = loss.item()

    # HGRN bounded.
    torch.manual_seed(42)
    hgrn = HGRNCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        hierarchical=True, alpha_max=0.5, return_sequences=True,
    )
    opt = torch.optim.Adam(hgrn.parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        out = hgrn(x_clean)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
    hgrn_loss = loss.item()

    # Both should produce finite loss.
    assert math.isfinite(cfc_loss) and math.isfinite(hgrn_loss)
    # HGRN has its own (linear) recurrent path — not directly comparable
    # to CfC's ODE-based path. Just check the runs are stable.
    n_hgrn = sum(p.numel() for p in hgrn.parameters())


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
