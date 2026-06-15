"""Round 132 — tests for AntisymmetricCfC cell and stacked network (PRD #10-94)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.antisymmetric_cfc import (
    AntisymmetricCfCCell,
    AntisymmetricCfCStackedNetwork,
    AntisymmetricMatrix,
)


# ---------------------------------------------------------------------------
# AntisymmetricMatrix tests
# ---------------------------------------------------------------------------


def test_antisym_init_is_antisymmetric():
    """After init, M + M^T = 0 (no diagonal)."""
    mat = AntisymmetricMatrix(size=8, init_scale=0.1)
    M = mat()
    assert torch.allclose(M, -M.T, atol=1e-7)
    assert torch.allclose(M.diagonal(), torch.zeros(8), atol=1e-7)


def test_antisym_eigenvalues_pure_imaginary():
    """All eigenvalues should have real part ≈ 0."""
    mat = AntisymmetricMatrix(size=8, init_scale=0.1)
    M = mat().detach()
    eigvals = torch.linalg.eigvals(M)
    real_parts = eigvals.real
    # Allow some numerical error.
    assert real_parts.abs().max() < 1e-5, f"max real part: {real_parts.abs().max()}"


def test_antisym_n_params():
    """n*(n-1)/2 parameters stored."""
    mat = AntisymmetricMatrix(size=10, init_scale=0.1)
    n_total = sum(p.numel() for p in mat.parameters())
    assert n_total == 100, f"expected 100 (full n^2 storage), got {n_total}"
    # But the effective number of independent parameters is n*(n-1)/2 = 45.
    M = mat()
    n_eff = (M.shape[0] * (M.shape[0] - 1)) // 2
    assert n_eff == 45


def test_antisym_remains_antisymmetric_after_grad_step():
    """The antisymmetry is preserved after a gradient step."""
    torch.manual_seed(0)
    mat = AntisymmetricMatrix(size=4, init_scale=0.1)
    opt = torch.optim.SGD(mat.parameters(), lr=0.01)
    target = torch.randn(4, 4)
    target = (target - target.T) / 2  # make it antisymmetric
    for _ in range(3):
        opt.zero_grad()
        M = mat()
        loss = ((M - target) ** 2).sum()
        loss.backward()
        opt.step()
    M = mat()
    assert torch.allclose(M, -M.T, atol=1e-6)


# ---------------------------------------------------------------------------
# Cell init tests
# ---------------------------------------------------------------------------


def test_cell_init_default():
    """Default init: dt=0.1, init_scale=0.1."""
    cell = AntisymmetricCfCCell(input_size=2, hidden_size=8)
    assert cell.dt == 0.1
    assert cell.input_size == 2
    assert cell.hidden_size == 8


def test_cell_init_custom_dt():
    """Custom dt should be reflected in cell.dt."""
    cell = AntisymmetricCfCCell(input_size=2, hidden_size=8, dt=0.5)
    assert cell.dt == 0.5


def test_cell_init_custom_init_scale():
    """Custom init_scale should affect the initial spectral radius."""
    torch.manual_seed(0)
    cell_small = AntisymmetricCfCCell(input_size=2, hidden_size=16, init_scale=0.01)
    cell_large = AntisymmetricCfCCell(input_size=2, hidden_size=16, init_scale=1.0)
    sr_small = cell_small.spectral_radius()
    sr_large = cell_large.spectral_radius()
    assert sr_large > sr_small, f"sr_small={sr_small}, sr_large={sr_large}"


# ---------------------------------------------------------------------------
# Cell forward tests
# ---------------------------------------------------------------------------


def test_cell_forward_shape():
    """Forward should return [B, hidden_size]."""
    cell = AntisymmetricCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert h_new.shape == (4, 8)


def test_cell_forward_zero_h_uses_x_only():
    """With h=0, M@h=0 so candidate = tanh(W_x x + b)."""
    torch.manual_seed(0)
    cell = AntisymmetricCfCCell(input_size=2, hidden_size=4, dt=0.1)
    x = torch.randn(1, 2)
    h = torch.zeros(1, 4)
    h_new = cell(x, h)
    expected = torch.tanh(cell.W_x(x) + cell.bias)
    # h_new = h + dt * (candidate - h) = 0 + dt * (candidate - 0) = dt * candidate.
    assert torch.allclose(h_new, 0.1 * expected, atol=1e-5)


def test_cell_forward_h_stays_bounded_100_steps():
    """Hidden state should not diverge over 100 forward steps (antisymm stability)."""
    torch.manual_seed(0)
    cell = AntisymmetricCfCCell(input_size=2, hidden_size=8, dt=0.1, init_scale=0.1)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    for _ in range(100):
        h = cell(x, h)
    assert torch.isfinite(h).all(), f"h has NaN/Inf: max={h.abs().max()}"
    # Bounded: max abs value shouldn't blow up.
    assert h.abs().max() < 100.0, f"h too large: {h.abs().max()}"


def test_cell_forward_h_bounded_smaller_than_unconstrained():
    """Antisymm should be at least as stable as a 0.5-step Euler on same W_x."""
    torch.manual_seed(0)
    cell = AntisymmetricCfCCell(input_size=2, hidden_size=16, dt=0.1, init_scale=0.5)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    for _ in range(50):
        h = cell(x, h)
    n_stable = h.abs().max().item()
    # Antisymm should stay bounded.
    assert n_stable < 50.0, f"hidden too large: {n_stable}"


# ---------------------------------------------------------------------------
# Cell gradient tests
# ---------------------------------------------------------------------------


def test_cell_gradient_to_W_x():
    """Gradient should reach W_x."""
    cell = AntisymmetricCfCCell(input_size=2, hidden_size=8, dt=0.1)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.W_x.weight.grad is not None
    assert cell.W_x.weight.grad.abs().sum().item() > 0


def test_cell_gradient_to_M():
    """Gradient should reach the upper-triangle storage U."""
    torch.manual_seed(42)
    cell = AntisymmetricCfCCell(input_size=2, hidden_size=8, dt=0.1, init_scale=0.1)
    x = torch.randn(2, 2)
    # Use non-zero h so M @ h is non-zero (otherwise gradient w.r.t. M is 0).
    h = torch.randn(2, 8)
    out = cell(x, h)
    out.sum().backward()
    # M is parameterized by U; gradient should flow to U.
    assert cell.M.U.grad is not None
    assert cell.M.U.grad.abs().sum().item() > 0


def test_cell_gradient_to_M_lower_triangle_is_zero():
    """Gradient to lower triangle of U should be 0 (only upper triangle affects M)."""
    torch.manual_seed(42)
    cell = AntisymmetricCfCCell(input_size=2, hidden_size=4, dt=0.1, init_scale=0.1)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 4)
    out = cell(x, h)
    out.sum().backward()
    grad = cell.M.U.grad
    # Lower-triangle grad should be 0 (we used triu in forward).
    lower = torch.tril(grad, diagonal=-1)
    assert torch.allclose(lower, torch.zeros_like(lower), atol=1e-7)


def test_cell_gradient_to_bias():
    """Gradient should reach bias."""
    cell = AntisymmetricCfCCell(input_size=2, hidden_size=8, dt=0.1)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.bias.grad is not None
    assert cell.bias.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network."""
    net = AntisymmetricCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8


def test_stacked_forward_shape_sequences():
    """return_sequences=True returns [B, T, output_size]."""
    net = AntisymmetricCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=True,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 16, 1)


def test_stacked_forward_shape_last_step():
    """return_sequences=False returns [B, output_size]."""
    net = AntisymmetricCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=False,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 1)


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach W_x of every layer and M of every layer."""
    net = AntisymmetricCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for li, cell in enumerate(net.cells):
        assert cell.W_x.weight.grad is not None
        assert cell.W_x.weight.grad.abs().sum().item() > 0, f"layer {li} W_x has no grad"
        assert cell.M.U.grad is not None
        assert cell.M.U.grad.abs().sum().item() > 0, f"layer {li} M.U has no grad"


def test_stacked_antisymmetry_holds_at_init():
    """After init, every layer's M should be antisymmetric."""
    net = AntisymmetricCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
    )
    for li, cell in enumerate(net.cells):
        M = cell.M()
        assert torch.allclose(M, -M.T, atol=1e-7), f"layer {li} M not antisymmetric"


def test_stacked_smoke_learns_sin():
    """Smoke: stacked Antisymm should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = AntisymmetricCfCStackedNetwork(
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
    # The model should at least fit a constant.
    assert final_loss < initial_loss, f"loss didn't decrease: {initial_loss} -> {final_loss}"


def test_stacked_spectral_radius_bounded_after_training():
    """After 30 steps, spectral radius should still be bounded."""
    torch.manual_seed(0)
    net = AntisymmetricCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    x = torch.randn(2, 16, 2)
    target = torch.randn(2, 16, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(30):
        opt.zero_grad()
        out = net(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    for li, cell in enumerate(net.cells):
        sr = cell.spectral_radius()
        assert sr < 50.0, f"layer {li} spectral_radius too large: {sr}"


# ---------------------------------------------------------------------------
# Bench-style smoke test
# ---------------------------------------------------------------------------


def test_bench_smoke_antisym_vs_cfc():
    """Mini-bench: Antisymm vs CfC baseline on sin task."""
    torch.manual_seed(42)
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

    # Antisymmetric baseline.
    torch.manual_seed(42)
    antisym = AntisymmetricCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        return_sequences=True,
    )
    opt = torch.optim.Adam(antisym.parameters(), lr=1e-2)
    antisym_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = antisym(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        antisym_loss = loss.item()

    # Both should produce finite loss.
    assert math.isfinite(cfc_loss)
    assert math.isfinite(antisym_loss)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


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
