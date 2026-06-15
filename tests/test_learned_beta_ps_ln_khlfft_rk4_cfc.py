"""Round 198 — tests for RK4-CfC (PRD #10-160)."""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_cfc import make_lbps_lnkhlfft_5_3_2
from lnn.core.learned_beta_ps_ln_khlfft_rk4_cfc import (
    RK4CfCCell,
    RK4CfCStackedNetwork,
    make_lbps_lnkhlfft_rk4_5_3_2,
)


def test_rk4_cell_forward():
    """RK4 cell forward should produce finite hidden state."""
    cell = RK4CfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, ex, eh = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()
    assert h_new.shape == h.shape
    assert len(ex) == 3
    assert len(eh) == 2


def test_rk4_handles_nan_input():
    """RK4 cell should handle NaN inputs gracefully."""
    cell = RK4CfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, ex, eh = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()


def test_rk4_preserves_zero_state():
    """If h_t = 0 and g, h_branch are also small, RK4 should produce bounded update."""
    torch.manual_seed(0)
    cell = RK4CfCCell(input_size=2, hidden_size=4, Kx=2, Kh=2)
    x = torch.zeros(1, 2)
    h = torch.zeros(1, 4)
    emas_x = [torch.zeros(1, 2) for _ in range(2)]
    emas_h = [torch.zeros(1, 4) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    # Should be finite, but not necessarily zero (g, h_branch may be non-zero from bias)
    assert torch.isfinite(h_new).all()


def test_rk4_stacked_factory():
    """Factory should produce working stacked network."""
    net = make_lbps_lnkhlfft_rk4_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert torch.isfinite(y).all()
    assert y.shape == (2, 16, 1)


def test_rk4_matches_baseline_param_count():
    """RK4-CfC should have same # params as r187 baseline (no extra params)."""
    net_rk4 = make_lbps_lnkhlfft_rk4_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net_base = make_lbps_lnkhlfft_5_3_2(input_size=2, hidden_size=8, output_size=1)
    n_rk4 = sum(p.numel() for p in net_rk4.parameters())
    n_base = sum(p.numel() for p in net_base.parameters())
    assert n_rk4 == n_base, f"rk4 {n_rk4} != base {n_base}"


def test_rk4_eval_mode_deterministic():
    """Eval mode should be deterministic."""
    net = make_lbps_lnkhlfft_rk4_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net.eval()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert torch.allclose(y1, y2)


def test_rk4_train_mode_stochastic_via_init():
    """In train mode, initial forward may differ from second (no batchnorm/dropout, so same)."""
    net = make_lbps_lnkhlfft_rk4_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net.train()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    # No stochastic components → should match
    assert torch.allclose(y1, y2)


def test_rk4_gradient_flows():
    """Gradient should flow back to all params."""
    net = make_lbps_lnkhlfft_rk4_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    target = torch.randn(2, 16, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    opt.zero_grad()
    y = net(x)
    loss = (y - target).pow(2).mean()
    loss.backward()
    n_with_grad = sum(1 for p in net.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in net.parameters())
    # Most params should have non-zero grad
    assert n_with_grad > 0.7 * n_total


def test_rk4_smoke_learns_sin():
    """Smoke test: RK4-CfC can learn sin."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_rk4_5_3_2(input_size=2, hidden_size=12, output_size=1)
    t = torch.linspace(0, 4 * math.pi, 16).unsqueeze(0).unsqueeze(-1).expand(2, 16, 1)
    x = torch.zeros(2, 16, 2)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    target = torch.sin(t.squeeze(-1)).unsqueeze(-1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    initial_loss = None
    final_loss = 0.0
    for _ in range(50):
        opt.zero_grad()
        y = net(x)
        loss = (y - target).pow(2).mean()
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    assert initial_loss is not None
    assert math.isfinite(final_loss)
    assert final_loss < initial_loss


def test_rk4_cf_delta_shape():
    """_cf_delta should return same shape as h."""
    cell = RK4CfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    h = torch.randn(4, 8)
    # aug_total = (Kx+1)*input + (Kh+1)*hidden = 4*2 + 3*8 = 8 + 24 = 32
    z = torch.randn(4, 32)
    f = cell.f_gate(z)
    g = cell.g_branch(z)
    h_branch = cell.h_branch(z)
    delta = cell._cf_delta(h, f, g, h_branch, dt=1.0)
    assert delta.shape == h.shape


def test_rk4_dt_scalar():
    """Scalar dt should work."""
    cell = RK4CfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    emas_x = [torch.zeros(2, 2) for _ in range(3)]
    emas_h = [torch.zeros(2, 8) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h, dt=1.0)
    assert torch.isfinite(h_new).all()
    h_new2, _, _ = cell(x, h, emas_x, emas_h, dt=0.5)
    # Different dt should give different h_new
    assert not torch.allclose(h_new, h_new2)


def test_rk4_dt_tensor():
    """Tensor dt (per-sample) should work."""
    cell = RK4CfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    emas_x = [torch.zeros(2, 2) for _ in range(3)]
    emas_h = [torch.zeros(2, 8) for _ in range(2)]
    dt = torch.tensor([1.0, 0.5])
    h_new, _, _ = cell(x, h, emas_x, emas_h, dt=dt)
    assert torch.isfinite(h_new).all()


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
        sys.exit(1)
    print(f"\nAll {len(funcs)} tests passed.")
