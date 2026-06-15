"""Round 171 — tests for LearnedPerScaleBeta-CfC (per-scale learnable β) (PRD #10-133)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_cfc import (
    LearnedBetaPSCfCCell,
    LearnedBetaPSCfCStackedNetwork,
    make_lb_ps_h3_75,
    make_lb_ps_h3_50,
    make_lb_ps_h3_90,
    make_lb_ps_h2_75,
    make_lb_ps_h4_75,
    make_lb_ps_h5_75,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cell_init():
    """Cell initializes with correct shape."""
    cell = LearnedBetaPSCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=3)
    assert cell.Kx == 3
    assert cell.Kh == 3
    assert cell.beta_x_raw.shape == (3,)
    assert cell.beta_h_raw.shape == (3,)


def test_cell_beta_sigmoid_correct():
    """Cell β properties return sigmoid of raw, init=0.75 should give ~0.75."""
    cell = LearnedBetaPSCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=3,
                                beta_x_init=0.75, beta_h_init=0.75)
    bx = cell.beta_x
    bh = cell.beta_h
    assert bx.shape == (3,)
    assert bh.shape == (3,)
    assert torch.allclose(bx, torch.full_like(bx, 0.75), atol=1e-3)
    assert torch.allclose(bh, torch.full_like(bh, 0.75), atol=1e-3)


def test_cell_forward():
    """Cell forward returns tuple (h_new, emas_x_new, emas_h_new)."""
    cell = LearnedBetaPSCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=3)
    x_t = torch.randn(2, 2)
    h_t = torch.randn(2, 8)
    ema_x = [torch.zeros(2, 2) for _ in range(3)]
    ema_h = [torch.zeros(2, 8) for _ in range(3)]
    h_next, new_ema_x, new_ema_h = cell(x_t, h_t, ema_x, ema_h)
    assert h_next.shape == (2, 8)
    assert len(new_ema_x) == 3
    assert len(new_ema_h) == 3


def test_beta_is_trainable():
    """β values are trainable nn.Parameter."""
    net = make_lb_ps_h3_75(input_size=2, hidden_size=8, output_size=1)
    for cell in net.cells:
        assert isinstance(cell.beta_x_raw, torch.nn.Parameter)
        assert isinstance(cell.beta_h_raw, torch.nn.Parameter)
        assert cell.beta_x_raw.requires_grad
        assert cell.beta_h_raw.requires_grad


def test_beta_updates_after_gradient_step():
    """β values change after a gradient step."""
    net = make_lb_ps_h3_75(input_size=2, hidden_size=8, output_size=1)
    initial_bx = net.cells[0].beta_x_raw.clone().detach()
    initial_bh = net.cells[0].beta_h_raw.clone().detach()
    x = torch.randn(2, 16, 2)
    y = torch.randn(2, 16, 1)
    out = net(x)
    loss = F.mse_loss(out, y)
    loss.backward()
    opt = torch.optim.Adam(net.parameters(), lr=0.1)
    opt.step()
    new_bx = net.cells[0].beta_x_raw.clone().detach()
    new_bh = net.cells[0].beta_h_raw.clone().detach()
    assert not torch.allclose(initial_bx, new_bx) or not torch.allclose(initial_bh, new_bh)


def test_factory_h3_75():
    """make_lb_ps_h3_75."""
    net = make_lb_ps_h3_75(input_size=2, hidden_size=8, output_size=1)
    assert net.num_layers == 3
    assert net.Kx == 5
    assert net.Kh == 3


def test_factory_h3_50():
    """make_lb_ps_h3_50: init β=0.5."""
    net = make_lb_ps_h3_50(input_size=2, hidden_size=8, output_size=1)
    assert abs(net.cells[0].beta_x.mean().item() - 0.5) < 0.01


def test_factory_h3_90():
    """make_lb_ps_h3_90: init β=0.9."""
    net = make_lb_ps_h3_90(input_size=2, hidden_size=8, output_size=1)
    assert abs(net.cells[0].beta_x.mean().item() - 0.9) < 0.01


def test_factory_h2_75():
    """make_lb_ps_h2_75: Kh=2."""
    net = make_lb_ps_h2_75(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 2


def test_factory_h4_75():
    """make_lb_ps_h4_75: Kh=4."""
    net = make_lb_ps_h4_75(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 4


def test_factory_h5_75():
    """make_lb_ps_h5_75: Kh=5."""
    net = make_lb_ps_h5_75(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 5


def test_forward_shape():
    """Network forward returns [B, T, output_size]."""
    net = make_lb_ps_h3_75(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_forward_handles_nan():
    """Network forward handles NaN inputs via nan_to_num."""
    net = make_lb_ps_h3_75(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_gradient_flows_all_layers_and_betas():
    """Gradient reaches all 3 layers + both β_x and β_h."""
    net = make_lb_ps_h3_75(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.beta_x_raw.grad is not None
        assert cell.beta_h_raw.grad is not None
        assert cell.beta_x_raw.grad.abs().sum().item() > 0
        assert cell.beta_h_raw.grad.abs().sum().item() > 0


def test_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = LearnedBetaPSCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx=5, Kh=3, return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_smoke_learns_sin():
    """make_lb_ps_h3_75 should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = make_lb_ps_h3_75(input_size=2, hidden_size=12, output_size=1)
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
        out = net(x)
        loss = F.mse_loss(out, target)
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    assert initial_loss is not None
    assert math.isfinite(final_loss)
    assert final_loss < initial_loss


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
