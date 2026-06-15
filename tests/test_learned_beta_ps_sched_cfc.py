"""Round 172 — tests for LearnedPerScaleBeta+Schedule-CfC (PRD #10-134)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_sched_cfc import (
    LearnedBetaPSSchedCfCCell,
    LearnedBetaPSSchedCfCStackedNetwork,
    make_lbps_h3_75_const,
    make_lbps_h3_75_linear,
    make_lbps_h3_75_reverse,
    make_lbps_h2_75_const,
    make_lbps_h2_75_reverse,
    make_lbps_h5_75_const,
    make_lbps_h5_75_reverse,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cell_init():
    """Cell initializes with correct shape."""
    cell = LearnedBetaPSSchedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=3,
                                     layer_idx=0, num_layers=3)
    assert cell.Kx == 3
    assert cell.Kh == 3
    assert cell.beta_x_raw.shape == (3,)
    assert cell.beta_h_raw.shape == (3,)


def test_cell_init_schedule_mode():
    """Cell respects schedule_mode parameter."""
    cell_const = LearnedBetaPSSchedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=3,
                                           schedule_mode="constant", layer_idx=0, num_layers=3)
    cell_rev = LearnedBetaPSSchedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=3,
                                         schedule_mode="reverse", layer_idx=0, num_layers=3)
    assert cell_const.schedule_mode == "constant"
    assert cell_rev.schedule_mode == "reverse"


def test_cell_forward():
    """Cell forward returns tuple (h_new, emas_x_new, emas_h_new)."""
    cell = LearnedBetaPSSchedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=3,
                                     layer_idx=0, num_layers=3)
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
    net = make_lbps_h3_75_const(input_size=2, hidden_size=8, output_size=1)
    for cell in net.cells:
        assert isinstance(cell.beta_x_raw, torch.nn.Parameter)
        assert isinstance(cell.beta_h_raw, torch.nn.Parameter)
        assert cell.beta_x_raw.requires_grad
        assert cell.beta_h_raw.requires_grad


def test_per_layer_schedule_differs():
    """Different layers have different β with linear/reverse schedule."""
    net = make_lbps_h3_75_reverse(input_size=2, hidden_size=8, output_size=1)
    # REVERSE: layer 0 has scale=1.0, layer 2 has scale=0.5
    bx_l0 = net.cells[0].beta_x
    bx_l2 = net.cells[2].beta_x
    # Should differ
    assert not torch.allclose(bx_l0, bx_l2)


def test_schedule_constant_keeps_beta():
    """Constant schedule keeps β at base value."""
    net = make_lbps_h3_75_const(input_size=2, hidden_size=8, output_size=1)
    for cell in net.cells:
        bx = cell.beta_x
        # Should be sigmoid of base raw, no schedule modification
        assert torch.allclose(bx, torch.full_like(bx, 0.75), atol=1e-3)


def test_factory_h3_const():
    """make_lbps_h3_75_const."""
    net = make_lbps_h3_75_const(input_size=2, hidden_size=8, output_size=1)
    assert net.num_layers == 3
    assert net.cells[0].schedule_mode == "constant"


def test_factory_h3_linear():
    """make_lbps_h3_75_linear."""
    net = make_lbps_h3_75_linear(input_size=2, hidden_size=8, output_size=1)
    assert net.cells[0].schedule_mode == "linear"


def test_factory_h3_reverse():
    """make_lbps_h3_75_reverse."""
    net = make_lbps_h3_75_reverse(input_size=2, hidden_size=8, output_size=1)
    assert net.cells[0].schedule_mode == "reverse"


def test_factory_h2_const():
    """make_lbps_h2_75_const: Kh=2."""
    net = make_lbps_h2_75_const(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 2


def test_factory_h2_reverse():
    """make_lbps_h2_75_reverse: Kh=2, reverse."""
    net = make_lbps_h2_75_reverse(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 2
    assert net.cells[0].schedule_mode == "reverse"


def test_factory_h5_const():
    """make_lbps_h5_75_const: Kh=5."""
    net = make_lbps_h5_75_const(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 5


def test_factory_h5_reverse():
    """make_lbps_h5_75_reverse: Kh=5, reverse."""
    net = make_lbps_h5_75_reverse(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 5
    assert net.cells[0].schedule_mode == "reverse"


def test_forward_shape():
    """Network forward returns [B, T, output_size]."""
    net = make_lbps_h3_75_const(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_forward_handles_nan():
    """Network forward handles NaN inputs via nan_to_num."""
    net = make_lbps_h3_75_const(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_gradient_flows_all_layers_and_betas():
    """Gradient reaches all 3 layers + both β_x and β_h."""
    net = make_lbps_h3_75_reverse(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.beta_x_raw.grad is not None
        assert cell.beta_h_raw.grad is not None
        assert cell.beta_x_raw.grad.abs().sum().item() > 0
        assert cell.beta_h_raw.grad.abs().sum().item() > 0


def test_smoke_learns_sin():
    """make_lbps_h3_75_const should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = make_lbps_h3_75_const(input_size=2, hidden_size=12, output_size=1)
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
