"""Round 177 — tests for LearnedBetaPS+KxKhGrid-CfC (PRD #10-139)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_grid_cfc import (
    make_lbps_grid_3_2,
    make_lbps_grid_3_3,
    make_lbps_grid_3_5,
    make_lbps_grid_5_2,
    make_lbps_grid_5_3,
    make_lbps_grid_5_5,
    make_lbps_grid_7_2,
    make_lbps_grid_7_3,
    make_lbps_grid_7_5,
)


def test_factory_3_2():
    """Kx=3, Kh=2 small-small."""
    net = make_lbps_grid_3_2(input_size=2, hidden_size=8, output_size=1)
    assert net.Kx == 3
    assert net.Kh == 2
    assert len(net.cells) == 3


def test_factory_5_3():
    """Kx=5, Kh=3 control."""
    net = make_lbps_grid_5_3(input_size=2, hidden_size=8, output_size=1)
    assert net.Kx == 5
    assert net.Kh == 3


def test_factory_7_5():
    """Kx=7, Kh=5 large-large."""
    net = make_lbps_grid_7_5(input_size=2, hidden_size=8, output_size=1)
    assert net.Kx == 7
    assert net.Kh == 5


def test_all_factories_wire_correctly():
    """All 9 factories wire Kx, Kh correctly."""
    pairs = [
        (make_lbps_grid_3_2, 3, 2),
        (make_lbps_grid_3_3, 3, 3),
        (make_lbps_grid_3_5, 3, 5),
        (make_lbps_grid_5_2, 5, 2),
        (make_lbps_grid_5_3, 5, 3),
        (make_lbps_grid_5_5, 5, 5),
        (make_lbps_grid_7_2, 7, 2),
        (make_lbps_grid_7_3, 7, 3),
        (make_lbps_grid_7_5, 7, 5),
    ]
    for fn, expected_Kx, expected_Kh in pairs:
        net = fn(input_size=2, hidden_size=8, output_size=1)
        assert net.Kx == expected_Kx, f"{fn.__name__}: Kx={net.Kx} != {expected_Kx}"
        assert net.Kh == expected_Kh, f"{fn.__name__}: Kh={net.Kh} != {expected_Kh}"


def test_forward_shape_3_2():
    """Forward returns correct shape with Kx=3, Kh=2."""
    net = make_lbps_grid_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_forward_shape_7_5():
    """Forward returns correct shape with Kx=7, Kh=5."""
    net = make_lbps_grid_7_5(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_forward_handles_nan():
    """Network handles NaN inputs via nan_to_num."""
    net = make_lbps_grid_3_5(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_gradient_flows():
    """Gradient reaches all 3 layers with different Kx/Kh combos."""
    net = make_lbps_grid_7_5(input_size=2, hidden_size=8, output_size=1)
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
    from lnn.core.learned_beta_ps_cfc import LearnedBetaPSCfCStackedNetwork
    net = LearnedBetaPSCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx=3, Kh=2, return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_smoke_learns_sin():
    """make_lbps_grid_3_2 should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = make_lbps_grid_3_2(input_size=2, hidden_size=12, output_size=1)
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
