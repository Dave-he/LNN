"""Round 176 — tests for LearnedPerScaleBeta+KxLadder-CfC (PRD #10-138)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_kxl_cfc import (
    LearnedBetaPSKxlCfCStackedNetwork,
    make_lbps_kxl_5_5_5,
    make_lbps_kxl_3_3_3,
    make_lbps_kxl_7_7_7,
    make_lbps_kxl_3_5_7,
    make_lbps_kxl_7_5_3,
    make_lbps_kxl_3_5_5,
    make_lbps_kxl_7_5_5,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_init_default_ladder():
    """Default Kx_ladder if None provided."""
    net = LearnedBetaPSKxlCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx_ladder=None, Kh=3,
    )
    assert net.Kx_ladder == [5, 5, 5]


def test_init_custom_ladder():
    """Custom Kx_ladder accepted."""
    net = LearnedBetaPSKxlCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx_ladder=[3, 5, 7], Kh=3,
    )
    assert net.Kx_ladder == [3, 5, 7]


def test_init_mismatch_raises():
    """Kx_ladder length must match num_layers."""
    try:
        LearnedBetaPSKxlCfCStackedNetwork(
            input_size=2, hidden_size=8, output_size=1, num_layers=3,
            Kx_ladder=[3, 5],  # only 2 values for 3 layers
        )
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_factory_5_5_5():
    """Kx=[5,5,5] control."""
    net = make_lbps_kxl_5_5_5(input_size=2, hidden_size=8, output_size=1)
    assert net.Kx_ladder == [5, 5, 5]


def test_factory_3_5_7():
    """Kx=[3,5,7] low-to-high."""
    net = make_lbps_kxl_3_5_7(input_size=2, hidden_size=8, output_size=1)
    assert net.Kx_ladder == [3, 5, 7]
    assert net.cells[0].Kx == 3
    assert net.cells[1].Kx == 5
    assert net.cells[2].Kx == 7


def test_factory_7_5_3():
    """Kx=[7,5,3] high-to-low."""
    net = make_lbps_kxl_7_5_3(input_size=2, hidden_size=8, output_size=1)
    assert net.Kx_ladder == [7, 5, 3]


def test_forward_shape():
    """Network forward returns [B, T, output_size]."""
    net = make_lbps_kxl_3_5_7(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_forward_handles_nan():
    """Network forward handles NaN inputs via nan_to_num."""
    net = make_lbps_kxl_3_5_7(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_gradient_flows_all_layers():
    """Gradient reaches all 3 layers with different Kx."""
    net = make_lbps_kxl_3_5_7(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.beta_x_raw.grad is not None
        assert cell.beta_h_raw.grad is not None
        assert cell.beta_x_raw.grad.abs().sum().item() > 0
        assert cell.beta_h_raw.grad.abs().sum().item() > 0


def test_beta_x_per_layer_shape():
    """β_x shape varies with Kx ladder."""
    net = make_lbps_kxl_3_5_7(input_size=2, hidden_size=8, output_size=1)
    assert net.cells[0].beta_x.shape == (3,)
    assert net.cells[1].beta_x.shape == (5,)
    assert net.cells[2].beta_x.shape == (7,)


def test_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = LearnedBetaPSKxlCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx_ladder=[3, 5, 7], Kh=3, return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_smoke_learns_sin():
    """make_lbps_kxl_5_5_5 should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = make_lbps_kxl_5_5_5(input_size=2, hidden_size=12, output_size=1)
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
