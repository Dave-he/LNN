"""Round 174 — tests for LearnedPerScaleBeta+PerLayerInit-CfC (PRD #10-136)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_init_cfc import (
    LearnedBetaPSInitCfCStackedNetwork,
    make_lbps_init_uniform,
    make_lbps_init_low_to_high,
    make_lbps_init_high_to_low,
    make_lbps_init_wide,
    make_lbps_init_narrow,
    make_lbps_init_kh2_low_to_high,
    make_lbps_init_kh2_high_to_low,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_init_default_inits():
    """Default inits if None provided."""
    net = LearnedBetaPSInitCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx=5, Kh=3, beta_x_inits=None, beta_h_inits=None,
    )
    assert net.beta_x_inits == [0.75, 0.75, 0.75]
    assert net.beta_h_inits == [0.75, 0.75, 0.75]


def test_init_custom_inits():
    """Custom per-layer inits accepted."""
    net = LearnedBetaPSInitCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx=5, Kh=3, beta_x_inits=[0.5, 0.75, 0.95], beta_h_inits=[0.95, 0.75, 0.5],
    )
    assert net.beta_x_inits == [0.5, 0.75, 0.95]
    assert net.beta_h_inits == [0.95, 0.75, 0.5]


def test_init_mismatch_raises():
    """Length mismatch raises."""
    try:
        LearnedBetaPSInitCfCStackedNetwork(
            input_size=2, hidden_size=8, output_size=1, num_layers=3,
            Kx=5, Kh=3, beta_x_inits=[0.5, 0.75],
        )
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_per_layer_inits_differ():
    """Different layers have different initial β values."""
    net = make_lbps_init_low_to_high(input_size=2, hidden_size=8, output_size=1)
    # All scales within a layer have same init, but layers differ.
    bx_l0 = net.cells[0].beta_x
    bx_l2 = net.cells[2].beta_x
    assert abs(bx_l0.mean().item() - 0.5) < 0.01
    assert abs(bx_l2.mean().item() - 0.95) < 0.01


def test_factory_uniform():
    """All layers same init."""
    net = make_lbps_init_uniform(input_size=2, hidden_size=8, output_size=1)
    assert net.beta_x_inits == [0.75, 0.75, 0.75]


def test_factory_low_to_high():
    """β_init ascending."""
    net = make_lbps_init_low_to_high(input_size=2, hidden_size=8, output_size=1)
    assert net.beta_x_inits == [0.5, 0.75, 0.95]


def test_factory_high_to_low():
    """β_init descending."""
    net = make_lbps_init_high_to_low(input_size=2, hidden_size=8, output_size=1)
    assert net.beta_x_inits == [0.95, 0.75, 0.5]


def test_factory_wide():
    """Wide spread [0.5, 0.85, 0.99]."""
    net = make_lbps_init_wide(input_size=2, hidden_size=8, output_size=1)
    assert net.beta_x_inits == [0.5, 0.85, 0.99]


def test_factory_narrow():
    """Narrow spread [0.7, 0.75, 0.8]."""
    net = make_lbps_init_narrow(input_size=2, hidden_size=8, output_size=1)
    assert net.beta_x_inits == [0.7, 0.75, 0.8]


def test_factory_kh2_low_to_high():
    """Kh=2 with ascending init."""
    net = make_lbps_init_kh2_low_to_high(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 2
    assert net.beta_x_inits == [0.5, 0.75, 0.95]


def test_forward_shape():
    """Network forward returns [B, T, output_size]."""
    net = make_lbps_init_low_to_high(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_forward_handles_nan():
    """Network forward handles NaN inputs via nan_to_num."""
    net = make_lbps_init_low_to_high(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_gradient_flows_all_layers():
    """Gradient reaches all 3 layers."""
    net = make_lbps_init_wide(input_size=2, hidden_size=8, output_size=1)
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
    net = LearnedBetaPSInitCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx=5, Kh=3, beta_x_inits=[0.5, 0.75, 0.95],
        return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_smoke_learns_sin():
    """make_lbps_init_uniform should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = make_lbps_init_uniform(input_size=2, hidden_size=12, output_size=1)
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
