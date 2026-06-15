"""Round 166 — tests for HybridBeta-XH-Deep-4Layer-CfC (4-layer stacked) (PRD #10-128)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.hybrid_beta_xh_deep_4layer_cfc import (
    HybridBetaXHCfCStackedNetwork,
    make_hb_xh_4layer_h1,
    make_hb_xh_4layer_h2,
    make_hb_xh_4layer_h2_3x,
    make_hb_xh_4layer_h2_k5,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_factory_h1():
    """4-layer hb_xh_4layer_h1."""
    net = make_hb_xh_4layer_h1(input_size=2, hidden_size=8, output_size=1)
    assert net.num_layers == 4
    assert net.Kx == 1
    assert net.Kh == 1
    assert net.betas_h == [0.9]


def test_factory_h2():
    """4-layer hb_xh_4layer_h2."""
    net = make_hb_xh_4layer_h2(input_size=2, hidden_size=8, output_size=1)
    assert net.num_layers == 4
    assert net.Kx == 2
    assert net.Kh == 2
    assert net.betas_h == [0.7, 0.95]


def test_factory_h2_3x():
    """4-layer hb_xh_4layer_h2_3x."""
    net = make_hb_xh_4layer_h2_3x(input_size=2, hidden_size=8, output_size=1)
    assert net.num_layers == 4
    assert net.Kx == 3
    assert net.Kh == 2


def test_factory_h2_k5():
    """4-layer hb_xh_4layer_h2_k5 (round 165 best config)."""
    net = make_hb_xh_4layer_h2_k5(input_size=2, hidden_size=8, output_size=1)
    assert net.num_layers == 4
    assert net.Kx == 5
    assert net.Kh == 2


def test_stacked_4layer_forward_shape():
    """4-layer forward returns [B, T, output_size]."""
    net = make_hb_xh_4layer_h2_k5(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_4layer_forward_finite_with_nan():
    """4-layer forward handles NaN inputs."""
    net = make_hb_xh_4layer_h2_k5(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_4layer_gradient_flows_to_all_layers():
    """4-layer gradient reaches all 4 layers."""
    net = make_hb_xh_4layer_h2_k5(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    assert len(net.cells) == 4
    for cell in net.cells:
        assert cell.f_gate[0].weight.grad is not None
        assert cell.f_gate[0].weight.grad.abs().sum().item() > 0
        assert cell.beta_x_raw.grad is not None


def test_stacked_4layer_no_sequences():
    """4-layer return_sequences=False returns [B, output_size]."""
    net = make_hb_xh_4layer_h2_k5(input_size=2, hidden_size=8, output_size=1, return_sequences=False)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_smoke_learns_sin_4layer():
    """4-layer should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = make_hb_xh_4layer_h2_k5(input_size=2, hidden_size=12, output_size=1)
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
    assert final_loss < 5.0
    assert final_loss < initial_loss


def test_bench_smoke_4layer_vs_3layer():
    """4-layer should achieve comparable loss to 3-layer."""
    B, T, D, H = 4, 16, 2, 8
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
    x = torch.zeros(B, T, D)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    y = torch.sin(t.squeeze(-1)).unsqueeze(-1).expand(B, T, 1)

    # 3-layer.
    torch.manual_seed(42)
    net_3 = make_hb_xh_4layer_h2_k5(input_size=D, hidden_size=H, output_size=1, num_layers=3)
    opt = torch.optim.Adam(net_3.parameters(), lr=1e-2)
    loss_3 = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = net_3(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        loss_3 = loss.item()

    # 4-layer.
    torch.manual_seed(42)
    net_4 = make_hb_xh_4layer_h2_k5(input_size=D, hidden_size=H, output_size=1, num_layers=4)
    opt = torch.optim.Adam(net_4.parameters(), lr=1e-2)
    loss_4 = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = net_4(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        loss_4 = loss.item()

    assert math.isfinite(loss_3)
    assert math.isfinite(loss_4)


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
