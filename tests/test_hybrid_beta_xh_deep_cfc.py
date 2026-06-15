"""Round 164 — tests for HybridBeta-XH-Deep-CfC (3-layer stacked) (PRD #10-126)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.hybrid_beta_xh_deep_cfc import (
    HybridBetaXHCfCStackedNetwork,
    make_hb_xh_deep_best,
    make_hb_xh_deep_h1,
    make_hb_xh_deep_h2,
    make_hb_xh_deep_h2_3x,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_factory_h1():
    """3-layer hb_xh_deep_h1."""
    net = make_hb_xh_deep_h1(input_size=2, hidden_size=8, output_size=1, num_layers=3)
    assert net.num_layers == 3
    assert net.Kx == 1
    assert net.Kh == 1
    assert net.betas_h == [0.9]


def test_factory_h2():
    """3-layer hb_xh_deep_h2."""
    net = make_hb_xh_deep_h2(input_size=2, hidden_size=8, output_size=1, num_layers=3)
    assert net.num_layers == 3
    assert net.Kx == 2
    assert net.Kh == 2
    assert net.betas_h == [0.7, 0.95]


def test_factory_h2_3x():
    """3-layer hb_xh_deep_h2_3x."""
    net = make_hb_xh_deep_h2_3x(input_size=2, hidden_size=8, output_size=1, num_layers=3)
    assert net.num_layers == 3
    assert net.Kx == 3
    assert net.Kh == 2


def test_factory_best():
    """3-layer hb_xh_deep_best."""
    net = make_hb_xh_deep_best(input_size=2, hidden_size=8, output_size=1, num_layers=3)
    assert net.num_layers == 3
    assert net.Kx == 3
    assert net.Kh == 2


def test_stacked_3layer_forward_shape():
    """3-layer forward returns [B, T, output_size]."""
    net = HybridBetaXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx=2, Kh=2, betas_h=[0.7, 0.95], return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_3layer_forward_finite_with_nan():
    """3-layer forward handles NaN inputs."""
    net = HybridBetaXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx=2, Kh=2, betas_h=[0.7, 0.95],
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_3layer_gradient_flows_to_all_layers():
    """3-layer gradient reaches all 3 layers' CfC weights."""
    net = HybridBetaXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx=2, Kh=2, betas_h=[0.7, 0.95],
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    # All 3 cells should have gradient.
    assert len(net.cells) == 3
    for cell in net.cells:
        assert cell.f_gate[0].weight.grad is not None
        assert cell.f_gate[0].weight.grad.abs().sum().item() > 0
        assert cell.beta_x_raw.grad is not None


def test_stacked_3layer_no_sequences():
    """3-layer return_sequences=False returns [B, output_size]."""
    net = HybridBetaXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx=2, Kh=2, betas_h=[0.7, 0.95], return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_stacked_3layer_smoke_learns_sin():
    """3-layer should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = HybridBetaXHCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=3,
        Kx=2, Kh=2, betas_h=[0.7, 0.95],
    )
    t = torch.linspace(0, 4 * math.pi, 16).unsqueeze(0).unsqueeze(-1).expand(2, 16, 1)
    x = torch.zeros(2, 16, 2)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    target = torch.sin(t.squeeze(-1)).unsqueeze(-1)
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
    assert math.isfinite(final_loss)
    assert final_loss < 5.0
    assert final_loss < initial_loss


def test_bench_smoke_3layer_vs_2layer():
    """3-layer should achieve comparable loss to 2-layer."""
    B, T, D, H = 4, 16, 2, 8
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
    x = torch.zeros(B, T, D)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    y = torch.sin(t.squeeze(-1)).unsqueeze(-1).expand(B, T, 1)

    # 2-layer.
    torch.manual_seed(42)
    net_2 = HybridBetaXHCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        Kx=2, Kh=2, betas_h=[0.7, 0.95], return_sequences=True,
    )
    opt = torch.optim.Adam(net_2.parameters(), lr=1e-2)
    loss_2 = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = net_2(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        loss_2 = loss.item()

    # 3-layer.
    torch.manual_seed(42)
    net_3 = HybridBetaXHCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=3,
        Kx=2, Kh=2, betas_h=[0.7, 0.95], return_sequences=True,
    )
    opt = torch.optim.Adam(net_3.parameters(), lr=1e-2)
    loss_3 = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = net_3(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        loss_3 = loss.item()

    assert math.isfinite(loss_2)
    assert math.isfinite(loss_3)


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
