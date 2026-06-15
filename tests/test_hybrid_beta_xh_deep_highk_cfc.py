"""Round 165 — tests for HybridBeta-XH-Deep-HighK-CfC (3-layer + high K) (PRD #10-127)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.hybrid_beta_xh_deep_highk_cfc import (
    HybridBetaXHCfCStackedNetwork,
    make_hb_xh_deep_h1_k4,
    make_hb_xh_deep_h1_k5,
    make_hb_xh_deep_h2_k4,
    make_hb_xh_deep_h2_k5,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_factory_h1_k4():
    """3-layer, Kx=4, Kh=1."""
    net = make_hb_xh_deep_h1_k4(input_size=2, hidden_size=8, output_size=1)
    assert net.num_layers == 3
    assert net.Kx == 4
    assert net.Kh == 1
    assert net.betas_h == [0.9]


def test_factory_h1_k5():
    """3-layer, Kx=5, Kh=1."""
    net = make_hb_xh_deep_h1_k5(input_size=2, hidden_size=8, output_size=1)
    assert net.num_layers == 3
    assert net.Kx == 5
    assert net.Kh == 1
    assert net.betas_h == [0.9]


def test_factory_h2_k4():
    """3-layer, Kx=4, Kh=2."""
    net = make_hb_xh_deep_h2_k4(input_size=2, hidden_size=8, output_size=1)
    assert net.num_layers == 3
    assert net.Kx == 4
    assert net.Kh == 2
    assert net.betas_h == [0.7, 0.95]


def test_factory_h2_k5():
    """3-layer, Kx=5, Kh=2."""
    net = make_hb_xh_deep_h2_k5(input_size=2, hidden_size=8, output_size=1)
    assert net.num_layers == 3
    assert net.Kx == 5
    assert net.Kh == 2
    assert net.betas_h == [0.7, 0.95]


def test_stacked_3layer_k4_forward_shape():
    """3-layer K=4 forward returns [B, T, output_size]."""
    net = make_hb_xh_deep_h2_k4(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_3layer_k5_forward_shape():
    """3-layer K=5 forward returns [B, T, output_size]."""
    net = make_hb_xh_deep_h2_k5(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_3layer_k4_forward_finite_with_nan():
    """3-layer K=4 forward handles NaN inputs."""
    net = make_hb_xh_deep_h1_k4(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_3layer_k5_forward_finite_with_nan():
    """3-layer K=5 forward handles NaN inputs."""
    net = make_hb_xh_deep_h1_k5(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_3layer_k5_gradient_flows_to_all_layers():
    """3-layer K=5 gradient reaches all 3 layers."""
    net = make_hb_xh_deep_h1_k5(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    assert len(net.cells) == 3
    for cell in net.cells:
        assert cell.f_gate[0].weight.grad is not None
        assert cell.f_gate[0].weight.grad.abs().sum().item() > 0
        assert cell.beta_x_raw.grad is not None


def test_smoke_learns_sin_k4():
    """3-layer K=4 should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = make_hb_xh_deep_h1_k4(input_size=2, hidden_size=12, output_size=1)
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
