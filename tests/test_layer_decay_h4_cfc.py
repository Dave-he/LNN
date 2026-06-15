"""Round 169 — tests for LayerDecay-H4-CfC (Kh=4 with constant β) (PRD #10-131)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.layer_decay_h4_cfc import (
    make_ld_constant_h4_default,
    make_ld_constant_h4_wide,
    make_ld_constant_h4_narrow,
    make_ld_constant_h3_k6,
    make_ld_constant_h3_wider,
    make_ld_constant_h3_finer,
    make_ld_constant_h5,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_factory_h4_default():
    """ld_constant_h4_default: 3-layer, Kx=5, Kh=4."""
    net = make_ld_constant_h4_default(input_size=2, hidden_size=8, output_size=1)
    assert net.num_layers == 3
    assert net.Kx == 5
    assert net.Kh == 4
    assert net.betas_h == [0.6, 0.75, 0.85, 0.95]


def test_factory_h4_wide():
    """ld_constant_h4_wide: 3-layer, Kx=5, Kh=4, wider β range."""
    net = make_ld_constant_h4_wide(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 4
    assert net.betas_h == [0.5, 0.7, 0.85, 0.99]


def test_factory_h4_narrow():
    """ld_constant_h4_narrow: 3-layer, Kx=5, Kh=4, narrow β range."""
    net = make_ld_constant_h4_narrow(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 4
    assert net.betas_h == [0.8, 0.85, 0.9, 0.95]


def test_factory_h3_k6():
    """ld_constant_h3_k6: 3-layer, Kx=6, Kh=3."""
    net = make_ld_constant_h3_k6(input_size=2, hidden_size=8, output_size=1)
    assert net.Kx == 6
    assert net.Kh == 3


def test_factory_h3_wider():
    """ld_constant_h3_wider: 3-layer, Kx=5, Kh=3, β ∈ {0.6, 0.8, 0.99}."""
    net = make_ld_constant_h3_wider(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 3
    assert net.betas_h == [0.6, 0.8, 0.99]


def test_factory_h3_finer():
    """ld_constant_h3_finer: 3-layer, Kx=5, Kh=3, β ∈ {0.75, 0.85, 0.95}."""
    net = make_ld_constant_h3_finer(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 3
    assert net.betas_h == [0.75, 0.85, 0.95]


def test_factory_h5():
    """ld_constant_h5: 3-layer, Kx=5, Kh=5."""
    net = make_ld_constant_h5(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 5
    assert net.betas_h == [0.5, 0.7, 0.85, 0.95, 0.99]


def test_forward_shape():
    """Network forward returns [B, T, output_size]."""
    net = make_ld_constant_h4_default(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_forward_handles_nan():
    """Network forward handles NaN inputs via nan_to_num."""
    net = make_ld_constant_h4_default(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_gradient_flows_all_layers():
    """Gradient reaches all 3 layers."""
    net = make_ld_constant_h4_default(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.beta_x_raw.grad is not None


def test_smoke_learns_sin():
    """ld_constant_h4_default should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = make_ld_constant_h4_default(input_size=2, hidden_size=12, output_size=1)
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
