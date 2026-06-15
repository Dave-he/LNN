"""Round 168 — tests for LayerDecay-H3-CfC (Kh=3 with REVERSE β) (PRD #10-130)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.layer_decay_h3_cfc import (
    make_ld_reverse_h3_k5,
    make_ld_reverse_h4_k5,
    make_ld_reverse_h3_wider,
    make_ld_reverse_h3_k6,
    make_ld_reverse_h3_h2,
    make_ld_constant_h3,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_factory_reverse_h3():
    """ld_reverse_h3_k5: 3-layer, Kx=5, Kh=3, REVERSE β ∈ [0.99, 0.75, 0.5]."""
    net = make_ld_reverse_h3_k5(input_size=2, hidden_size=8, output_size=1)
    assert net.num_layers == 3
    assert net.Kx == 5
    assert net.Kh == 3
    assert net.betas_h == [0.99, 0.75, 0.5]
    assert net.mode == "reverse"
    # Schedule should differ across layers.
    assert net.layer_betas_h[0][0] > net.layer_betas_h[1][0] > net.layer_betas_h[2][0]


def test_factory_reverse_h4():
    """ld_reverse_h4_k5: 3-layer, Kx=5, Kh=4, REVERSE β."""
    net = make_ld_reverse_h4_k5(input_size=2, hidden_size=8, output_size=1)
    assert net.num_layers == 3
    assert net.Kx == 5
    assert net.Kh == 4
    assert net.mode == "reverse"


def test_factory_reverse_h3_wider():
    """ld_reverse_h3_wider: 3-layer, Kx=5, Kh=3, REVERSE β ∈ [0.999, 0.7, 0.3]."""
    net = make_ld_reverse_h3_wider(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 3
    # Layer 0 should have β ≈ 0.999 (max), layer 2 should have β ≈ 0.3 (min).
    assert abs(net.layer_betas_h[0][0] - 0.999) < 1e-6
    assert abs(net.layer_betas_h[-1][0] - 0.3) < 1e-6


def test_factory_reverse_h3_k6():
    """ld_reverse_h3_k6: 3-layer, Kx=6, Kh=3, REVERSE β."""
    net = make_ld_reverse_h3_k6(input_size=2, hidden_size=8, output_size=1)
    assert net.Kx == 6
    assert net.Kh == 3


def test_factory_reverse_h3_h2():
    """ld_reverse_h3_h2: 3-layer, Kx=5, Kh=3, REVERSE β ∈ [0.95, 0.85, 0.7] (narrow range)."""
    net = make_ld_reverse_h3_h2(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 3
    # Narrower range than h3_k5.
    assert abs(net.layer_betas_h[0][0] - 0.95) < 1e-6


def test_factory_constant_h3():
    """ld_constant_h3: 3-layer, Kx=5, Kh=3, constant β ∈ {0.7, 0.85, 0.95}."""
    net = make_ld_constant_h3(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 3
    assert net.mode == "constant"
    # Constant means all layers have same betas.
    assert net.layer_betas_h[0] == net.layer_betas_h[1] == net.layer_betas_h[2]


def test_forward_shape():
    """Network forward returns [B, T, output_size]."""
    net = make_ld_reverse_h3_k5(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_forward_handles_nan():
    """Network forward handles NaN inputs via nan_to_num."""
    net = make_ld_reverse_h3_k5(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_gradient_flows_all_layers():
    """Gradient reaches all 3 layers + per-feature β."""
    net = make_ld_reverse_h3_k5(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.beta_x_raw.grad is not None


def test_smoke_learns_sin():
    """ld_reverse_h3_k5 should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = make_ld_reverse_h3_k5(input_size=2, hidden_size=12, output_size=1)
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
