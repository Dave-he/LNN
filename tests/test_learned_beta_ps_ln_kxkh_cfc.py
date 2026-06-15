"""Round 181 — tests for LearnedBetaPS+LN+KxKh-CfC (PRD #10-143)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_kxkh_cfc import (
    LearnedBetaPSLNKxKhCfCStackedNetwork,
    make_lbps_ln_kxkh_5_5_5_2_5_2,
    make_lbps_ln_kxkh_5_5_5_5_3_2,
    make_lbps_ln_kxkh_3_5_7_2_5_2,
    make_lbps_ln_kxkh_7_5_3_5_3_2,
    make_lbps_ln_kxkh_3_5_7_5_3_2,
    make_lbps_ln_kxkh_7_5_3_2_5_2,
)


def test_init_default_ladders():
    net = LearnedBetaPSLNKxKhCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
    )
    assert net.Kx_ladder == [5, 5, 5]
    assert net.Kh_ladder == [3, 3, 3]


def test_init_custom_ladders():
    net = LearnedBetaPSLNKxKhCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx_ladder=[3, 5, 7], Kh_ladder=[2, 3, 5],
    )
    assert net.Kx_ladder == [3, 5, 7]
    assert net.Kh_ladder == [2, 3, 5]
    assert net.cells[0].Kx == 3
    assert net.cells[0].Kh == 2
    assert net.cells[1].Kx == 5
    assert net.cells[1].Kh == 3
    assert net.cells[2].Kx == 7
    assert net.cells[2].Kh == 5


def test_init_mismatch_raises():
    try:
        LearnedBetaPSLNKxKhCfCStackedNetwork(
            input_size=2, hidden_size=8, output_size=1, num_layers=3,
            Kx_ladder=[3, 5],
        )
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_factory_5_5_5_2_5_2():
    net = make_lbps_ln_kxkh_5_5_5_2_5_2(input_size=2, hidden_size=8, output_size=1)
    assert net.Kx_ladder == [5, 5, 5]
    assert net.Kh_ladder == [2, 5, 2]


def test_factory_3_5_7_2_5_2():
    net = make_lbps_ln_kxkh_3_5_7_2_5_2(input_size=2, hidden_size=8, output_size=1)
    assert net.Kx_ladder == [3, 5, 7]
    assert net.Kh_ladder == [2, 5, 2]


def test_forward_shape():
    net = make_lbps_ln_kxkh_3_5_7_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_forward_handles_nan():
    net = make_lbps_ln_kxkh_7_5_3_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_gradient_flows_all_layers():
    net = make_lbps_ln_kxkh_3_5_7_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.beta_x_raw.grad is not None
        assert cell.beta_h_raw.grad is not None
        assert cell.beta_x_raw.grad.abs().sum().item() > 0
        assert cell.beta_h_raw.grad.abs().sum().item() > 0


def test_layer_norm_per_layer():
    """Each layer has its own LayerNorm with correct aug dim."""
    net = make_lbps_ln_kxkh_3_5_7_5_3_2(input_size=2, hidden_size=8, output_size=1)
    # Layer 0: Kx=3, Kh=5, in_size=2, H=8 → aug = (3+1)*2 + (5+1)*8 = 8 + 48 = 56
    # Layer 1: Kx=5, Kh=3, in_size=8, H=8 → aug = (5+1)*8 + (3+1)*8 = 48 + 32 = 80
    # Layer 2: Kx=7, Kh=2, in_size=8, H=8 → aug = (7+1)*8 + (2+1)*8 = 64 + 24 = 88
    assert net.cells[0].layer_norm.normalized_shape == (56,)
    assert net.cells[1].layer_norm.normalized_shape == (80,)
    assert net.cells[2].layer_norm.normalized_shape == (88,)


def test_no_sequences():
    net = LearnedBetaPSLNKxKhCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx_ladder=[3, 5, 7], Kh_ladder=[2, 3, 5], return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_ln_kxkh_5_5_5_2_5_2(input_size=2, hidden_size=12, output_size=1)
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
