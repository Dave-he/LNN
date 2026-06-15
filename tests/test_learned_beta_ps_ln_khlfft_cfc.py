"""Round 187 — tests for LearnedBetaPS+LN+Khl+FFT-CfC (PRD #10-149)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_cfc import (
    LearnedBetaPSLNKhlFftCfCStackedNetwork,
    make_lbps_lnkhlfft_2_5_2,
    make_lbps_lnkhlfft_5_3_2,
)


def test_cell_init():
    net = make_lbps_lnkhlfft_2_5_2(input_size=2, hidden_size=8, output_size=1)
    assert net.input_size == 2
    assert net.augmented_input_size == 4
    assert net.Kh_ladder == [2, 5, 2]


def test_factory_2_5_2():
    net = make_lbps_lnkhlfft_2_5_2(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh_ladder == [2, 5, 2]


def test_factory_5_3_2():
    net = make_lbps_lnkhlfft_5_3_2(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh_ladder == [5, 3, 2]


def test_forward_shape_stacked():
    net = make_lbps_lnkhlfft_2_5_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_forward_handles_nan_stacked():
    net = make_lbps_lnkhlfft_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_fft_then_ladder():
    """Verify FFT encoder feeds into ladder correctly."""
    net = make_lbps_lnkhlfft_2_5_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    # FFT encoder should produce 2*input_size=4 features per timestep
    x_aug = net.fft_encoder(x)
    assert x_aug.shape == (2, 16, 4)
    # First 2 are original, last 2 are FFT magnitude
    assert torch.allclose(x_aug[..., :2], x)


def test_gradient_flows():
    """Verify gradient flows to all params after a few training steps."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_2_5_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    target = torch.randn(2, 16, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        y = net(x)
        loss = F.mse_loss(y, target)
        loss.backward()
        opt.step()
    opt.zero_grad()
    y = net(x)
    loss = F.mse_loss(y, target)
    loss.backward()
    for cell in net.cfc_net.cells:
        assert cell.beta_x_raw.grad is not None
        assert cell.beta_h_raw.grad is not None
        assert cell.layer_norm.weight.grad is not None
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in net.parameters()
    )
    assert has_grad, "no gradient flowed to any parameter"


def test_no_sequences():
    net = LearnedBetaPSLNKhlFftCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx=5, Kh_ladder=[2, 5, 2], return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_2_5_2(input_size=2, hidden_size=12, output_size=1)
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


def test_different_ladders_different_cells():
    """Verify 2_5_2 and 5_3_2 produce different model structures."""
    net_a = make_lbps_lnkhlfft_2_5_2(input_size=2, hidden_size=8, output_size=1)
    net_b = make_lbps_lnkhlfft_5_3_2(input_size=2, hidden_size=8, output_size=1)
    assert net_a.Kh_ladder != net_b.Kh_ladder
    # At least one cell should have different Kh (cells 0 and 1 differ; cell 2 ties)
    diffs = [net_a.cfc_net.cells[i].Kh != net_b.cfc_net.cells[i].Kh for i in range(3)]
    assert sum(diffs) >= 2, f"expected at least 2 different Kh values, got {diffs}"
    # Specifically, cells 0 and 1 should differ
    assert net_a.cfc_net.cells[0].Kh != net_b.cfc_net.cells[0].Kh
    assert net_a.cfc_net.cells[1].Kh != net_b.cfc_net.cells[1].Kh


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
