"""Tests for PerLayerScaleCfCStackedNetwork (Round 223, PRD #10-185)."""
from __future__ import annotations

import torch

from lnn.core.learned_beta_ps_ln_khlfft_perlayer_scale_cfc import (
    PerLayerScaleCfCStackedNetwork,
    make_lbps_lnkhlfft_perlayer_234_5_3_2,
)


def test_perlayer_scale_forward():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_perlayer_234_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    assert y.shape == (4, 32, 1)
    assert torch.isfinite(y).all()


def test_perlayer_scale_no_nan():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_perlayer_234_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    assert not torch.isnan(y).any()


def test_perlayer_scale_grads_flow():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_perlayer_234_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    y.sum().backward()
    for name, p in net.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {name}"


def test_perlayer_scale_layer0_is_2scale_layer1_is_3scale_layer2_is_4scale():
    """Test that layer 0 has 2 scales, layer 1 has 3, layer 2 has 4."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_perlayer_234_5_3_2(input_size=2, hidden_size=16, output_size=1)
    cell0, cell1, cell2 = net.cells
    # Layer 0: 2 scales
    assert hasattr(cell0, "spec_mask1")
    assert hasattr(cell0, "spec_mask2")
    assert not hasattr(cell0, "spec_mask3")
    # Layer 1: 3 scales
    assert hasattr(cell1, "spec_mask1")
    assert hasattr(cell1, "spec_mask2")
    assert hasattr(cell1, "spec_mask3")
    assert not hasattr(cell1, "spec_mask4")
    # Layer 2: 4 scales
    assert hasattr(cell2, "spec_mask1")
    assert hasattr(cell2, "spec_mask2")
    assert hasattr(cell2, "spec_mask3")
    assert hasattr(cell2, "spec_mask4")


def test_perlayer_scale_eval_deterministic():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_perlayer_234_5_3_2(input_size=2, hidden_size=16, output_size=1)
    net.eval()
    x = torch.randn(4, 32, 2)
    with torch.no_grad():
        y1 = net(x)
        y2 = net(x)
    assert torch.allclose(y1, y2)


def test_perlayer_scale_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_perlayer_234_5_3_2(input_size=2, hidden_size=16, output_size=1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    t = torch.linspace(0, 4 * 3.14159, 32)
    x = torch.zeros(8, 32, 2)
    x[..., 0] = torch.sin(t).unsqueeze(0).expand(8, -1)
    x[..., 1] = torch.cos(t).unsqueeze(0).expand(8, -1)
    y = x[..., 0:1]
    for _ in range(20):
        opt.zero_grad()
        yp = net(x)
        loss = (yp - y).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    assert loss.item() < 0.1


def test_perlayer_scale_long_seq():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_perlayer_234_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(2, 64, 2)
    y = net(x)
    assert y.shape == (2, 64, 1)
    assert torch.isfinite(y).all()


def test_perlayer_scale_zero_input():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_perlayer_234_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.zeros(2, 32, 2)
    y = net(x)
    assert y.shape == (2, 32, 1)
    assert torch.isfinite(y).all()


def test_perlayer_scale_custom_scale_counts():
    """Test with custom scale counts."""
    torch.manual_seed(0)
    net = PerLayerScaleCfCStackedNetwork(
        input_size=2, hidden_size=16, output_size=1,
        num_layers=3, scale_counts=[4, 4, 4],
        Kh_ladder=[3, 3, 3], Kx=5,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)
    assert torch.isfinite(y).all()


def test_perlayer_scale_total_params_match_expectation():
    """Test that 3-layer network with [2,3,4] scales has expected params."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_perlayer_234_5_3_2(input_size=2, hidden_size=16, output_size=1)
    total_params = sum(p.numel() for p in net.parameters())
    # Should be > 1000 params (3 cells + FFT + head)
    assert total_params > 1000
