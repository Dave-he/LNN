"""Tests for AdaptiveScaleSpectralCfCStackedNetwork (Round 224, PRD #10-186)."""
from __future__ import annotations

import torch

from lnn.core.learned_beta_ps_ln_khlfft_adaptive_scale_cfc import (
    AdaptiveScaleSpectralCfCStackedNetwork,
    make_lbps_lnkhlfft_adaptive_scale_5_3_2,
)


def test_adaptive_scale_forward():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_adaptive_scale_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    assert y.shape == (4, 32, 1)
    assert torch.isfinite(y).all()


def test_adaptive_scale_no_nan():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_adaptive_scale_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    assert not torch.isnan(y).any()


def test_adaptive_scale_grads_flow():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_adaptive_scale_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    y.sum().backward()
    for name, p in net.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {name}"


def test_adaptive_scale_3_branches_per_cell():
    """Each cell has 3 scale branches (a/b/c = 2/3/4)."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_adaptive_scale_5_3_2(input_size=2, hidden_size=16, output_size=1)
    for cell in net.cells:
        # Branch a: 2 scales (mask1, mask2)
        assert hasattr(cell, "spec_mask1_a")
        assert hasattr(cell, "spec_mask2_a")
        assert not hasattr(cell, "spec_mask3_a")
        # Branch b: 3 scales (mask1, mask2, mask3)
        assert hasattr(cell, "spec_mask1_b")
        assert hasattr(cell, "spec_mask2_b")
        assert hasattr(cell, "spec_mask3_b")
        assert not hasattr(cell, "spec_mask4_b")
        # Branch c: 4 scales (mask1, mask2, mask3, mask4)
        assert hasattr(cell, "spec_mask1_c")
        assert hasattr(cell, "spec_mask2_c")
        assert hasattr(cell, "spec_mask3_c")
        assert hasattr(cell, "spec_mask4_c")
        # Gate
        assert hasattr(cell, "scale_router")


def test_adaptive_scale_gate_softmax_3way():
    """Cell returns weights for 3 branches summing to 1."""
    torch.manual_seed(0)
    cell = make_lbps_lnkhlfft_adaptive_scale_5_3_2(input_size=2, hidden_size=16, output_size=1).cells[0]
    x_t = torch.randn(4, 4)  # 4 = augmented input size
    h_t = torch.zeros(4, 16)
    emas_x = [torch.zeros(4, 4) for _ in range(5)]
    emas_h = [torch.zeros(4, 16) for _ in range(5)]
    h_new, _, _, weights = cell(x_t, h_t, emas_x, emas_h)
    assert weights.shape == (4, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_adaptive_scale_eval_deterministic():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_adaptive_scale_5_3_2(input_size=2, hidden_size=16, output_size=1)
    net.eval()
    x = torch.randn(4, 32, 2)
    with torch.no_grad():
        y1 = net(x)
        y2 = net(x)
    assert torch.allclose(y1, y2)


def test_adaptive_scale_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_adaptive_scale_5_3_2(input_size=2, hidden_size=16, output_size=1)
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


def test_adaptive_scale_long_seq():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_adaptive_scale_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(2, 64, 2)
    y = net(x)
    assert y.shape == (2, 64, 1)
    assert torch.isfinite(y).all()


def test_adaptive_scale_zero_input():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_adaptive_scale_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.zeros(2, 32, 2)
    y = net(x)
    assert y.shape == (2, 32, 1)
    assert torch.isfinite(y).all()


def test_adaptive_scale_total_params_larger_than_r223():
    """3-branch adaptive should have more params than fixed per-layer."""
    from lnn.core.learned_beta_ps_ln_khlfft_perlayer_scale_cfc import make_lbps_lnkhlfft_perlayer_234_5_3_2
    torch.manual_seed(0)
    net_r224 = make_lbps_lnkhlfft_adaptive_scale_5_3_2(input_size=2, hidden_size=16, output_size=1)
    net_r223 = make_lbps_lnkhlfft_perlayer_234_5_3_2(input_size=2, hidden_size=16, output_size=1)
    p224 = sum(p.numel() for p in net_r224.parameters())
    p223 = sum(p.numel() for p in net_r223.parameters())
    assert p224 > p223, f"r224={p224} should have more params than r223={p223}"