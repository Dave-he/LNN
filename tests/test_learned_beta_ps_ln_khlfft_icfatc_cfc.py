"""Tests for ICFATCSpectralCfCStackedNetwork (Round 228, PRD #10-190)."""
from __future__ import annotations

import torch

from lnn.core.learned_beta_ps_ln_khlfft_icfatc_cfc import (
    ICFATCSpectralCfCStackedNetwork,
    make_lbps_lnkhlfft_icfatc_5_3_2,
)


def test_icfatc_forward():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_icfatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    assert y.shape == (4, 32, 1)
    assert torch.isfinite(y).all()


def test_icfatc_no_nan():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_icfatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    assert not torch.isnan(y).any()


def test_icfatc_grads_flow():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_icfatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    y.sum().backward()
    for name, p in net.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {name}"


def test_icfatc_ic_mix_proj_per_sample():
    """ic_mix_proj should produce per-sample, per-feature mix."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_icfatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    for cell in net.cells:
        # Layer 0 has 4 (augmented input = 2*input_size=4)
        # Other layers have hidden_size=16
        assert cell.ic_mix_proj.out_features == 16


def test_icfatc_eval_deterministic():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_icfatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    net.eval()
    x = torch.randn(4, 32, 2)
    with torch.no_grad():
        y1 = net(x)
        y2 = net(x)
    assert torch.allclose(y1, y2)


def test_icfatc_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_icfatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
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


def test_icfatc_long_seq():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_icfatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(2, 64, 2)
    y = net(x)
    assert y.shape == (2, 64, 1)
    assert torch.isfinite(y).all()


def test_icfatc_zero_input():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_icfatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.zeros(2, 32, 2)
    y = net(x)
    assert y.shape == (2, 32, 1)
    assert torch.isfinite(y).all()


def test_icfatc_time_scale_positive():
    torch.manual_seed(0)
    cell = make_lbps_lnkhlfft_icfatc_5_3_2(input_size=2, hidden_size=16, output_size=1).cells[0]
    h = torch.randn(4, 16)
    x_t = torch.randn(4, 4)  # augmented input is 4
    ts = cell._adaptive_time_scale(h, x_t)
    assert (ts > 0).all()
    assert torch.isfinite(ts).all()


def test_icfatc_different_x_produces_different_mix():
    """Different input should produce different mix (input conditioning works)."""
    torch.manual_seed(0)
    cell = make_lbps_lnkhlfft_icfatc_5_3_2(input_size=2, hidden_size=16, output_size=1).cells[0]
    h = torch.randn(4, 16)
    x_t1 = torch.tensor([[1.0, 0.5, 0.0, 0.0]] * 4)
    x_t2 = torch.tensor([[-1.0, -0.5, 0.0, 0.0]] * 4)
    ts1 = cell._adaptive_time_scale(h, x_t1)
    ts2 = cell._adaptive_time_scale(h, x_t2)
    # Different input should produce different time_scale
    assert not torch.allclose(ts1, ts2, atol=1e-3)


def test_icfatc_x_with_zero_vs_random_h():
    """Mix depends on x, not just h. With same h, different x → different mix."""
    torch.manual_seed(0)
    cell = make_lbps_lnkhlfft_icfatc_5_3_2(input_size=2, hidden_size=16, output_size=1).cells[0]
    h = torch.randn(4, 16)  # same h
    x_zero = torch.zeros(4, 4)
    x_random = torch.randn(4, 4)
    mix_zero = torch.sigmoid(cell.ic_mix_proj(x_zero) + cell.ic_mix_bias)
    mix_random = torch.sigmoid(cell.ic_mix_proj(x_random) + cell.ic_mix_bias)
    assert not torch.allclose(mix_zero, mix_random, atol=1e-3)


def test_icfatc_param_count_more_than_fatc():
    """IC-FATC adds ic_mix_proj (input_size→hidden_size) per cell vs FATC scalar mix."""
    from lnn.core.learned_beta_ps_ln_khlfft_fatc_cfc import make_lbps_lnkhlfft_fatc_5_3_2
    torch.manual_seed(0)
    net_ic = make_lbps_lnkhlfft_icfatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    net_fatc = make_lbps_lnkhlfft_fatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    p_ic = sum(p.numel() for p in net_ic.parameters())
    p_fatc = sum(p.numel() for p in net_fatc.parameters())
    # IC-FATC: per cell adds Linear(input_size=2 or hidden_size=16, 16) + bias
    # Layer 0: 4*16+16=80; Layers 1,2: 16*16+16=272; total per 3 cells: 80+272+272=624
    assert p_ic > p_fatc
