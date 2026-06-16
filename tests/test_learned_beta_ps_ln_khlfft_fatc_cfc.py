"""Tests for FATCSpectralCfCStackedNetwork (Round 225, PRD #10-187)."""
from __future__ import annotations

import torch

from lnn.core.learned_beta_ps_ln_khlfft_fatc_cfc import (
    FATCSpectralCfCStackedNetwork,
    make_lbps_lnkhlfft_fatc_5_3_2,
)


def test_fatc_forward():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_fatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    assert y.shape == (4, 32, 1)
    assert torch.isfinite(y).all()


def test_fatc_no_nan():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_fatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    assert not torch.isnan(y).any()


def test_fatc_grads_flow():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_fatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    y.sum().backward()
    for name, p in net.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {name}"


def test_fatc_has_freq_to_scale_and_mix():
    """Each cell has freq_to_scale and fatc_mix params."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_fatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    for cell in net.cells:
        assert hasattr(cell, "freq_to_scale")
        assert hasattr(cell, "fatc_mix")
        assert hasattr(cell, "time_scale")


def test_fatc_eval_deterministic():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_fatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    net.eval()
    x = torch.randn(4, 32, 2)
    with torch.no_grad():
        y1 = net(x)
        y2 = net(x)
    assert torch.allclose(y1, y2)


def test_fatc_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_fatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
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


def test_fatc_long_seq():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_fatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(2, 64, 2)
    y = net(x)
    assert y.shape == (2, 64, 1)
    assert torch.isfinite(y).all()


def test_fatc_zero_input():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_fatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.zeros(2, 32, 2)
    y = net(x)
    assert y.shape == (2, 32, 1)
    assert torch.isfinite(y).all()


def test_fatc_time_scale_positive():
    """FATC mix should give positive time_scale."""
    torch.manual_seed(0)
    cell = make_lbps_lnkhlfft_fatc_5_3_2(input_size=2, hidden_size=16, output_size=1).cells[0]
    h = torch.randn(4, 16)
    ts = cell._adaptive_time_scale(h)
    assert (ts > 0).all()
    assert torch.isfinite(ts).all()


def test_fatc_freq_changes_time_scale():
    """Different frequency input → different time_scale."""
    torch.manual_seed(0)
    cell = make_lbps_lnkhlfft_fatc_5_3_2(input_size=2, hidden_size=16, output_size=1).cells[0]
    # Low-frequency signal (slow sinusoid)
    t = torch.linspace(0, 2 * 3.14159, 16)
    h_low = torch.sin(t * 0.5).unsqueeze(0).expand(4, -1)
    # High-frequency signal
    h_high = torch.sin(t * 5.0).unsqueeze(0).expand(4, -1)
    ts_low = cell._adaptive_time_scale(h_low)
    ts_high = cell._adaptive_time_scale(h_high)
    # Time scales should differ
    assert not torch.allclose(ts_low, ts_high, atol=1e-3)


def test_fatc_total_params_similar_to_r216():
    """FATC adds few params over r216 (just freq_to_scale + fatc_mix per cell)."""
    from lnn.core.learned_beta_ps_ln_khlfft_4spectralbiasdrop_cfc import make_lbps_lnkhlfft_4spectralbiasdrop_5_3_2
    torch.manual_seed(0)
    net_r225 = make_lbps_lnkhlfft_fatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    net_r216 = make_lbps_lnkhlfft_4spectralbiasdrop_5_3_2(input_size=2, hidden_size=16, output_size=1)
    p225 = sum(p.numel() for p in net_r225.parameters())
    p216 = sum(p.numel() for p in net_r216.parameters())
    # r225 should have a few more params due to freq_to_scale (16*9=144) and fatc_mix (1)
    assert p225 > p216
    assert p225 - p216 < 500, f"expected small param increase, got {p225 - p216}"