"""Tests for PhaseFATCSpectralCfCStackedNetwork (Round 226, PRD #10-188)."""
from __future__ import annotations

import torch

from lnn.core.learned_beta_ps_ln_khlfft_phasefatc_cfc import (
    PhaseFATCSpectralCfCStackedNetwork,
    make_lbps_lnkhlfft_phasefatc_5_3_2,
)


def test_phasefatc_forward():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_phasefatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    assert y.shape == (4, 32, 1)
    assert torch.isfinite(y).all()


def test_phasefatc_no_nan():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_phasefatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    assert not torch.isnan(y).any()


def test_phasefatc_grads_flow():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_phasefatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    y.sum().backward()
    for name, p in net.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {name}"


def test_phasefatc_freq_to_scale_3x_input():
    """freq_to_scale should take 3x n_freq input (mag + cos + sin)."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_phasefatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    n_freq = 16 // 2 + 1  # 9
    expected_in = n_freq * 3  # 27
    for cell in net.cells:
        assert cell.freq_to_scale.in_features == expected_in
        assert cell.freq_to_scale.out_features == 16


def test_phasefatc_eval_deterministic():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_phasefatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    net.eval()
    x = torch.randn(4, 32, 2)
    with torch.no_grad():
        y1 = net(x)
        y2 = net(x)
    assert torch.allclose(y1, y2)


def test_phasefatc_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_phasefatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
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


def test_phasefatc_long_seq():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_phasefatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(2, 64, 2)
    y = net(x)
    assert y.shape == (2, 64, 1)
    assert torch.isfinite(y).all()


def test_phasefatc_zero_input():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_phasefatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.zeros(2, 32, 2)
    y = net(x)
    assert y.shape == (2, 32, 1)
    assert torch.isfinite(y).all()


def test_phasefatc_time_scale_positive():
    torch.manual_seed(0)
    cell = make_lbps_lnkhlfft_phasefatc_5_3_2(input_size=2, hidden_size=16, output_size=1).cells[0]
    h = torch.randn(4, 16)
    ts = cell._adaptive_time_scale(h)
    assert (ts > 0).all()
    assert torch.isfinite(ts).all()


def test_phasefatc_phase_shifts_produce_different_scale():
    """Different phase should produce different time_scale."""
    torch.manual_seed(0)
    cell = make_lbps_lnkhlfft_phasefatc_5_3_2(input_size=2, hidden_size=16, output_size=1).cells[0]
    # Same magnitude, different phase
    H1 = torch.tensor([1.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.complex64)
    H2 = torch.tensor([1.0, -0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.complex64)
    # h = irfft(H) - need 16 dim
    h1 = torch.fft.irfft(H1, n=16).unsqueeze(0).expand(4, -1)
    h2 = torch.fft.irfft(H2, n=16).unsqueeze(0).expand(4, -1)
    ts1 = cell._adaptive_time_scale(h1)
    ts2 = cell._adaptive_time_scale(h2)
    # Phase should change time_scale (since cos/sin differ)
    assert not torch.allclose(ts1, ts2, atol=1e-3)


def test_phasefatc_more_params_than_fatc():
    """PhaseFATC has 3x input features for freq_to_scale vs FATC."""
    from lnn.core.learned_beta_ps_ln_khlfft_fatc_cfc import make_lbps_lnkhlfft_fatc_5_3_2
    torch.manual_seed(0)
    net_r226 = make_lbps_lnkhlfft_phasefatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    net_r225 = make_lbps_lnkhlfft_fatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    p226 = sum(p.numel() for p in net_r226.parameters())
    p225 = sum(p.numel() for p in net_r225.parameters())
    # r226 has Linear(27, 16) instead of Linear(9, 16) -> +18*16 = 288 params per cell
    assert p226 > p225
    assert p226 - p225 < 1000, f"expected modest param increase, got {p226 - p225}"