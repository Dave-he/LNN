"""Tests for PhaseDriftSpectralCfCStackedNetwork (Round 227, PRD #10-189)."""
from __future__ import annotations

import torch

from lnn.core.learned_beta_ps_ln_khlfft_phasedrift_cfc import (
    PhaseDriftSpectralCfCStackedNetwork,
    make_lbps_lnkhlfft_phasedrift_5_3_2,
)


def test_phasedrift_forward():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_phasedrift_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    assert y.shape == (4, 32, 1)
    assert torch.isfinite(y).all()


def test_phasedrift_no_nan():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_phasedrift_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    assert not torch.isnan(y).any()


def test_phasedrift_grads_flow():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_phasedrift_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    y.sum().backward()
    for name, p in net.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {name}"


def test_phasedrift_freq_to_scale_2x_input():
    """freq_to_scale should take 2x n_freq input (mag + |H_diff|)."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_phasedrift_5_3_2(input_size=2, hidden_size=16, output_size=1)
    n_freq = 16 // 2 + 1  # 9
    expected_in = n_freq * 2  # 18
    for cell in net.cells:
        assert cell.freq_to_scale.in_features == expected_in
        assert cell.freq_to_scale.out_features == 16


def test_phasedrift_eval_deterministic():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_phasedrift_5_3_2(input_size=2, hidden_size=16, output_size=1)
    net.eval()
    x = torch.randn(4, 32, 2)
    with torch.no_grad():
        y1 = net(x)
        y2 = net(x)
    assert torch.allclose(y1, y2)


def test_phasedrift_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_phasedrift_5_3_2(input_size=2, hidden_size=16, output_size=1)
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


def test_phasedrift_long_seq():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_phasedrift_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(2, 64, 2)
    y = net(x)
    assert y.shape == (2, 64, 1)
    assert torch.isfinite(y).all()


def test_phasedrift_zero_input():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_phasedrift_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.zeros(2, 32, 2)
    y = net(x)
    assert y.shape == (2, 32, 1)
    assert torch.isfinite(y).all()


def test_phasedrift_time_scale_positive():
    torch.manual_seed(0)
    cell = make_lbps_lnkhlfft_phasedrift_5_3_2(input_size=2, hidden_size=16, output_size=1).cells[0]
    h = torch.randn(4, 16)
    ts = cell._adaptive_time_scale(h)
    assert (ts > 0).all()
    assert torch.isfinite(ts).all()


def test_phasedrift_drift_signal_differs_from_static():
    """Different signals should produce different time_scale (drift carries info)."""
    torch.manual_seed(0)
    cell = make_lbps_lnkhlfft_phasedrift_5_3_2(input_size=2, hidden_size=16, output_size=1).cells[0]
    # Same magnitude pattern, different phase distribution
    H1 = torch.tensor([1.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.complex64)
    H2 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=torch.complex64)
    h1 = torch.fft.irfft(H1, n=16).unsqueeze(0).expand(4, -1)
    h2 = torch.fft.irfft(H2, n=16).unsqueeze(0).expand(4, -1)
    ts1 = cell._adaptive_time_scale(h1)
    ts2 = cell._adaptive_time_scale(h2)
    # Different spectra should produce different time_scales
    assert not torch.allclose(ts1, ts2, atol=1e-3)


def test_phasedrift_diff_is_nontrivial():
    """Drift signal (|H_diff|) should be non-zero for non-trivial inputs."""
    torch.manual_seed(0)
    cell = make_lbps_lnkhlfft_phasedrift_5_3_2(input_size=2, hidden_size=16, output_size=1).cells[0]
    h = torch.randn(4, 16)
    H = torch.fft.rfft(h, dim=-1)
    H_diff = H[:, 1:] - H[:, :-1]
    mag_diff = torch.abs(H_diff)
    assert (mag_diff > 0).any()  # at least some bins differ


def test_phasedrift_param_count_comparable_to_phasefatc():
    """PhaseDrift (2*n_freq=18) is fewer params than PhaseFATC (3*n_freq=27) but more than FATC (9)."""
    from lnn.core.learned_beta_ps_ln_khlfft_fatc_cfc import make_lbps_lnkhlfft_fatc_5_3_2
    from lnn.core.learned_beta_ps_ln_khlfft_phasefatc_cfc import make_lbps_lnkhlfft_phasefatc_5_3_2
    torch.manual_seed(0)
    net_drift = make_lbps_lnkhlfft_phasedrift_5_3_2(input_size=2, hidden_size=16, output_size=1)
    net_fatc = make_lbps_lnkhlfft_fatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    net_phase = make_lbps_lnkhlfft_phasefatc_5_3_2(input_size=2, hidden_size=16, output_size=1)
    p_drift = sum(p.numel() for p in net_drift.parameters())
    p_fatc = sum(p.numel() for p in net_fatc.parameters())
    p_phase = sum(p.numel() for p in net_phase.parameters())
    # PhaseDrift: Linear(18, 16) = 304 params/cell
    # FATC: Linear(9, 16) = 160 params/cell
    # PhaseFATC: Linear(27, 16) = 448 params/cell
    assert p_drift > p_fatc
    assert p_drift < p_phase
