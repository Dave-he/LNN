"""Tests for TwoScaleSpectralBiasDropCfCCell (Round 222, PRD #10-184)."""
from __future__ import annotations

import torch

from lnn.core.learned_beta_ps_ln_khlfft_2spectralbiasdrop_cfc import (
    TwoScaleSpectralBiasDropCfCCell,
    make_lbps_lnkhlfft_2spectralbiasdrop_5_3_2,
)


def test_2spectralbiasdrop_forward():
    torch.manual_seed(0)
    cell = TwoScaleSpectralBiasDropCfCCell(input_size=2, hidden_size=16, Kx=5, Kh=3)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    emas_x = [torch.zeros(4, 2) for _ in range(5)]
    emas_h = [torch.zeros(4, 16) for _ in range(3)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    assert h_new.shape == (4, 16)
    assert torch.isfinite(h_new).all()


def test_2spectralbiasdrop_no_nan():
    torch.manual_seed(0)
    cell = TwoScaleSpectralBiasDropCfCCell(input_size=2, hidden_size=16, Kx=5, Kh=3)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    emas_x = [torch.zeros(4, 2) for _ in range(5)]
    emas_h = [torch.zeros(4, 16) for _ in range(3)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    assert not torch.isnan(h_new).any()


def test_2spectralbiasdrop_grads_flow():
    torch.manual_seed(0)
    cell = TwoScaleSpectralBiasDropCfCCell(input_size=2, hidden_size=16, Kx=5, Kh=3)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    emas_x = [torch.zeros(4, 2) for _ in range(5)]
    emas_h = [torch.zeros(4, 16) for _ in range(3)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    h_new.sum().backward()
    for name, p in cell.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {name}"


def test_2spectralbiasdrop_2_masks_2_biases_have_grad():
    torch.manual_seed(0)
    cell = TwoScaleSpectralBiasDropCfCCell(input_size=2, hidden_size=16, Kx=5, Kh=3)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    emas_x = [torch.zeros(4, 2) for _ in range(5)]
    emas_h = [torch.zeros(4, 16) for _ in range(3)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    h_new.sum().backward()
    mask_names = {f"spec_mask{i}" for i in range(1, 3)}
    bias_names = {f"spec_bias{i}" for i in range(1, 3)}
    found_masks = {name.split(".")[0] for name, p in cell.named_parameters() if name.split(".")[0] in mask_names}
    found_biases = {name.split(".")[0] for name, p in cell.named_parameters() if name.split(".")[0] in bias_names}
    assert found_masks == mask_names
    assert found_biases == bias_names
    for name, p in cell.named_parameters():
        if name.split(".")[0] in mask_names | bias_names:
            assert p.grad is not None, f"no grad for {name}"


def test_2spectralbiasdrop_stacked_network():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_2spectralbiasdrop_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    assert y.shape == (4, 32, 1)
    assert torch.isfinite(y).all()


def test_2spectralbiasdrop_eval_deterministic():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_2spectralbiasdrop_5_3_2(input_size=2, hidden_size=16, output_size=1)
    net.eval()
    x = torch.randn(4, 32, 2)
    with torch.no_grad():
        y1 = net(x)
        y2 = net(x)
    assert torch.allclose(y1, y2)


def test_2spectralbiasdrop_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_2spectralbiasdrop_5_3_2(input_size=2, hidden_size=16, output_size=1)
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


def test_2spectralbiasdrop_long_seq():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_2spectralbiasdrop_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(2, 64, 2)
    y = net(x)
    assert y.shape == (2, 64, 1)
    assert torch.isfinite(y).all()


def test_2spectralbiasdrop_zero_input():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_2spectralbiasdrop_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.zeros(2, 32, 2)
    y = net(x)
    assert y.shape == (2, 32, 1)
    assert torch.isfinite(y).all()


def test_2spectralbiasdrop_2_scales_only():
    """Test that 2-scale variant has exactly 2 scales (not 3 or 4)."""
    torch.manual_seed(0)
    cell = TwoScaleSpectralBiasDropCfCCell(input_size=2, hidden_size=16, Kx=5, Kh=3)
    has_mask3 = any("spec_mask3" in name for name, _ in cell.named_parameters())
    has_mask4 = any("spec_mask4" in name for name, _ in cell.named_parameters())
    has_mask5 = any("spec_mask5" in name for name, _ in cell.named_parameters())
    assert not has_mask3
    assert not has_mask4
    assert not has_mask5
