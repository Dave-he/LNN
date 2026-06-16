"""Tests for 4ScaleSpectralBiasDrop3CfC (Round 218, PRD #10-180)."""
from __future__ import annotations

import torch

from lnn.core.learned_beta_ps_ln_khlfft_4spectralbiasdrop3_cfc import (
    make_lbps_lnkhlfft_4spectralbiasdrop3_5_3_2,
)


def test_4spectralbiasdrop3_forward():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_4spectralbiasdrop3_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    assert y.shape == (4, 32, 1)
    assert torch.isfinite(y).all()


def test_4spectralbiasdrop3_no_nan():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_4spectralbiasdrop3_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    assert not torch.isnan(y).any()


def test_4spectralbiasdrop3_grads_flow():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_4spectralbiasdrop3_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(4, 32, 2)
    y = net(x)
    y.sum().backward()
    for name, p in net.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {name}"


def test_4spectralbiasdrop3_dropout_p_is_03():
    from lnn.core.learned_beta_ps_ln_khlfft_4spectralbiasdrop3_cfc import (
        make_lbps_lnkhlfft_4spectralbiasdrop3_5_3_2,
    )
    net = make_lbps_lnkhlfft_4spectralbiasdrop3_5_3_2(input_size=2, hidden_size=16, output_size=1)
    for cell in net.cells:
        assert cell.dropout_p == 0.3


def test_4spectralbiasdrop3_eval_deterministic():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_4spectralbiasdrop3_5_3_2(input_size=2, hidden_size=16, output_size=1)
    net.eval()
    x = torch.randn(4, 32, 2)
    with torch.no_grad():
        y1 = net(x)
        y2 = net(x)
    assert torch.allclose(y1, y2)


def test_4spectralbiasdrop3_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_4spectralbiasdrop3_5_3_2(input_size=2, hidden_size=16, output_size=1)
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


def test_4spectralbiasdrop3_long_seq():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_4spectralbiasdrop3_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(2, 64, 2)
    y = net(x)
    assert y.shape == (2, 64, 1)
    assert torch.isfinite(y).all()


def test_4spectralbiasdrop3_zero_input():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_4spectralbiasdrop3_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.zeros(2, 32, 2)
    y = net(x)
    assert y.shape == (2, 32, 1)
    assert torch.isfinite(y).all()


def test_4spectralbiasdrop3_higher_p_than_r216():
    from lnn.core.learned_beta_ps_ln_khlfft_4spectralbiasdrop_cfc import (
        make_lbps_lnkhlfft_4spectralbiasdrop_5_3_2,
    )
    from lnn.core.learned_beta_ps_ln_khlfft_4spectralbiasdrop3_cfc import (
        make_lbps_lnkhlfft_4spectralbiasdrop3_5_3_2,
    )
    net_216 = make_lbps_lnkhlfft_4spectralbiasdrop_5_3_2(input_size=2, hidden_size=16, output_size=1)
    net_218 = make_lbps_lnkhlfft_4spectralbiasdrop3_5_3_2(input_size=2, hidden_size=16, output_size=1)
    for c216, c218 in zip(net_216.cells, net_218.cells):
        assert c216.dropout_p < c218.dropout_p


def test_4spectralbiasdrop3_4_masks_4_biases():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_4spectralbiasdrop3_5_3_2(input_size=2, hidden_size=16, output_size=1)
    mask_names = {f"spec_mask{i}" for i in range(1, 5)}
    bias_names = {f"spec_bias{i}" for i in range(1, 5)}
    found_masks = set()
    found_biases = set()
    for cell in net.cells:
        for name in cell.state_dict():
            base = name.split(".")[0]
            if base in mask_names:
                found_masks.add(base)
            if base in bias_names:
                found_biases.add(base)
    assert len(found_masks) == 4
    assert len(found_biases) == 4
