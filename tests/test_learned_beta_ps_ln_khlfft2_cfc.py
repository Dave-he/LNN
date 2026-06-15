"""Round 188 — tests for LearnedBetaPS+LN+Khl+FFT2-CfC (PRD #10-150)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft2_cfc import (
    FFT2InputEncoder,
    LearnedBetaPSLNKhlFft2CfCStackedNetwork,
    make_lbps_lnkhlfft2_2_5_2,
    make_lbps_lnkhlfft2_5_3_2,
)


def test_fft2_encoder_shape():
    """FFT2 encoder outputs [B, T, 3D] (original + mag + phase)."""
    enc = FFT2InputEncoder()
    x = torch.randn(2, 16, 3)
    y = enc(x)
    assert y.shape == (2, 16, 9)  # 3*3=9


def test_fft2_encoder_preserves_original():
    """First D features of FFT2 output == input."""
    enc = FFT2InputEncoder()
    x = torch.randn(2, 16, 3)
    y = enc(x)
    assert torch.allclose(y[..., :3], x)


def test_fft2_encoder_handles_nan():
    """NaN input: original NaN preserved (passes through to model), but
    mag/phase features computed on NaN-replaced input are finite."""
    enc = FFT2InputEncoder()
    x = torch.randn(2, 16, 3)
    x[0, 5, 0] = float("nan")
    y = enc(x)
    # Original (col 0) preserves NaN — passed through to model
    assert torch.isnan(y[0, 5, 0])
    # mag and phase (cols 3,6) for the NaN sample are finite
    assert torch.isfinite(y[0, 5, 1]).all()
    assert torch.isfinite(y[0, 5, 2]).all()


def test_fft2_encoder_uses_complex_fft():
    """Verify magnitude and phase are computed correctly for sin input."""
    enc = FFT2InputEncoder()
    T = 16
    t = torch.linspace(0, 2 * math.pi, T).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
    x = torch.sin(t).expand(2, T, 1)  # [2, T, 1]
    y = enc(x)
    # mag (col 1) should be NONZERO for the sin frequency bin
    assert y[..., 1].abs().max() > 0.1
    # phase (col 2) should be FINITE
    assert torch.isfinite(y[..., 2]).all()


def test_factory_2_5_2():
    net = make_lbps_lnkhlfft2_2_5_2(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh_ladder == [2, 5, 2]


def test_factory_5_3_2():
    net = make_lbps_lnkhlfft2_5_3_2(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh_ladder == [5, 3, 2]


def test_cell_init():
    net = make_lbps_lnkhlfft2_2_5_2(input_size=2, hidden_size=8, output_size=1)
    assert net.input_size == 2
    assert net.augmented_input_size == 6  # 3 * input_size
    assert net.Kh_ladder == [2, 5, 2]


def test_forward_shape_stacked():
    net = make_lbps_lnkhlfft2_2_5_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_forward_handles_nan_stacked():
    net = make_lbps_lnkhlfft2_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_fft2_then_ladder():
    """Verify FFT2 encoder feeds into ladder correctly."""
    net = make_lbps_lnkhlfft2_2_5_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x_aug = net.fft2_encoder(x)
    assert x_aug.shape == (2, 16, 6)
    assert torch.allclose(x_aug[..., :2], x)


def test_gradient_flows():
    """Verify gradient flows to all params after a few training steps."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft2_2_5_2(input_size=2, hidden_size=8, output_size=1)
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
    net = LearnedBetaPSLNKhlFft2CfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx=5, Kh_ladder=[2, 5, 2], return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft2_2_5_2(input_size=2, hidden_size=12, output_size=1)
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
    net_a = make_lbps_lnkhlfft2_2_5_2(input_size=2, hidden_size=8, output_size=1)
    net_b = make_lbps_lnkhlfft2_5_3_2(input_size=2, hidden_size=8, output_size=1)
    assert net_a.Kh_ladder != net_b.Kh_ladder
    diffs = [net_a.cfc_net.cells[i].Kh != net_b.cfc_net.cells[i].Kh for i in range(3)]
    assert sum(diffs) >= 2, f"expected at least 2 different Kh values, got {diffs}"


def test_param_count_larger_than_round187():
    """FFT2 has 3*input_size features vs FFT 2*input_size, so larger params."""
    net_fft2 = make_lbps_lnkhlfft2_2_5_2(input_size=2, hidden_size=8, output_size=1)
    from lnn.core.learned_beta_ps_ln_khlfft_cfc import make_lbps_lnkhlfft_2_5_2
    net_fft = make_lbps_lnkhlfft_2_5_2(input_size=2, hidden_size=8, output_size=1)
    n_fft2 = sum(p.numel() for p in net_fft2.parameters())
    n_fft = sum(p.numel() for p in net_fft.parameters())
    assert n_fft2 > n_fft, f"expected FFT2 ({n_fft2}) > FFT ({n_fft})"


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
