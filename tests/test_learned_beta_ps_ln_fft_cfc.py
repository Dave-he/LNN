"""Round 186 — tests for LearnedBetaPS+LN+FFT-CfC (PRD #10-148)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_fft_cfc import (
    FFTInputEncoder,
    LearnedBetaPSLNFftCfCStackedNetwork,
    make_lbps_lnfft_h3_75,
    make_lbps_lnfft_h2_75,
    make_lbps_lnfft_h5_75,
)


def test_fft_encoder_shape():
    enc = FFTInputEncoder()
    x = torch.randn(2, 32, 2)
    y = enc(x)
    # FFT bins for T=32: T//2+1 = 17, padded to 32
    # Concat: 2*2 = 4 features
    assert y.shape == (2, 32, 4)


def test_fft_encoder_handles_nan():
    enc = FFTInputEncoder()
    x = torch.randn(2, 32, 2)
    x[0, 5, 0] = float("nan")
    y = enc(x)
    # NaN should propagate to original feature (FFT part is from nan_to_num)
    # Note: original x is preserved (with NaN), FFT part is clean
    assert torch.isnan(y[0, 5, 0]).item()  # original x[0,5,0]
    assert torch.isfinite(y[0, 5, 1]).item()  # FFT part
    assert torch.isfinite(y[0, 5, 2]).item()  # original x[0,5,1]
    assert torch.isfinite(y[0, 5, 3]).item()  # FFT part


def test_fft_encoder_zero_input():
    enc = FFTInputEncoder()
    x = torch.zeros(2, 32, 2)
    y = enc(x)
    # All zero
    assert torch.allclose(y, torch.zeros_like(y))


def test_fft_encoder_dc_preserved():
    """DC component (bin 0) should equal sum of input."""
    enc = FFTInputEncoder()
    x = torch.ones(1, 32, 1)  # constant = DC
    y = enc(x)
    # FFT of all-ones is [32, 0, 0, ...] — magnitude is [32, 0, 0, ...]
    # After padding to T=32, the FFT feature at t=0 is 32.0
    assert abs(y[0, 0, 1].item() - 32.0) < 1e-3  # DC bin = 32


def test_cell_init():
    net = make_lbps_lnfft_h3_75(input_size=2, hidden_size=8, output_size=1)
    assert net.input_size == 2
    assert net.augmented_input_size == 4
    assert net.Kh_ladder == [3, 3, 3]


def test_factory_h3_75():
    net = make_lbps_lnfft_h3_75(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh_ladder == [3, 3, 3]


def test_factory_h2_75():
    net = make_lbps_lnfft_h2_75(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh_ladder == [2, 2, 2]


def test_factory_h5_75():
    net = make_lbps_lnfft_h5_75(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh_ladder == [5, 5, 5]


def test_forward_shape_stacked():
    net = make_lbps_lnfft_h3_75(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_forward_handles_nan_stacked():
    net = make_lbps_lnfft_h3_75(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_gradient_flows():
    """Verify gradient flows to all params after a few training steps."""
    torch.manual_seed(0)
    net = make_lbps_lnfft_h3_75(input_size=2, hidden_size=8, output_size=1)
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
    net = LearnedBetaPSLNFftCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx=5, Kh_ladder=[3, 3, 3], return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnfft_h3_75(input_size=2, hidden_size=12, output_size=1)
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
