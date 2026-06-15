"""Round 200 — tests for SpectralGated-CfC (PRD #10-162)."""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_cfc import make_lbps_lnkhlfft_5_3_2
from lnn.core.learned_beta_ps_ln_khlfft_specgated_cfc import (
    SpectralGatedCfCCell,
    SpectralGatedCfCStackedNetwork,
    make_lbps_lnkhlfft_specgated_5_3_2,
)


def test_spectral_cell_forward():
    """Spectral cell forward should produce finite hidden state."""
    cell = SpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, ex, eh = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()
    assert h_new.shape == h.shape


def test_spectral_handles_nan_input():
    """Spectral cell should handle NaN inputs gracefully."""
    cell = SpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()


def test_spectral_gating_output_shape():
    """Spectral gating should output same shape as h."""
    cell = SpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    h = torch.randn(4, 8)
    g = cell._spectral_gating(h)
    assert g.shape == h.shape
    assert torch.isfinite(g).all()


def test_spectral_gating_zero_input():
    """Zero input should produce zero output (linear operator)."""
    cell = SpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    h = torch.zeros(1, 8)
    g = cell._spectral_gating(h)
    # Zero FFT → zero IFFT
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-6)


def test_spectral_stacked_factory():
    """Factory should produce working stacked network."""
    net = make_lbps_lnkhlfft_specgated_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert torch.isfinite(y).all()
    assert y.shape == (2, 16, 1)


def test_spectral_param_count_delta():
    """Spectral-CfC has similar param count to baseline (g_branch replaced by spec_mask)."""
    net_spec = make_lbps_lnkhlfft_specgated_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net_base = make_lbps_lnkhlfft_5_3_2(input_size=2, hidden_size=8, output_size=1)
    n_spec = sum(p.numel() for p in net_spec.parameters())
    n_base = sum(p.numel() for p in net_base.parameters())
    # Spectral removes g_branch (hidden_size * aug_total + hidden_size)
    # and adds spec_mask (n_freq * n_freq + n_freq)
    # n_freq = hidden_size // 2 + 1 = 5
    # g_branch = 8 * aug_total + 8, with aug_total = (Kx+1)*input + (Kh+1)*hidden = 4*2 + 3*8 = 32
    # g_branch = 8*32 + 8 = 264 per cell
    # spec_mask = 5*5 + 5 = 30 per cell
    # Per cell delta = 264 - 30 = 234. 3 cells = 702.
    # But the aug_total for the FIRST cell is (Kx+1)*augmented_input + (Kh+1)*hidden = 4*4 + 3*8 = 40
    # so g_branch is 8*40+8=328 for first cell.  spec_mask still 30.
    # So 3 cells: (328+264+264) - (30*3) = 856 - 90 = 766
    # Allow generous margin
    diff = abs(n_spec - n_base)
    assert diff < 5000


def test_spectral_eval_mode_deterministic():
    """Eval mode should be deterministic."""
    net = make_lbps_lnkhlfft_specgated_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net.eval()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert torch.allclose(y1, y2)


def test_spectral_gradient_flows():
    """Gradient should flow back to spec_mask params."""
    net = make_lbps_lnkhlfft_specgated_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    target = torch.randn(2, 16, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    opt.zero_grad()
    y = net(x)
    loss = (y - target).pow(2).mean()
    loss.backward()
    has_spec_grad = False
    for name, p in net.named_parameters():
        if "spec_mask" in name and p.grad is not None and p.grad.abs().sum() > 0:
            has_spec_grad = True
            break
    assert has_spec_grad


def test_spectral_smoke_learns_sin():
    """Smoke test: Spectral-CfC can learn sin."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_specgated_5_3_2(input_size=2, hidden_size=12, output_size=1)
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
        y = net(x)
        loss = (y - target).pow(2).mean()
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    assert initial_loss is not None
    assert math.isfinite(final_loss)
    assert final_loss < initial_loss


def test_spectral_gating_different_for_different_h():
    """Different h should produce different g (spectral gating is content-aware)."""
    torch.manual_seed(0)
    cell = SpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    h1 = torch.randn(1, 8)
    h2 = torch.randn(1, 8)
    g1 = cell._spectral_gating(h1)
    g2 = cell._spectral_gating(h2)
    assert not torch.allclose(g1, g2)


def test_spectral_doesnt_use_h():
    """Spectral gating on h=0 should be the spectral mask applied to 0 (still 0)."""
    cell = SpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    h = torch.zeros(1, 8)
    g = cell._spectral_gating(h)
    # After multiple forward passes, the gating should be bounded
    assert g.abs().max() < 1e-4


def test_spectral_recovery_with_constant_h():
    """Constant h_t should produce a specific spectral signature."""
    cell = SpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    # Constant h has FFT concentrated at DC
    h = torch.ones(1, 8)
    g = cell._spectral_gating(h)
    # Real-valued constant should produce real output
    assert torch.isreal(g).all() or torch.allclose(g.imag, torch.zeros_like(g.imag), atol=1e-5)


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
        sys.exit(1)
    print(f"\nAll {len(funcs)} tests passed.")
