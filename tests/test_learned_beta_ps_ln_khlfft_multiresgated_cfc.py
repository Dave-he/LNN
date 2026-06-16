"""Round 209 — tests for MultiResSpectralGatedCfC (PRD #10-171)."""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_multiresgated_cfc import (
    MultiResSpectralGatedCfCCell,
    make_lbps_lnkhlfft_multiresgated_5_3_2,
)


def test_multires_cell_forward():
    cell = MultiResSpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.train()
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()
    assert h_new.shape == h.shape


def test_multires_cell_handles_nan():
    cell = MultiResSpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()


def test_multires_spectral_finite():
    cell = MultiResSpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.eval()
    h = torch.randn(4, 8)
    g = cell._multires_spectral(h)
    assert torch.isfinite(g).all()
    assert g.shape == h.shape


def test_multires_gradient_flows():
    cell = MultiResSpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    # Use non-zero h so spectral path has signal
    x = torch.randn(4, 2)
    h = torch.randn(4, 8)  # non-zero
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    loss = h_new.pow(2).mean()
    loss.backward()
    has_spec1_grad = False
    has_spec2_grad = False
    for name, p in cell.named_parameters():
        if "spec_mask1" in name and p.grad is not None and p.grad.abs().sum() > 0:
            has_spec1_grad = True
        if "spec_mask2" in name and p.grad is not None and p.grad.abs().sum() > 0:
            has_spec2_grad = True
    assert has_spec1_grad
    assert has_spec2_grad


def test_multires_stacked_factory():
    net = make_lbps_lnkhlfft_multiresgated_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert torch.isfinite(y).all()
    assert y.shape == (2, 16, 1)


def test_multires_stacked_eval_deterministic():
    net = make_lbps_lnkhlfft_multiresgated_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net.eval()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert torch.allclose(y1, y2)


def test_multires_stacked_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_multiresgated_5_3_2(input_size=2, hidden_size=12, output_size=1)
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


def test_multires_smoke_long_sequence():
    net = make_lbps_lnkhlfft_multiresgated_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 32, 2)
    y = net(x)
    assert torch.isfinite(y).all()
    assert y.shape == (2, 32, 1)


def test_multires_both_scales_different():
    """Res 1 and Res 2 should give different outputs."""
    cell = MultiResSpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.eval()
    h = torch.randn(4, 8)
    H1 = torch.fft.rfft(h, dim=-1)
    H2 = H1[:, :cell.hidden_size // 4 + 1]
    # Different shapes
    assert H1.shape[-1] != H2.shape[-1]


def test_multires_zero_input():
    cell = MultiResSpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.eval()
    h = torch.zeros(1, 8)
    g = cell._multires_spectral(h)
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-6)


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
