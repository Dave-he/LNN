"""Round 212 — tests for 4ScaleSpectralCfC (PRD #10-174)."""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_4spectral_cfc import (
    FourScaleSpectralCfCCell,
    make_lbps_lnkhlfft_4spectral_5_3_2,
)


def test_4spectral_cell_forward():
    cell = FourScaleSpectralCfCCell(input_size=2, hidden_size=16, Kx=3, Kh=2)
    cell.train()
    x = torch.randn(4, 2)
    h = torch.zeros(4, 16)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 16) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()
    assert h_new.shape == h.shape


def test_4spectral_handles_nan():
    cell = FourScaleSpectralCfCCell(input_size=2, hidden_size=16, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 16)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 16) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()


def test_4spectral_gradient_flows():
    cell = FourScaleSpectralCfCCell(input_size=2, hidden_size=16, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    h = torch.randn(4, 16)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 16) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    loss = h_new.pow(2).mean()
    loss.backward()
    has_grad = False
    for name, p in cell.named_parameters():
        if "spec_mask" in name and p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad


def test_4spectral_stacked_factory():
    net = make_lbps_lnkhlfft_4spectral_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert torch.isfinite(y).all()
    assert y.shape == (2, 16, 1)


def test_4spectral_stacked_eval_deterministic():
    net = make_lbps_lnkhlfft_4spectral_5_3_2(input_size=2, hidden_size=16, output_size=1)
    net.eval()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert torch.allclose(y1, y2)


def test_4spectral_stacked_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_4spectral_5_3_2(input_size=2, hidden_size=12, output_size=1)
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


def test_4spectral_smoke_long_sequence():
    net = make_lbps_lnkhlfft_4spectral_5_3_2(input_size=2, hidden_size=16, output_size=1)
    x = torch.randn(2, 32, 2)
    y = net(x)
    assert torch.isfinite(y).all()
    assert y.shape == (2, 32, 1)


def test_4spectral_zero_input():
    cell = FourScaleSpectralCfCCell(input_size=2, hidden_size=16, Kx=3, Kh=2)
    cell.eval()
    h = torch.zeros(1, 16)
    g = cell._4scale_spectral(h)
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-6)


def test_4spectral_has_4_masks():
    cell = FourScaleSpectralCfCCell(input_size=2, hidden_size=16, Kx=3, Kh=2)
    # Count unique mask layers (each linear has weight + bias)
    mask_idxs = {n.split(".")[0] for n, _ in cell.named_parameters() if "spec_mask" in n}
    assert len(mask_idxs) == 4


def test_4spectral_4_masks_have_grad():
    """All 4 mask layers should receive non-zero gradients."""
    cell = FourScaleSpectralCfCCell(input_size=2, hidden_size=16, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    h = torch.randn(4, 16)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 16) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    loss = h_new.pow(2).mean()
    loss.backward()
    grad_masks = set()
    for name, p in cell.named_parameters():
        if "spec_mask" in name and p.grad is not None and p.grad.abs().sum() > 0:
            grad_masks.add(name.split(".")[0])
    assert len(grad_masks) == 4


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
