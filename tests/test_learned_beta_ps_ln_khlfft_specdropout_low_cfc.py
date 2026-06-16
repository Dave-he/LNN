"""Round 205 — tests for SpectralDropoutLow-CfC (PRD #10-167)."""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_specdropout_low_cfc import (
    SpectralDropoutLowCfCCell,
    make_lbps_lnkhlfft_specdroplow_5_3_2,
)


def test_spectral_dropout_low_cell_forward():
    cell = SpectralDropoutLowCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.train()
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()
    assert h_new.shape == h.shape


def test_spectral_dropout_low_handles_nan():
    cell = SpectralDropoutLowCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()


def test_spectral_dropout_low_p_02():
    """Default dropout_p should be 0.2."""
    cell = SpectralDropoutLowCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    assert cell.dropout_p == 0.2


def test_spectral_dropout_low_active_in_training_only():
    cell = SpectralDropoutLowCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2, dropout_p=0.2)
    h = torch.randn(4, 8)
    cell.train()
    g_train1 = cell._spectral_gating(h)
    g_train2 = cell._spectral_gating(h)
    cell.eval()
    g_eval1 = cell._spectral_gating(h)
    g_eval2 = cell._spectral_gating(h)
    assert torch.allclose(g_eval1, g_eval2)
    assert not torch.allclose(g_train1, g_train2)


def test_spectral_dropout_low_stacked_factory():
    net = make_lbps_lnkhlfft_specdroplow_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert torch.isfinite(y).all()
    assert y.shape == (2, 16, 1)


def test_spectral_dropout_low_eval_mode_deterministic():
    net = make_lbps_lnkhlfft_specdroplow_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net.eval()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert torch.allclose(y1, y2)


def test_spectral_dropout_low_gradient_flows():
    net = make_lbps_lnkhlfft_specdroplow_5_3_2(input_size=2, hidden_size=8, output_size=1)
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


def test_spectral_dropout_low_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_specdroplow_5_3_2(input_size=2, hidden_size=12, output_size=1, dropout_p=0.2)
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


def test_spectral_dropout_low_p_configurable():
    cell1 = SpectralDropoutLowCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2, dropout_p=0.1)
    cell2 = SpectralDropoutLowCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2, dropout_p=0.4)
    assert cell1.dropout_p == 0.1
    assert cell2.dropout_p == 0.4


def test_spectral_dropout_low_mask_variation_in_train():
    torch.manual_seed(0)
    cell = SpectralDropoutLowCfCCell(input_size=2, hidden_size=16, Kx=3, Kh=2, dropout_p=0.2)
    cell.train()
    h = torch.randn(8, 16)
    g_runs = [cell._spectral_gating(h) for _ in range(5)]
    for i in range(1, len(g_runs)):
        assert not torch.allclose(g_runs[0], g_runs[i])


def test_spectral_dropout_low_gating_zero_input():
    cell = SpectralDropoutLowCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2, dropout_p=0.2)
    cell.eval()
    h = torch.zeros(1, 8)
    g = cell._spectral_gating(h)
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-6)


def test_spectral_dropout_low_p_zero_no_dropout():
    cell = SpectralDropoutLowCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2, dropout_p=0.0)
    cell.train()
    h = torch.randn(4, 8)
    g1 = cell._spectral_gating(h)
    g2 = cell._spectral_gating(h)
    assert torch.allclose(g1, g2)


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
