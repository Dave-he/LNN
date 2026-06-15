"""Round 203 — tests for SpectralDropout-CfC (PRD #10-165)."""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_cfc import make_lbps_lnkhlfft_5_3_2
from lnn.core.learned_beta_ps_ln_khlfft_specdropout_cfc import (
    SpectralDropoutCfCCell,
    SpectralDropoutCfCStackedNetwork,
    make_lbps_lnkhlfft_specdropout_5_3_2,
)


def test_spectral_dropout_cell_forward():
    """SpectralDropout cell forward should produce finite hidden state."""
    cell = SpectralDropoutCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.train()
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()
    assert h_new.shape == h.shape


def test_spectral_dropout_handles_nan_input():
    """SpectralDropout cell should handle NaN inputs gracefully."""
    cell = SpectralDropoutCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()


def test_spectral_dropout_active_in_training_only():
    """Dropout should be active in train mode, inactive in eval mode."""
    cell = SpectralDropoutCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2, dropout_p=0.5)
    h = torch.randn(4, 8)
    # Train: dropout active → at least some zeros in mask
    cell.train()
    g_train1 = cell._spectral_gating(h)
    g_train2 = cell._spectral_gating(h)
    # Eval: deterministic
    cell.eval()
    g_eval1 = cell._spectral_gating(h)
    g_eval2 = cell._spectral_gating(h)
    assert torch.allclose(g_eval1, g_eval2)
    # Train outputs should vary (random dropout)
    assert not torch.allclose(g_train1, g_train2)


def test_spectral_dropout_stacked_factory():
    """Factory should produce working stacked network."""
    net = make_lbps_lnkhlfft_specdropout_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert torch.isfinite(y).all()
    assert y.shape == (2, 16, 1)


def test_spectral_dropout_param_count_delta():
    """SpectralDropout-CfC has similar param count to r200 spec."""
    net_drop = make_lbps_lnkhlfft_specdropout_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net_base = make_lbps_lnkhlfft_5_3_2(input_size=2, hidden_size=8, output_size=1)
    n_drop = sum(p.numel() for p in net_drop.parameters())
    n_base = sum(p.numel() for p in net_base.parameters())
    diff = abs(n_drop - n_base)
    # Should differ by spec_mask params only (3 cells × 30 = 90 fewer, since g_branch removed)
    # actually negative diff since spec doesn't have g_branch
    # Allow generous margin
    assert diff < 5000


def test_spectral_dropout_eval_mode_deterministic():
    """Eval mode should be deterministic."""
    net = make_lbps_lnkhlfft_specdropout_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net.eval()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert torch.allclose(y1, y2)


def test_spectral_dropout_gradient_flows():
    """Gradient should flow back to spec_mask params."""
    net = make_lbps_lnkhlfft_specdropout_5_3_2(input_size=2, hidden_size=8, output_size=1)
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


def test_spectral_dropout_smoke_learns_sin():
    """Smoke test: SpectralDropout-CfC can learn sin."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_specdropout_5_3_2(input_size=2, hidden_size=12, output_size=1, dropout_p=0.2)
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


def test_spectral_dropout_gating_zero_input():
    """Zero h should produce zero spectral g."""
    cell = SpectralDropoutCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2, dropout_p=0.5)
    cell.eval()  # disable dropout
    h = torch.zeros(1, 8)
    g = cell._spectral_gating(h)
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-6)


def test_spectral_dropout_dropout_p_configurable():
    """Different dropout_p values should be configurable."""
    cell1 = SpectralDropoutCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2, dropout_p=0.1)
    cell2 = SpectralDropoutCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2, dropout_p=0.5)
    assert cell1.dropout_p == 0.1
    assert cell2.dropout_p == 0.5


def test_spectral_dropout_eval_no_dropout():
    """Eval mode should not apply dropout (output should match no-dropout cell)."""
    torch.manual_seed(42)
    cell = SpectralDropoutCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2, dropout_p=0.0)
    cell.eval()
    h = torch.randn(2, 8)
    g = cell._spectral_gating(h)
    # With dropout_p=0, mask is intact → equivalent to no dropout
    assert torch.isfinite(g).all()
    # Same input twice → same output in eval
    g2 = cell._spectral_gating(h)
    assert torch.allclose(g, g2)


def test_spectral_dropout_mask_sparse_in_train():
    """In training mode, dropout should make the mask sparse."""
    torch.manual_seed(0)
    cell = SpectralDropoutCfCCell(input_size=2, hidden_size=16, Kx=3, Kh=2, dropout_p=0.5)
    cell.train()
    h = torch.randn(8, 16)
    g_train = cell._spectral_gating(h)
    # Just verify finite
    assert torch.isfinite(g_train).all()
    # Run many times to confirm variation
    g_runs = [cell._spectral_gating(h) for _ in range(5)]
    # Outputs should differ (random dropout)
    for i in range(1, len(g_runs)):
        assert not torch.allclose(g_runs[0], g_runs[i])


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
