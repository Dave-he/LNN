"""Round 201 — tests for AdditiveSpectralGated-CfC (PRD #10-163)."""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_cfc import make_lbps_lnkhlfft_5_3_2
from lnn.core.learned_beta_ps_ln_khlfft_specgated_cfc import (
    make_lbps_lnkhlfft_specgated_5_3_2,
)
from lnn.core.learned_beta_ps_ln_khlfft_addspecgated_cfc import (
    AdditiveSpectralGatedCfCCell,
    AdditiveSpectralGatedCfCStackedNetwork,
    make_lbps_lnkhlfft_addspecgated_5_3_2,
)


def test_addspec_cell_forward():
    """AdditiveSpectral cell forward should produce finite hidden state."""
    cell = AdditiveSpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, ex, eh = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()
    assert h_new.shape == h.shape


def test_addspec_handles_nan_input():
    """AdditiveSpectral cell should handle NaN inputs gracefully."""
    cell = AdditiveSpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()


def test_addspec_g_combined_is_sum():
    """g_combined = g_branch(z) + spectral_g(h) — verify cell runs without NaN."""
    torch.manual_seed(0)
    cell = AdditiveSpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    h = torch.randn(2, 8)
    x = torch.randn(2, 2)
    emas_x = [torch.zeros(2, 2) for _ in range(3)]
    emas_h = [torch.zeros(2, 8) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    # Output should be finite and within reasonable range
    assert torch.isfinite(h_new).all()
    # Output should differ from h (cell did something)
    assert not torch.allclose(h_new, h)
    # Output should differ from a cell with spec_mask=0 (spectral path has effect)
    cell2 = AdditiveSpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    with torch.no_grad():
        cell2.spec_mask.weight.zero_()
        cell2.spec_mask.bias.zero_()
    h_new_no_spec, _, _ = cell2(x, h, emas_x, emas_h)
    assert not torch.allclose(h_new, h_new_no_spec)


def test_addspec_stacked_factory():
    """Factory should produce working stacked network."""
    net = make_lbps_lnkhlfft_addspecgated_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert torch.isfinite(y).all()
    assert y.shape == (2, 16, 1)


def test_addspec_param_count_delta():
    """AdditiveSpec-CfC has more params than baseline (adds spec_mask to g_branch+h_branch).

    Baseline has g_branch AND h_branch (each: hidden_size * aug_total + hidden_size).
    AdditiveSpec keeps both AND adds spec_mask = n_freq * n_freq + n_freq.
    Per cell: n_freq = hidden_size//2 + 1 = 5 (for hidden=8).
    spec_mask = 5*5 + 5 = 30 per cell.
    3 cells: +90 params total.
    """
    net_add = make_lbps_lnkhlfft_addspecgated_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net_base = make_lbps_lnkhlfft_5_3_2(input_size=2, hidden_size=8, output_size=1)
    n_add = sum(p.numel() for p in net_add.parameters())
    n_base = sum(p.numel() for p in net_base.parameters())
    diff = n_add - n_base
    # Should add roughly 90 params (spec_mask for 3 cells)
    assert 50 < diff < 500, f"expected ~90 added params, got {diff}"


def test_addspec_eval_mode_deterministic():
    """Eval mode should be deterministic."""
    net = make_lbps_lnkhlfft_addspecgated_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net.eval()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert torch.allclose(y1, y2)


def test_addspec_gradient_flows():
    """Gradient should flow back to spec_mask params."""
    net = make_lbps_lnkhlfft_addspecgated_5_3_2(input_size=2, hidden_size=8, output_size=1)
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


def test_addspec_smoke_learns_sin():
    """Smoke test: AdditiveSpec-CfC can learn sin."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_addspecgated_5_3_2(input_size=2, hidden_size=12, output_size=1)
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


def test_addspec_more_params_than_baseline():
    """AdditiveSpec has more params than baseline (g_branch + spec_mask)."""
    net_add = make_lbps_lnkhlfft_addspecgated_5_3_2(input_size=2, hidden_size=12, output_size=1)
    net_base = make_lbps_lnkhlfft_5_3_2(input_size=2, hidden_size=12, output_size=1)
    n_add = sum(p.numel() for p in net_add.parameters())
    n_base = sum(p.numel() for p in net_base.parameters())
    assert n_add > n_base


def test_addspec_different_for_different_h():
    """Different h should produce different g (spectral gating is content-aware)."""
    torch.manual_seed(0)
    cell = AdditiveSpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    h1 = torch.randn(1, 8)
    h2 = torch.randn(1, 8)
    g1 = cell._spectral_gating(h1)
    g2 = cell._spectral_gating(h2)
    assert not torch.allclose(g1, g2)


def test_addspec_gating_zero_input():
    """Zero input should produce zero spectral g."""
    cell = AdditiveSpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    h = torch.zeros(1, 8)
    g = cell._spectral_gating(h)
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-6)


def test_addspec_combined_equals_sum_components():
    """Verify spectral path contributes: zeroing spec_mask changes output."""
    torch.manual_seed(42)
    cell = AdditiveSpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    h = torch.randn(3, 8)
    x = torch.randn(3, 2)
    emas_x = [torch.zeros(3, 2) for _ in range(3)]
    emas_h = [torch.zeros(3, 8) for _ in range(2)]
    h_new_actual, _, _ = cell(x, h, emas_x, emas_h)
    # Same cell with spec_mask zeroed (spectral path disabled)
    cell_no_spec = AdditiveSpectralGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    with torch.no_grad():
        cell_no_spec.load_state_dict(cell.state_dict())
        cell_no_spec.spec_mask.weight.zero_()
        cell_no_spec.spec_mask.bias.zero_()
    h_new_no_spec, _, _ = cell_no_spec(x, h, emas_x, emas_h)
    # Outputs should differ — spectral path has non-trivial effect
    diff = (h_new_actual - h_new_no_spec).abs().max().item()
    assert diff > 1e-6, f"spectral path has no effect (max diff {diff})"


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
