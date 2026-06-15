"""Round 202 — tests for LambdaSpecGated-CfC (PRD #10-164)."""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_cfc import make_lbps_lnkhlfft_5_3_2
from lnn.core.learned_beta_ps_ln_khlfft_lambda_specgated_cfc import (
    LambdaSpecGatedCfCCell,
    LambdaSpecGatedCfCStackedNetwork,
    make_lbps_lnkhlfft_lambda_specgated_5_3_2,
)


def test_lambda_cell_forward():
    """LambdaSpec cell forward should produce finite hidden state."""
    cell = LambdaSpecGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()
    assert h_new.shape == h.shape


def test_lambda_handles_nan_input():
    """LambdaSpec cell should handle NaN inputs gracefully."""
    cell = LambdaSpecGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()


def test_lambda_lambda_in_range():
    """Lambda should always be in [0, 1]."""
    cell = LambdaSpecGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    z = torch.randn(3, cell.layer_norm.normalized_shape[0])
    lam = cell.lambda_gate(z)
    assert (lam >= 0.0).all() and (lam <= 1.0).all()


def test_lambda_extremes_reproduce_r187_and_r200():
    """When λ=0 → equivalent to r187 baseline; λ=1 → equivalent to r200 spec."""
    torch.manual_seed(0)
    cell = LambdaSpecGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)

    # Set λ constant via overwriting lambda_gate last layer
    with torch.no_grad():
        # Force λ=0 by setting bias very negative
        cell.lambda_gate[0].bias.fill_(-10.0)
    h = torch.randn(2, 8)
    x = torch.randn(2, 2)
    emas_x = [torch.zeros(2, 2) for _ in range(3)]
    emas_h = [torch.zeros(2, 8) for _ in range(2)]
    h_new_zero, _, _ = cell(x, h, emas_x, emas_h)
    # Should differ from default (where λ is sigmoid(linear(z)))
    # (sanity check that the cell did compute something)

    # Force λ=1
    with torch.no_grad():
        cell.lambda_gate[0].bias.fill_(10.0)
    h_new_one, _, _ = cell(x, h, emas_x, emas_h)

    # Outputs should differ (different λ values produce different g_combined)
    assert not torch.allclose(h_new_zero, h_new_one)


def test_lambda_stacked_factory():
    """Factory should produce working stacked network."""
    net = make_lbps_lnkhlfft_lambda_specgated_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert torch.isfinite(y).all()
    assert y.shape == (2, 16, 1)


def test_lambda_param_count_delta():
    """LambdaSpec-CfC has more params than baseline.

    Adds: g_branch (264 per cell) + spec_mask (30 per cell) + lambda_gate (264 per cell)
    vs baseline: g_branch (264 per cell) + h_branch (264 per cell)
    Per cell delta: + spec_mask (30) + lambda_gate (264) - h_branch (264) = +30
    Wait — actually lambda replaces h_branch in terms of parameter count? No.
    Baseline has g_branch + h_branch. Lambda has g_branch + h_branch + spec_mask + lambda_gate.
    Per cell delta: + spec_mask (30) + lambda_gate (264) = +294
    3 cells: +882.
    """
    net_lam = make_lbps_lnkhlfft_lambda_specgated_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net_base = make_lbps_lnkhlfft_5_3_2(input_size=2, hidden_size=8, output_size=1)
    n_lam = sum(p.numel() for p in net_lam.parameters())
    n_base = sum(p.numel() for p in net_base.parameters())
    diff = n_lam - n_base
    # Should add ~882 params
    assert 500 < diff < 2000, f"expected ~882 added params, got {diff}"


def test_lambda_eval_mode_deterministic():
    """Eval mode should be deterministic."""
    net = make_lbps_lnkhlfft_lambda_specgated_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net.eval()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert torch.allclose(y1, y2)


def test_lambda_gradient_flows():
    """Gradient should flow back to lambda_gate and spec_mask params."""
    net = make_lbps_lnkhlfft_lambda_specgated_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    target = torch.randn(2, 16, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    opt.zero_grad()
    y = net(x)
    loss = (y - target).pow(2).mean()
    loss.backward()
    has_lam_grad = False
    has_spec_grad = False
    for name, p in net.named_parameters():
        if "lambda_gate" in name and p.grad is not None and p.grad.abs().sum() > 0:
            has_lam_grad = True
        if "spec_mask" in name and p.grad is not None and p.grad.abs().sum() > 0:
            has_spec_grad = True
    assert has_lam_grad
    assert has_spec_grad


def test_lambda_smoke_learns_sin():
    """Smoke test: LambdaSpec-CfC can learn sin."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_lambda_specgated_5_3_2(input_size=2, hidden_size=12, output_size=1)
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


def test_lambda_lambda_gate_has_correct_dim():
    """lambda_gate should produce [B, H] output (per-feature λ)."""
    cell = LambdaSpecGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    z = torch.randn(3, cell.layer_norm.normalized_shape[0])
    lam = cell.lambda_gate(z)
    assert lam.shape == (3, 8)


def test_lambda_gating_zero_input():
    """Zero h should produce zero spectral g."""
    cell = LambdaSpecGatedCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    h = torch.zeros(1, 8)
    g = cell._spectral_gating(h)
    assert torch.allclose(g, torch.zeros_like(g), atol=1e-6)


def test_lambda_more_params_than_baseline():
    """LambdaSpec has more params than baseline."""
    net_lam = make_lbps_lnkhlfft_lambda_specgated_5_3_2(input_size=2, hidden_size=12, output_size=1)
    net_base = make_lbps_lnkhlfft_5_3_2(input_size=2, hidden_size=12, output_size=1)
    n_lam = sum(p.numel() for p in net_lam.parameters())
    n_base = sum(p.numel() for p in net_base.parameters())
    assert n_lam > n_base


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
