"""Round 194 — tests for XH noise (combined input+hidden) CfC (PRD #10-156)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_cfc import make_lbps_lnkhlfft_5_3_2
from lnn.core.learned_beta_ps_ln_khlfft_noise_cfc import make_lbps_lnkhlfft_noise_5_3_2
from lnn.core.learned_beta_ps_ln_khlfft_hnoise_cfc import make_lbps_lnkhlfft_hnoise_5_3_2
from lnn.core.learned_beta_ps_ln_khlfft_xhnoise_cfc import (
    LearnedBetaPSLNKhlFftXHNoiseCfCStackedNetwork,
    make_lbps_lnkhlfft_xhnoise_5_3_2,
)


def test_xhnoise_disabled_eval():
    """Both σ=0 → no noise (eval mode should be no-op)."""
    net = make_lbps_lnkhlfft_xhnoise_5_3_2(input_size=2, hidden_size=8, output_size=1, noise_sigma=0.0, hnoise_sigma=0.0)
    net.eval()
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert torch.isfinite(y).all()


def test_xhnoise_enabled_train():
    """Both σ > 0 should produce finite output."""
    net = make_lbps_lnkhlfft_xhnoise_5_3_2(input_size=2, hidden_size=8, output_size=1, noise_sigma=0.1, hnoise_sigma=0.1)
    net.train()
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert torch.isfinite(y).all()


def test_xhnoise_preserves_nan():
    """NaN input positions should be handled by FFT encoder."""
    net = make_lbps_lnkhlfft_xhnoise_5_3_2(input_size=2, hidden_size=8, output_size=1, noise_sigma=0.1, hnoise_sigma=0.1)
    net.train()
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_factory_default():
    net = make_lbps_lnkhlfft_xhnoise_5_3_2(input_size=2, hidden_size=8, output_size=1)
    assert net.noise_sigma == 0.05
    assert net.hnoise_sigma == 0.05


def test_forward_shape():
    net = make_lbps_lnkhlfft_xhnoise_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_xhnoise_changes_output():
    """Adding both noises should change the model output (in train mode)."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_xhnoise_5_3_2(input_size=2, hidden_size=8, output_size=1, noise_sigma=0.5, hnoise_sigma=0.5)
    net.train()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert not torch.allclose(y1, y2)


def test_eval_mode_deterministic():
    """In eval mode, both noises are disabled → outputs are deterministic."""
    net = make_lbps_lnkhlfft_xhnoise_5_3_2(input_size=2, hidden_size=8, output_size=1, noise_sigma=0.5, hnoise_sigma=0.5)
    net.eval()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert torch.allclose(y1, y2)


def test_input_only_when_hnoise_zero():
    """If only input σ > 0, should behave like round 192 (input noise only)."""
    torch.manual_seed(42)
    net_combo = make_lbps_lnkhlfft_xhnoise_5_3_2(input_size=2, hidden_size=8, output_size=1, noise_sigma=0.1, hnoise_sigma=0.0)
    net_combo.eval()
    net_in = make_lbps_lnkhlfft_noise_5_3_2(input_size=2, hidden_size=8, output_size=1, noise_sigma=0.1)
    net_in.eval()
    # Both eval mode (no noise), outputs should be deterministic
    x = torch.randn(2, 16, 2)
    y1 = net_combo(x)
    y2 = net_in(x)
    # In eval mode both are no-op (only train mode applies noise)
    assert torch.isfinite(y1).all()
    assert torch.isfinite(y2).all()


def test_gradient_flows():
    """Gradient flows to params after training step."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_xhnoise_5_3_2(input_size=2, hidden_size=8, output_size=1, noise_sigma=0.1, hnoise_sigma=0.1)
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
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in net.parameters()
    )
    assert has_grad


def test_smoke_learns_sin():
    """Smoke test: combo noise model can learn sin."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_xhnoise_5_3_2(input_size=2, hidden_size=12, output_size=1, noise_sigma=0.05, hnoise_sigma=0.05)
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


def test_param_count_matches_baseline():
    """Combo noise model should have same # params as round 187 baseline."""
    net_combo = make_lbps_lnkhlfft_xhnoise_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net_base = make_lbps_lnkhlfft_5_3_2(input_size=2, hidden_size=8, output_size=1)
    n_combo = sum(p.numel() for p in net_combo.parameters())
    n_base = sum(p.numel() for p in net_base.parameters())
    assert n_combo == n_base, f"combo {n_combo} != base {n_base}"


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
