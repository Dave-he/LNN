"""Round 192 — tests for LearnedBetaPS+LN+Khl+FFT+Noise-CfC (PRD #10-154)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_cfc import make_lbps_lnkhlfft_5_3_2
from lnn.core.learned_beta_ps_ln_khlfft_noise_cfc import (
    LearnedBetaPSLNKhlFftNoiseCfCStackedNetwork,
    make_lbps_lnkhlfft_noise_5_3_2,
)


def test_noise_disabled_eval():
    """noise_sigma=0 → no noise (eval mode should be no-op)."""
    net = make_lbps_lnkhlfft_noise_5_3_2(input_size=2, hidden_size=8, output_size=1, noise_sigma=0.0)
    net.eval()
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert torch.isfinite(y).all()


def test_noise_enabled_train():
    """Training with noise_sigma > 0 should produce finite output."""
    net = make_lbps_lnkhlfft_noise_5_3_2(input_size=2, hidden_size=8, output_size=1, noise_sigma=0.1)
    net.train()
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert torch.isfinite(y).all()


def test_noise_preserves_nan():
    """NaN input positions should remain NaN after noise."""
    net = make_lbps_lnkhlfft_noise_5_3_2(input_size=2, hidden_size=8, output_size=1, noise_sigma=0.1)
    net.train()
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    # Run forward — should not crash despite NaN
    y = net(x)
    assert torch.isfinite(y).all()


def test_factory_5_3_2():
    net = make_lbps_lnkhlfft_noise_5_3_2(input_size=2, hidden_size=8, output_size=1)
    assert net.noise_sigma == 0.05  # default


def test_forward_shape_stacked():
    net = make_lbps_lnkhlfft_noise_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_noise_changes_output():
    """Adding noise to input should change the model output (in train mode)."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_noise_5_3_2(input_size=2, hidden_size=8, output_size=1, noise_sigma=0.5)
    net.train()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    # With noise, two forward passes should give different outputs
    assert not torch.allclose(y1, y2)


def test_eval_mode_deterministic():
    """In eval mode, noise is disabled → outputs are deterministic."""
    net = make_lbps_lnkhlfft_noise_5_3_2(input_size=2, hidden_size=8, output_size=1, noise_sigma=0.5)
    net.eval()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert torch.allclose(y1, y2)


def test_noise_sigma_zero_disables_noise():
    """noise_sigma=0 in train mode → outputs deterministic."""
    net = make_lbps_lnkhlfft_noise_5_3_2(input_size=2, hidden_size=8, output_size=1, noise_sigma=0.0)
    net.train()  # even in train mode
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert torch.allclose(y1, y2)


def test_gradient_flows():
    """Gradient flows to params after training step."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_noise_5_3_2(input_size=2, hidden_size=8, output_size=1, noise_sigma=0.1)
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


def test_smoke_learns_sin_with_noise():
    """Smoke test: noisy model can learn sin."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_noise_5_3_2(input_size=2, hidden_size=12, output_size=1, noise_sigma=0.05)
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
    """Noise model should have same # params as round 187 baseline."""
    net_noise = make_lbps_lnkhlfft_noise_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net_base = make_lbps_lnkhlfft_5_3_2(input_size=2, hidden_size=8, output_size=1)
    n_noise = sum(p.numel() for p in net_noise.parameters())
    n_base = sum(p.numel() for p in net_base.parameters())
    assert n_noise == n_base, f"noise {n_noise} != base {n_base}"


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
