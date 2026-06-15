"""Round 197 — tests for Mixup CfC (PRD #10-159)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_cfc import make_lbps_lnkhlfft_5_3_2
from lnn.core.learned_beta_ps_ln_khlfft_mixup_cfc import (
    LearnedBetaPSLNKhlFftMixupCfCStackedNetwork,
    make_lbps_lnkhlfft_mixup_5_3_2,
    mixup_loss,
    sample_mixup_lambda,
)


def test_mixup_disabled_eval():
    """mixup_alpha=0 → no mixup (eval mode should be no-op)."""
    net = make_lbps_lnkhlfft_mixup_5_3_2(input_size=2, hidden_size=8, output_size=1, mixup_alpha=0.0)
    net.eval()
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert torch.isfinite(y).all()


def test_mixup_enabled_train():
    """Training with mixup_alpha > 0 should produce finite output."""
    net = make_lbps_lnkhlfft_mixup_5_3_2(input_size=2, hidden_size=8, output_size=1, mixup_alpha=0.2)
    net.train()
    x = torch.randn(2, 16, 2)
    out = net(x)
    # In train mode with mixup, returns tuple (y, idx, lam)
    y, idx, lam = out
    assert torch.isfinite(y).all()
    assert idx.shape == (2,)
    assert lam.shape == (2, 1, 1)


def test_mixup_preserves_nan():
    """NaN input positions should be handled by FFT encoder."""
    net = make_lbps_lnkhlfft_mixup_5_3_2(input_size=2, hidden_size=8, output_size=1, mixup_alpha=0.2)
    net.train()
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    out = net(x)
    y, idx, lam = out
    assert torch.isfinite(y).all()


def test_factory_default():
    net = make_lbps_lnkhlfft_mixup_5_3_2(input_size=2, hidden_size=8, output_size=1)
    assert net.mixup_alpha == 0.2


def test_mixup_lambda_in_range():
    """Lambda sampled from Beta(α, α) should be in [0, 1]."""
    torch.manual_seed(0)
    for _ in range(10):
        lam = sample_mixup_lambda(0.2, 16, torch.device("cpu"))
        assert (lam >= 0.0).all() and (lam <= 1.0).all()


def test_mixup_lambda_distribution():
    """For α=0.2, lambda should be skewed toward 0 or 1."""
    torch.manual_seed(0)
    lams = sample_mixup_lambda(0.2, 1000, torch.device("cpu"))
    # For Beta(0.2, 0.2), most samples should be near 0 or 1
    near_extremes = ((lams < 0.2) | (lams > 0.8)).float().mean()
    assert near_extremes > 0.5  # more than half near extremes


def test_mixup_loss_correctness():
    """Mixup loss should be weighted combination of two losses."""
    y = torch.randn(4, 16, 1)
    t = torch.randn(4, 16, 1)
    idx = torch.tensor([1, 0, 3, 2])
    lam = torch.tensor([[[0.7]], [[0.7]], [[0.7]], [[0.7]]])
    loss = mixup_loss(y, t, idx, lam)
    expected = 0.7 * F.mse_loss(y, t) + 0.3 * F.mse_loss(y, t[idx])
    assert torch.allclose(loss, expected, atol=1e-6)


def test_mixup_changes_output():
    """Mixup should change the model output (in train mode)."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_mixup_5_3_2(input_size=2, hidden_size=8, output_size=1, mixup_alpha=0.4)
    net.train()
    x = torch.randn(4, 16, 2)
    out1 = net(x)
    out2 = net(x)
    y1 = out1[0]
    y2 = out2[0]
    assert not torch.allclose(y1, y2)


def test_eval_mode_deterministic():
    """In eval mode, mixup is disabled → outputs are deterministic."""
    net = make_lbps_lnkhlfft_mixup_5_3_2(input_size=2, hidden_size=8, output_size=1, mixup_alpha=0.4)
    net.eval()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert torch.allclose(y1, y2)


def test_mixup_alpha_zero_disables():
    """mixup_alpha=0 in train mode → outputs deterministic."""
    net = make_lbps_lnkhlfft_mixup_5_3_2(input_size=2, hidden_size=8, output_size=1, mixup_alpha=0.0)
    net.train()
    x = torch.randn(2, 16, 2)
    out1 = net(x)
    out2 = net(x)
    # mixup_alpha=0 should return plain tensor, not tuple
    assert torch.is_tensor(out1)
    assert torch.is_tensor(out2)
    assert torch.allclose(out1, out2)


def test_gradient_flows():
    """Gradient flows to params after training step with mixup loss."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_mixup_5_3_2(input_size=2, hidden_size=8, output_size=1, mixup_alpha=0.2)
    x = torch.randn(2, 16, 2)
    target = torch.randn(2, 16, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        out = net(x)
        y, idx, lam = out
        loss = mixup_loss(y, target, idx, lam)
        loss.backward()
        opt.step()
    opt.zero_grad()
    out = net(x)
    y, idx, lam = out
    loss = mixup_loss(y, target, idx, lam)
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in net.parameters()
    )
    assert has_grad


def test_smoke_learns_sin_with_mixup():
    """Smoke test: mixup model can learn sin."""
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_mixup_5_3_2(input_size=2, hidden_size=12, output_size=1, mixup_alpha=0.2)
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
        y, idx, lam = out
        loss = mixup_loss(y, target, idx, lam)
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
    """Mixup model should have same # params as round 187 baseline."""
    net_mixup = make_lbps_lnkhlfft_mixup_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net_base = make_lbps_lnkhlfft_5_3_2(input_size=2, hidden_size=8, output_size=1)
    n_mixup = sum(p.numel() for p in net_mixup.parameters())
    n_base = sum(p.numel() for p in net_base.parameters())
    assert n_mixup == n_base, f"mixup {n_mixup} != base {n_base}"


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
