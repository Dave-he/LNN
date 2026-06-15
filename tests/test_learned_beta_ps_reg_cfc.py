"""Round 175 — tests for LearnedPerScaleBeta+Reg-CfC (PRD #10-137)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_reg_cfc import (
    LearnedBetaPSRegCfCStackedNetwork,
    make_lbps_reg_l01,
    make_lbps_reg_l001,
    make_lbps_reg_l1,
    make_lbps_reg_l10,
    make_lbps_reg_kh2_l01,
    make_lbps_reg_kh5_l01,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_init_default_params():
    """Default params."""
    net = LearnedBetaPSRegCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
    )
    assert net.beta_target == 0.75
    assert net.reg_lambda == 0.01


def test_reg_loss_at_init_zero():
    """At init (β=0.75 = target), reg loss should be ~0."""
    net = make_lbps_reg_l01(input_size=2, hidden_size=8, output_size=1)
    reg = net.reg_loss()
    assert abs(reg.item()) < 0.01  # small but not necessarily 0 (numerical)


def test_reg_loss_pulls_beta_back():
    """If we set β away from target, reg loss should be >0."""
    net = make_lbps_reg_l01(input_size=2, hidden_size=8, output_size=1)
    # Move all β to 0.5 (away from target 0.75)
    for cell in net.cells:
        # Compute logit(0.5) = 0
        cell.beta_x_raw.data.fill_(0.0)
        cell.beta_h_raw.data.fill_(0.0)
    reg = net.reg_loss()
    # Should be λ * mean((0.5-0.75)^2) = 0.01 * 0.0625 = 0.000625
    assert reg.item() > 0.0005


def test_factory_l01():
    """make_lbps_reg_l01: λ=0.01."""
    net = make_lbps_reg_l01(input_size=2, hidden_size=8, output_size=1)
    assert net.reg_lambda == 0.01


def test_factory_l001():
    """make_lbps_reg_l001: λ=0.001."""
    net = make_lbps_reg_l001(input_size=2, hidden_size=8, output_size=1)
    assert net.reg_lambda == 0.001


def test_factory_l1():
    """make_lbps_reg_l1: λ=1.0."""
    net = make_lbps_reg_l1(input_size=2, hidden_size=8, output_size=1)
    assert net.reg_lambda == 1.0


def test_factory_l10():
    """make_lbps_reg_l10: λ=10.0."""
    net = make_lbps_reg_l10(input_size=2, hidden_size=8, output_size=1)
    assert net.reg_lambda == 10.0


def test_factory_kh2_l01():
    """make_lbps_reg_kh2_l01: Kh=2."""
    net = make_lbps_reg_kh2_l01(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 2
    assert net.reg_lambda == 0.01


def test_factory_kh5_l01():
    """make_lbps_reg_kh5_l01: Kh=5."""
    net = make_lbps_reg_kh5_l01(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 5


def test_forward_shape():
    """Network forward returns [B, T, output_size]."""
    net = make_lbps_reg_l01(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_forward_handles_nan():
    """Network forward handles NaN inputs via nan_to_num."""
    net = make_lbps_reg_l01(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_gradient_flows_through_reg():
    """Gradient flows through reg loss to β."""
    net = make_lbps_reg_l01(input_size=2, hidden_size=8, output_size=1)
    # Move β away from target so reg has gradient
    for cell in net.cells:
        cell.beta_x_raw.data.fill_(-1.0)  # sigmoid(-1) ≈ 0.27
    x = torch.randn(2, 16, 2)
    y = net(x)
    y_task = y.mean()
    reg = net.reg_loss()
    total = y_task + reg
    total.backward()
    for cell in net.cells:
        assert cell.beta_x_raw.grad is not None
        assert cell.beta_x_raw.grad.abs().sum().item() > 0


def test_smoke_learns_sin():
    """make_lbps_reg_l01 should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = make_lbps_reg_l01(input_size=2, hidden_size=12, output_size=1)
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
        task_loss = F.mse_loss(out, target)
        reg = net.reg_loss()
        loss = task_loss + reg
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    assert initial_loss is not None
    assert math.isfinite(final_loss)
    assert final_loss < initial_loss


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
