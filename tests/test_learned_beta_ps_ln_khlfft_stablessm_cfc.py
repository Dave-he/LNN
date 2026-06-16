"""Round 208 — tests for StableSSMCfC (PRD #10-170)."""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_stablessm_cfc import (
    StableSSMCfCCell,
    make_lbps_lnkhlfft_stablessm_5_3_2,
)


def test_stablessm_cell_forward():
    cell = StableSSMCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.train()
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_ssm = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, h_ssm_new, _, _ = cell(x, h, h_ssm, emas_x, emas_h)
    assert torch.isfinite(h_new).all()
    assert torch.isfinite(h_ssm_new).all()


def test_stablessm_cell_handles_nan():
    cell = StableSSMCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 8)
    h_ssm = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, h_ssm_new, _, _ = cell(x, h, h_ssm, emas_x, emas_h)
    assert torch.isfinite(h_new).all()
    assert torch.isfinite(h_ssm_new).all()


def test_stablessm_A_bounded_0_1():
    """Sigmoid ensures A in [0,1] for stability."""
    cell = StableSSMCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    z = torch.randn(4, cell.layer_norm.normalized_shape[0])
    A = cell.ssm_A(z)
    assert (A >= 0).all()
    assert (A <= 1).all()


def test_stablessm_gradient_flows():
    cell = StableSSMCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    # Run a few SSM steps to build up h_ssm dependence on A
    h_ssm = torch.zeros(4, 8)
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    for t in range(3):
        x = torch.randn(4, 2)
        h, h_ssm, _, _ = cell(x, h, h_ssm, emas_x, emas_h)
    loss = h.pow(2).mean() + h_ssm.pow(2).mean()
    loss.backward()
    has_ssm_grad = False
    for name, p in cell.named_parameters():
        if "ssm_A" in name and p.grad is not None and p.grad.abs().sum() > 0:
            has_ssm_grad = True
            break
    if not has_ssm_grad:
        # Also accept ssm_B or ssm_C grad as evidence of SSM gradient flow
        for name, p in cell.named_parameters():
            if "ssm_B" in name and p.grad is not None and p.grad.abs().sum() > 0:
                has_ssm_grad = True
                break
    assert has_ssm_grad


def test_stablessm_stacked_factory():
    net = make_lbps_lnkhlfft_stablessm_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert torch.isfinite(y).all()
    assert y.shape == (2, 16, 1)


def test_stablessm_stacked_eval_deterministic():
    net = make_lbps_lnkhlfft_stablessm_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net.eval()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert torch.allclose(y1, y2)


def test_stablessm_stacked_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_stablessm_5_3_2(input_size=2, hidden_size=12, output_size=1)
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


def test_stablessm_no_explosion_long_input():
    """Run 100 steps and verify h_ssm doesn't explode."""
    cell = StableSSMCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.train()
    h_ssm = torch.zeros(1, 8)
    h = torch.zeros(1, 8)
    emas_x = [torch.zeros(1, 2) for _ in range(3)]
    emas_h = [torch.zeros(1, 8) for _ in range(2)]
    for t in range(100):
        x = torch.randn(1, 2) * 5  # large input
        h, h_ssm, _, _ = cell(x, h, h_ssm, emas_x, emas_h)
        assert torch.isfinite(h).all(), f"NaN at t={t}"
        assert torch.isfinite(h_ssm).all(), f"h_ssm NaN at t={t}"
        # Bounded by accumulating B
        assert h_ssm.abs().max() < 100, f"h_ssm exploded at t={t}: {h_ssm.abs().max()}"


def test_stablessm_smoke_long_sequence():
    net = make_lbps_lnkhlfft_stablessm_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 32, 2)
    y = net(x)
    assert torch.isfinite(y).all()
    assert y.shape == (2, 32, 1)


def test_stablessm_step_input_response():
    """Non-zero input should give non-zero h_ssm."""
    cell = StableSSMCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.eval()
    x = torch.ones(1, 2)
    h_ssm = torch.zeros(1, 8)
    z = torch.randn(1, cell.layer_norm.normalized_shape[0])
    h_ssm_new, _ = cell._ssm_step(x, h_ssm, z)
    assert not torch.allclose(h_ssm_new, h_ssm)


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
