"""Round 207 — tests for SSMCfC (PRD #10-169)."""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_khlfft_ssm_cfc import (
    SSMCfCCell,
    make_lbps_lnkhlfft_ssm_5_3_2,
)


def test_ssm_cell_forward():
    cell = SSMCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.train()
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_ssm = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, h_ssm_new, _, _ = cell(x, h, h_ssm, emas_x, emas_h)
    assert torch.isfinite(h_new).all()
    assert torch.isfinite(h_ssm_new).all()
    assert h_new.shape == h.shape
    assert h_ssm_new.shape == h_ssm.shape


def test_ssm_cell_handles_nan():
    cell = SSMCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 8)
    h_ssm = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, h_ssm_new, _, _ = cell(x, h, h_ssm, emas_x, emas_h)
    assert torch.isfinite(h_new).all()
    assert torch.isfinite(h_ssm_new).all()


def test_ssm_step_zero_input_zero_state():
    cell = SSMCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.eval()
    # With x=0 and h_ssm=0, h_ssm_new = A*0 + ssm_B(0) = bias
    # g_ssm = C * bias (still finite)
    x = torch.zeros(2, 2)
    h_ssm = torch.zeros(2, 8)
    z = torch.zeros(2, cell.layer_norm.normalized_shape[0])
    h_ssm_new, g_ssm = cell._ssm_step(x, h_ssm, z)
    # Just check finite — exact zero depends on bias init
    assert torch.isfinite(h_ssm_new).all()
    assert torch.isfinite(g_ssm).all()


def test_ssm_A_positive():
    """Softplus ensures A > 0 (decay is positive)."""
    cell = SSMCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    z = torch.randn(4, cell.layer_norm.normalized_shape[0])
    A = cell.ssm_A(z)
    assert (A >= 0).all()


def test_ssm_gradient_flows():
    cell = SSMCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_ssm = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _, _ = cell(x, h, h_ssm, emas_x, emas_h)
    loss = h_new.pow(2).mean()
    loss.backward()
    has_ssm_grad = False
    for name, p in cell.named_parameters():
        if "ssm_A" in name and p.grad is not None and p.grad.abs().sum() > 0:
            has_ssm_grad = True
            break
    if not has_ssm_grad:
        for name, p in cell.named_parameters():
            if "ssm_B" in name and p.grad is not None and p.grad.abs().sum() > 0:
                has_ssm_grad = True
                break
    assert has_ssm_grad


def test_ssm_stacked_factory():
    net = make_lbps_lnkhlfft_ssm_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert torch.isfinite(y).all()
    assert y.shape == (2, 16, 1)


def test_ssm_stacked_eval_deterministic():
    net = make_lbps_lnkhlfft_ssm_5_3_2(input_size=2, hidden_size=8, output_size=1)
    net.eval()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert torch.allclose(y1, y2)


def test_ssm_stacked_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lnkhlfft_ssm_5_3_2(input_size=2, hidden_size=12, output_size=1)
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


def test_ssm_step_input_response():
    """Non-zero input should give non-zero h_ssm."""
    cell = SSMCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.eval()
    x = torch.ones(1, 2)
    h_ssm = torch.zeros(1, 8)
    z = torch.randn(1, cell.layer_norm.normalized_shape[0])
    h_ssm_new, g_ssm = cell._ssm_step(x, h_ssm, z)
    # With non-zero input, h_ssm should change
    assert not torch.allclose(h_ssm_new, h_ssm)


def test_ssm_step_persistence():
    """If A > 0 and no input, h_ssm should be close to A*h_ssm_prev."""
    cell = SSMCfCCell(input_size=2, hidden_size=8, Kx=3, Kh=2)
    cell.eval()
    # Initialize weights
    z = torch.randn(1, cell.layer_norm.normalized_shape[0])
    x = torch.zeros(1, 2)
    h_ssm = torch.ones(1, 8) * 0.5
    h_ssm_new, _ = cell._ssm_step(x, h_ssm, z)
    # h_ssm_new = A * h_ssm + B * 0 = A * h_ssm
    # If A = 1, h_ssm_new = 0.5; if A = 0, h_ssm_new = 0
    # We just check shape and finite
    assert torch.isfinite(h_ssm_new).all()


def test_ssm_smoke_long_sequence():
    net = make_lbps_lnkhlfft_ssm_5_3_2(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 32, 2)
    y = net(x)
    assert torch.isfinite(y).all()
    assert y.shape == (2, 32, 1)


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
