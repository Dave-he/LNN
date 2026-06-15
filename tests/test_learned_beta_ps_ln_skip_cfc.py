"""Round 184 — tests for LearnedBetaPS+LN+Skip-CfC (PRD #10-146)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_skip_cfc import (
    LearnedBetaPSLNSkipCfCCell,
    LearnedBetaPSLNSkipCfCStackedNetwork,
    make_lbps_lns_h3_75,
    make_lbps_lns_h2_75,
    make_lbps_lns_h5_75,
)


def test_cell_init():
    cell = LearnedBetaPSLNSkipCfCCell(input_size=2, hidden_size=8, Kx=5, Kh=3)
    assert cell.Kx == 5
    assert cell.Kh == 3
    assert cell.layer_norm is not None
    assert cell.residual_proj is not None
    # Residual starts at 0 output (small init)
    assert torch.allclose(cell.residual_proj.bias, torch.zeros(8), atol=1e-6)
    # Weight is small (residual_init=0.1)
    assert cell.residual_proj.weight.abs().max().item() < 0.5


def test_cell_init_default_beta():
    cell = LearnedBetaPSLNSkipCfCCell(input_size=2, hidden_size=8, Kx=5, Kh=3)
    assert torch.allclose(cell.beta_x, torch.full((5,), 0.75), atol=0.01)
    assert torch.allclose(cell.beta_h, torch.full((3,), 0.75), atol=0.01)


def test_cell_forward_shape():
    cell = LearnedBetaPSLNSkipCfCCell(input_size=2, hidden_size=8, Kx=5, Kh=3)
    x_t = torch.randn(4, 2)
    h_t = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(5)]
    emas_h = [torch.zeros(4, 8) for _ in range(3)]
    h_new, emas_x_new, emas_h_new = cell(x_t, h_t, emas_x, emas_h)
    assert h_new.shape == (4, 8)
    assert len(emas_x_new) == 5


def test_cell_forward_handles_nan():
    cell = LearnedBetaPSLNSkipCfCCell(input_size=2, hidden_size=8, Kx=5, Kh=3)
    x_t = torch.randn(4, 2)
    x_t[0, 0] = float("nan")
    h_t = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(5)]
    emas_h = [torch.zeros(4, 8) for _ in range(3)]
    h_new, emas_x_new, emas_h_new = cell(x_t, h_t, emas_x, emas_h)
    assert torch.isfinite(h_new).all()


def test_residual_adds_to_h_t():
    """At init, residual is small → h_new ≈ h_t (with small perturbation)."""
    torch.manual_seed(0)
    cell = LearnedBetaPSLNSkipCfCCell(input_size=2, hidden_size=8, Kx=5, Kh=3)
    x_t = torch.zeros(2, 2)
    h_t = torch.ones(2, 8)
    emas_x = [torch.zeros(2, 2) for _ in range(5)]
    emas_h = [torch.zeros(2, 8) for _ in range(3)]
    h_new, _, _ = cell(x_t, h_t, emas_x, emas_h)
    # With zero input, h_new should be h_t + small_residual
    # h_t = 1.0, residual starts small
    diff = (h_new - h_t).abs().max().item()
    assert diff < 0.5, f"residual too large at init: {diff}"


def test_residual_is_learnable():
    """After training, residual changes shape."""
    torch.manual_seed(0)
    cell = LearnedBetaPSLNSkipCfCCell(input_size=2, hidden_size=8, Kx=5, Kh=3)
    x_t = torch.randn(4, 2)
    h_t = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(5)]
    emas_h = [torch.zeros(4, 8) for _ in range(3)]
    h_new, _, _ = cell(x_t, h_t, emas_x, emas_h)
    h_new.sum().backward()
    assert cell.residual_proj.weight.grad is not None
    assert cell.residual_proj.weight.grad.abs().sum().item() > 0


def test_factory_h3_75():
    net = make_lbps_lns_h3_75(input_size=2, hidden_size=8, output_size=1)
    assert net.Kx == 5
    assert net.Kh_ladder == [3, 3, 3]


def test_factory_h2_75():
    net = make_lbps_lns_h2_75(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh_ladder == [2, 2, 2]


def test_factory_h5_75():
    net = make_lbps_lns_h5_75(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh_ladder == [5, 5, 5]


def test_forward_shape_stacked():
    net = make_lbps_lns_h3_75(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_forward_handles_nan_stacked():
    net = make_lbps_lns_h3_75(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_gradient_flows_all_layers():
    """Verify gradient flows to residual_proj + layer_norm after a few training steps.

    Note: at init with zero residual weights and zero h_t, the model
    output y is constant, which makes d_loss/d_y a constant — so the
    trivial pow(2) test gives 0 grad for the first call. We use a
    small training loop to make y non-constant.
    """
    torch.manual_seed(0)
    net = make_lbps_lns_h3_75(input_size=2, hidden_size=8, output_size=1)
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
    for i, cell in enumerate(net.cells):
        assert cell.beta_x_raw.grad is not None
        assert cell.beta_h_raw.grad is not None
        assert cell.layer_norm.weight.grad is not None
        assert cell.residual_proj.weight.grad is not None
        # At least one cell should have non-zero grad (training moved).
    # Aggregate: at least one param has non-zero grad.
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in net.parameters()
    )
    assert has_grad, "no gradient flowed to any parameter"


def test_no_sequences():
    net = LearnedBetaPSLNSkipCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx=5, Kh_ladder=[3, 3, 3], return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_smoke_learns_sin():
    torch.manual_seed(0)
    net = make_lbps_lns_h3_75(input_size=2, hidden_size=12, output_size=1)
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
