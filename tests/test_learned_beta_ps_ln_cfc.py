"""Round 179 — tests for LearnedBetaPS+LN-CfC (PRD #10-141)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ln_cfc import (
    LearnedBetaPSLNCfCCell,
    LearnedBetaPSLNCfCStackedNetwork,
    make_lbps_ln_h3_75,
    make_lbps_ln_h2_75,
    make_lbps_ln_h5_75,
)


def test_cell_init():
    cell = LearnedBetaPSLNCfCCell(input_size=2, hidden_size=8, Kx=5, Kh=3)
    assert cell.Kx == 5
    assert cell.Kh == 3
    assert cell.layer_norm is not None
    assert cell.beta_x_raw.shape == (5,)
    assert cell.beta_h_raw.shape == (3,)


def test_cell_init_default_beta():
    """Initial β is close to 0.75."""
    cell = LearnedBetaPSLNCfCCell(input_size=2, hidden_size=8, Kx=5, Kh=3)
    assert torch.allclose(cell.beta_x, torch.full((5,), 0.75), atol=0.01)
    assert torch.allclose(cell.beta_h, torch.full((3,), 0.75), atol=0.01)


def test_cell_forward_shape():
    cell = LearnedBetaPSLNCfCCell(input_size=2, hidden_size=8, Kx=5, Kh=3)
    x_t = torch.randn(4, 2)
    h_t = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(5)]
    emas_h = [torch.zeros(4, 8) for _ in range(3)]
    h_new, emas_x_new, emas_h_new = cell(x_t, h_t, emas_x, emas_h)
    assert h_new.shape == (4, 8)
    assert len(emas_x_new) == 5
    assert all(e.shape == (4, 2) for e in emas_x_new)


def test_cell_forward_handles_nan():
    cell = LearnedBetaPSLNCfCCell(input_size=2, hidden_size=8, Kx=5, Kh=3)
    x_t = torch.randn(4, 2)
    x_t[0, 0] = float("nan")
    h_t = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(5)]
    emas_h = [torch.zeros(4, 8) for _ in range(3)]
    h_new, emas_x_new, emas_h_new = cell(x_t, h_t, emas_x, emas_h)
    assert torch.isfinite(h_new).all()


def test_factory_h3_75():
    net = make_lbps_ln_h3_75(input_size=2, hidden_size=8, output_size=1)
    assert net.Kx == 5
    assert net.Kh == 3


def test_factory_h2_75():
    net = make_lbps_ln_h2_75(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 2


def test_factory_h5_75():
    net = make_lbps_ln_h5_75(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 5


def test_forward_shape_stacked():
    net = make_lbps_ln_h3_75(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_forward_handles_nan_stacked():
    net = make_lbps_ln_h3_75(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_layer_norm_normalizes():
    """LayerNorm produces zero mean / unit variance (in expectation)."""
    cell = LearnedBetaPSLNCfCCell(input_size=2, hidden_size=8, Kx=5, Kh=3)
    # Manually verify LayerNorm behavior
    x = torch.randn(100, 22)  # combined dim = (5+1)*2 + (3+1)*8 = 12 + 32 = 44... wait
    # Actually: aug_input_size = (5+1)*2 = 12, aug_hidden_size = (3+1)*8 = 32
    # combined = 12 + 32 = 44
    x = torch.randn(100, 44)
    out = cell.layer_norm(x)
    assert abs(out.mean().item()) < 0.01
    assert abs(out.std().item() - 1.0) < 0.01


def test_gradient_flows_all_layers():
    net = make_lbps_ln_h3_75(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.beta_x_raw.grad is not None
        assert cell.beta_h_raw.grad is not None
        assert cell.beta_x_raw.grad.abs().sum().item() > 0
        assert cell.beta_h_raw.grad.abs().sum().item() > 0
        # LayerNorm weight should also have gradient
        assert cell.layer_norm.weight.grad is not None


def test_no_sequences():
    net = LearnedBetaPSLNCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx=5, Kh=3, return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_smoke_learns_sin():
    """make_lbps_ln_h3_75 should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = make_lbps_ln_h3_75(input_size=2, hidden_size=12, output_size=1)
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
