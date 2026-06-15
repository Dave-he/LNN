"""Round 178 — tests for LearnedBetaPS+IC-CfC (PRD #10-140)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_ps_ic_cfc import (
    LearnedBetaPSICfCCell,
    LearnedBetaPSICfCStackedNetwork,
    make_lbps_ic_h3_75,
    make_lbps_ic_h2_75,
    make_lbps_ic_h5_75,
    make_lbps_ic_h3_50,
    make_lbps_ic_h3_90,
)


def test_cell_init():
    """Cell initializes with Kx, Kh."""
    cell = LearnedBetaPSICfCCell(input_size=2, hidden_size=8, Kx=5, Kh=3)
    assert cell.Kx == 5
    assert cell.Kh == 3
    # beta_x_proj has shape [Kx, input_size]
    assert cell.beta_x_proj.weight.shape == (5, 2)
    assert cell.beta_x_proj.bias.shape == (5,)
    assert cell.beta_h_proj.weight.shape == (3, 8)
    assert cell.beta_h_proj.bias.shape == (3,)


def test_beta_init_at_target():
    """Initial β is close to beta_init for zero input."""
    cell = LearnedBetaPSICfCCell(
        input_size=2, hidden_size=8, Kx=5, Kh=3,
        beta_x_init=0.75, beta_h_init=0.75,
    )
    x_zero = torch.zeros(2, 2)  # [B, input_size]
    beta_x = torch.sigmoid(cell.beta_x_proj(x_zero))
    # All batches should give same value at init (zero weight)
    assert torch.allclose(beta_x, torch.full((2, 5), 0.75), atol=0.01)


def test_cell_forward_shape():
    """Cell forward returns expected shapes."""
    cell = LearnedBetaPSICfCCell(input_size=2, hidden_size=8, Kx=5, Kh=3)
    x_t = torch.randn(4, 2)
    h_t = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(5)]
    emas_h = [torch.zeros(4, 8) for _ in range(3)]
    h_new, emas_x_new, emas_h_new = cell(x_t, h_t, emas_x, emas_h)
    assert h_new.shape == (4, 8)
    assert len(emas_x_new) == 5
    assert all(e.shape == (4, 2) for e in emas_x_new)
    assert len(emas_h_new) == 3
    assert all(e.shape == (4, 8) for e in emas_h_new)


def test_cell_forward_handles_nan():
    """Cell handles NaN inputs."""
    cell = LearnedBetaPSICfCCell(input_size=2, hidden_size=8, Kx=5, Kh=3)
    x_t = torch.randn(4, 2)
    x_t[0, 0] = float("nan")
    h_t = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(5)]
    emas_h = [torch.zeros(4, 8) for _ in range(3)]
    h_new, emas_x_new, emas_h_new = cell(x_t, h_t, emas_x, emas_h)
    assert torch.isfinite(h_new).all()
    assert all(torch.isfinite(e).all() for e in emas_x_new)


def test_input_dependent_beta():
    """Different inputs give different β values."""
    cell = LearnedBetaPSICfCCell(input_size=2, hidden_size=8, Kx=5, Kh=3)
    # Set non-zero weights so input affects β
    nn_init = torch.nn.init.normal_
    nn_init(cell.beta_x_proj.weight, std=0.1)
    x_t = torch.randn(4, 2)
    h_t = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(5)]
    emas_h = [torch.zeros(4, 8) for _ in range(3)]
    h_new, emas_x_new, emas_h_new = cell(x_t, h_t, emas_x, emas_h)
    # emas_x_new should differ across batches (per-sample β)
    assert not torch.allclose(emas_x_new[0][0], emas_x_new[0][1])


def test_factory_h3_75():
    """Factory h3_75 has Kx=5, Kh=3."""
    net = make_lbps_ic_h3_75(input_size=2, hidden_size=8, output_size=1)
    assert net.Kx == 5
    assert net.Kh == 3
    assert len(net.cells) == 3


def test_factory_h2_75():
    """Factory h2_75 has Kh=2."""
    net = make_lbps_ic_h2_75(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 2


def test_factory_h5_75():
    """Factory h5_75 has Kh=5."""
    net = make_lbps_ic_h5_75(input_size=2, hidden_size=8, output_size=1)
    assert net.Kh == 5


def test_forward_shape_stacked():
    """Stacked network forward returns [B, T, output_size]."""
    net = make_lbps_ic_h3_75(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_forward_handles_nan_stacked():
    """Stacked network handles NaN inputs."""
    net = make_lbps_ic_h3_75(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_gradient_flows_all_layers():
    """Gradient reaches β_x_proj and β_h_proj in all 3 layers."""
    net = make_lbps_ic_h3_75(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.beta_x_proj.weight.grad is not None
        assert cell.beta_x_proj.bias.grad is not None
        assert cell.beta_h_proj.weight.grad is not None
        assert cell.beta_h_proj.bias.grad is not None
        # Non-zero gradient on weight (proves input-conditioning matters)
        assert cell.beta_x_proj.weight.grad.abs().sum().item() > 0


def test_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = LearnedBetaPSICfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx=5, Kh=3, return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_smoke_learns_sin():
    """make_lbps_ic_h3_75 should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = make_lbps_ic_h3_75(input_size=2, hidden_size=12, output_size=1)
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
