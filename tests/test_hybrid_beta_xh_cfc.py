"""Round 163 — tests for HybridBeta-XH-CfC (Scalar β on h + Per-Feature β on x) (PRD #10-125)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.hybrid_beta_xh_cfc import (
    HybridBetaXHCfCCell,
    HybridBetaXHCfCStackedNetwork,
)


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------


def test_cell_init_default():
    """Default Kx=1, Kh=1, scalar β=0.9 on h."""
    cell = HybridBetaXHCfCCell(
        input_size=2, hidden_size=8, Kx=1, Kh=1,
    )
    assert cell.Kx == 1
    assert cell.Kh == 1
    assert cell.betas_h == [0.9]


def test_cell_init_with_betas_h():
    """Kx=2, Kh=2, scalar β ∈ {0.7, 0.95} on h."""
    cell = HybridBetaXHCfCCell(
        input_size=2, hidden_size=8, Kx=2, Kh=2,
        betas_h=[0.7, 0.95],
    )
    assert cell.betas_h == [0.7, 0.95]


def test_cell_init_mismatched_betas_h():
    """Mismatched betas_h length should raise."""
    try:
        HybridBetaXHCfCCell(
            input_size=2, hidden_size=8, Kx=1, Kh=2,
            betas_h=[0.7],  # only 1, but Kh=2
        )
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_init_invalid_mode():
    """Invalid mode should raise."""
    try:
        HybridBetaXHCfCCell(
            input_size=2, hidden_size=8, Kx=1, Kh=1,
            mode_x="invalid", mode_h="diff",
        )
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_init_zero_K():
    """Zero K should raise."""
    try:
        HybridBetaXHCfCCell(
            input_size=2, hidden_size=8, Kx=0, Kh=1,
        )
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_step_shape_diff_3_2():
    """Kx=3 Kh=2 diff step shape."""
    cell = HybridBetaXHCfCCell(
        input_size=2, hidden_size=8, Kx=3, Kh=2,
        betas_h=[0.7, 0.95],
    )
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, emas_x_new, emas_h_new = cell(x, h, emas_x, emas_h)
    assert h_new.shape == (4, 8)
    assert len(emas_x_new) == 3
    for e in emas_x_new:
        assert e.shape == (4, 2)
    assert len(emas_h_new) == 2
    for e in emas_h_new:
        assert e.shape == (4, 8)


def test_cell_step_shape_concat_2_2():
    """Kx=2 Kh=2 concat step shape."""
    cell = HybridBetaXHCfCCell(
        input_size=2, hidden_size=8, Kx=2, Kh=2,
        betas_h=[0.7, 0.95], mode_x="concat", mode_h="concat",
    )
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(2)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, emas_x_new, emas_h_new = cell(x, h, emas_x, emas_h)
    assert h_new.shape == (4, 8)
    assert len(emas_x_new) == 2
    assert len(emas_h_new) == 2


def test_cell_step_finite():
    """Forward output is finite."""
    cell = HybridBetaXHCfCCell(
        input_size=2, hidden_size=8, Kx=3, Kh=2,
        betas_h=[0.7, 0.95],
    )
    x = torch.randn(4, 2) * 5.0
    h = torch.randn(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, emas_x_new, emas_h_new = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()
    for e in emas_x_new:
        assert torch.isfinite(e).all()
    for e in emas_h_new:
        assert torch.isfinite(e).all()


def test_cell_step_handles_nan():
    """NaN input handled."""
    cell = HybridBetaXHCfCCell(
        input_size=2, hidden_size=8, Kx=2, Kh=2,
        betas_h=[0.7, 0.95],
    )
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.randn(4, 8)
    h[1, 0] = float("nan")
    emas_x = [torch.zeros(4, 2) for _ in range(2)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, emas_x_new, emas_h_new = cell(x, h, emas_x, emas_h)
    assert torch.isfinite(h_new).all()
    for e in emas_x_new:
        assert torch.isfinite(e).all()
    for e in emas_h_new:
        assert torch.isfinite(e).all()


def test_cell_ema_x_per_feature():
    """Per-feature β on x-side."""
    cell = HybridBetaXHCfCCell(
        input_size=3, hidden_size=4, Kx=1, Kh=1,
        beta_init=0.0,  # sigmoid(0) = 0.5
    )
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    h = torch.zeros(2, 4)
    ema_x = [torch.zeros(2, 3)]
    ema_h = [torch.zeros(2, 4)]
    _, emas_x_new, _ = cell(x, h, ema_x, ema_h)
    # With β=0.5, ema_new = 0.5 * 0 + 0.5 * x = 0.5 * x
    assert torch.allclose(emas_x_new[0], 0.5 * x, atol=1e-5)


def test_cell_ema_h_scalar():
    """Scalar β on h-side (no per-feature)."""
    cell = HybridBetaXHCfCCell(
        input_size=2, hidden_size=4, Kx=1, Kh=1,
        betas_h=[0.5],  # scalar β=0.5
    )
    x = torch.zeros(2, 2)
    h = torch.ones(2, 4)
    ema_x = [torch.zeros(2, 2)]
    ema_h = [torch.zeros(2, 4)]
    _, _, emas_h_new = cell(x, h, ema_x, ema_h)
    # With β=0.5, ema_h_new = 0.5 * 0 + 0.5 * 1 = 0.5
    assert torch.allclose(emas_h_new[0], 0.5 * h, atol=1e-5)


def test_cell_ema_h_scalar_2():
    """K=2 scalar β ∈ {0.7, 0.95} on h-side."""
    cell = HybridBetaXHCfCCell(
        input_size=2, hidden_size=4, Kx=1, Kh=2,
        betas_h=[0.7, 0.95],
    )
    x = torch.zeros(2, 2)
    h = torch.ones(2, 4)
    ema_x = [torch.zeros(2, 2)]
    ema_h = [torch.zeros(2, 4), torch.zeros(2, 4)]
    _, _, emas_h_new = cell(x, h, ema_x, ema_h)
    # ema_h_new[0] = 0.7 * 0 + 0.3 * 1 = 0.3
    # ema_h_new[1] = 0.95 * 0 + 0.05 * 1 = 0.05
    assert torch.allclose(emas_h_new[0], 0.3 * h, atol=1e-5)
    assert torch.allclose(emas_h_new[1], 0.05 * h, atol=1e-5)


def test_cell_gradient_flows_beta_x():
    """Gradient reaches beta_x parameters (learned)."""
    cell = HybridBetaXHCfCCell(
        input_size=2, hidden_size=8, Kx=2, Kh=2,
        betas_h=[0.7, 0.95],
    )
    x = torch.randn(4, 2)
    h = torch.randn(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(2)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    h_new.sum().backward()
    assert cell.beta_x_raw.grad is not None
    assert cell.beta_x_raw.grad.abs().sum().item() > 0


def test_cell_no_h_beta_params():
    """No h-side β parameters (betas_h is fixed, not learnable)."""
    cell = HybridBetaXHCfCCell(
        input_size=2, hidden_size=8, Kx=2, Kh=2,
        betas_h=[0.7, 0.95],
    )
    # betas_h is a list, not a parameter
    assert not hasattr(cell, "beta_h_raw")
    assert isinstance(cell.betas_h, list)


def test_cell_gradient_flows_cfc():
    """Gradient reaches CfC weights."""
    cell = HybridBetaXHCfCCell(
        input_size=2, hidden_size=8, Kx=3, Kh=2,
        betas_h=[0.7, 0.95],
    )
    x = torch.randn(4, 2)
    h = torch.randn(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    h_new.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_cell_per_feature_beta_shape():
    """beta_x has shape [Kx, D]."""
    cell = HybridBetaXHCfCCell(
        input_size=3, hidden_size=5, Kx=2, Kh=4,
        betas_h=[0.7, 0.95, 0.99, 0.5],
    )
    assert cell.beta_x.shape == (2, 3)
    assert cell.betas_h == [0.7, 0.95, 0.99, 0.5]


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network."""
    net = HybridBetaXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        Kx=2, Kh=2, betas_h=[0.7, 0.95],
    )
    assert net.num_layers == 2
    assert net.Kx == 2
    assert net.Kh == 2
    assert net.betas_h == [0.7, 0.95]


def test_stacked_forward_shape_diff_3_2():
    """Forward returns [B, T, output_size] (diff 3/2)."""
    net = HybridBetaXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        Kx=3, Kh=2, betas_h=[0.7, 0.95], return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_concat_2_2():
    """Forward returns [B, T, output_size] (concat 2/2)."""
    net = HybridBetaXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        Kx=2, Kh=2, betas_h=[0.7, 0.95],
        mode_x="concat", mode_h="concat",
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = HybridBetaXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        Kx=2, Kh=2, betas_h=[0.7, 0.95], return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_stacked_forward_finite_with_nan():
    """Forward handles NaN inputs."""
    net = HybridBetaXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        Kx=2, Kh=2, betas_h=[0.7, 0.95],
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers_and_betas():
    """Gradient reaches all layers' CfC weights AND beta parameters."""
    net = HybridBetaXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        Kx=2, Kh=2, betas_h=[0.7, 0.95],
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.f_gate[0].weight.grad is not None
        assert cell.f_gate[0].weight.grad.abs().sum().item() > 0
        assert cell.beta_x_raw.grad is not None


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_learns_sin():
    """Smoke: HybridBeta-XH-CfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = HybridBetaXHCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2,
        Kx=2, Kh=2, betas_h=[0.7, 0.95],
    )
    t = torch.linspace(0, 4 * math.pi, 16).unsqueeze(0).unsqueeze(-1).expand(2, 16, 1)
    x = torch.zeros(2, 16, 2)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    target = torch.sin(t.squeeze(-1)).unsqueeze(-1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    initial_loss = None
    final_loss = None
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
    assert math.isfinite(final_loss)
    assert final_loss < 5.0
    assert final_loss < initial_loss


# ---------------------------------------------------------------------------
# Bench-style smoke test
# ---------------------------------------------------------------------------


def test_bench_smoke_hb_xh_vs_cfc():
    """Mini-bench: HybridBeta-XH-CfC vs CfC baseline on sin task."""
    from lnn.core.cfc import CfCNetwork

    B, T, D, H = 4, 16, 2, 8
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
    x = torch.zeros(B, T, D)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    y = torch.sin(t.squeeze(-1)).unsqueeze(-1).expand(B, T, 1)

    # CfC baseline.
    torch.manual_seed(42)
    cfc = CfCNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2, return_sequences=True,
    )
    opt = torch.optim.Adam(cfc.parameters(), lr=1e-2)
    cfc_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = cfc(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        cfc_loss = loss.item()

    # HybridBeta-XH-CfC.
    torch.manual_seed(42)
    hb_xh = HybridBetaXHCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        Kx=2, Kh=2, betas_h=[0.7, 0.95], return_sequences=True,
    )
    opt = torch.optim.Adam(hb_xh.parameters(), lr=1e-2)
    hb_xh_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = hb_xh(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        hb_xh_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(hb_xh_loss)


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
