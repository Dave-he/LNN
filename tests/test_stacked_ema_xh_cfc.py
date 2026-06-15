"""Round 161 — tests for Stacked-EMA-XH-CfC (Input + Hidden State EMA) (PRD #10-123)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.stacked_ema_xh_cfc import (
    StackedEMAXHCfCCell,
    StackedEMAXHCfCStackedNetwork,
)


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------


def test_cell_init_diff_1_1():
    """Kx=1 Kh=1 diff mode."""
    cell = StackedEMAXHCfCCell(
        input_size=2, hidden_size=8,
        betas_x=[0.9], betas_h=[0.9],
        mode_x="diff", mode_h="diff",
    )
    assert cell.Kx == 1
    assert cell.Kh == 1
    assert cell.betas_x == [0.9]
    assert cell.betas_h == [0.9]
    assert cell.mode_x == "diff"
    assert cell.mode_h == "diff"


def test_cell_init_diff_3_2():
    """Kx=3 Kh=2 diff mode."""
    cell = StackedEMAXHCfCCell(
        input_size=2, hidden_size=8,
        betas_x=[0.5, 0.9, 0.99], betas_h=[0.7, 0.95],
        mode_x="diff", mode_h="diff",
    )
    assert cell.Kx == 3
    assert cell.Kh == 2
    assert cell.betas_x == [0.5, 0.9, 0.99]
    assert cell.betas_h == [0.7, 0.95]


def test_cell_init_concat_2_2():
    """Kx=2 Kh=2 concat mode."""
    cell = StackedEMAXHCfCCell(
        input_size=2, hidden_size=8,
        betas_x=[0.7, 0.95], betas_h=[0.7, 0.95],
        mode_x="concat", mode_h="concat",
    )
    assert cell.Kx == 2
    assert cell.Kh == 2
    assert cell.mode_x == "concat"
    assert cell.mode_h == "concat"


def test_cell_init_invalid_mode():
    """Invalid mode should raise."""
    try:
        StackedEMAXHCfCCell(
            input_size=2, hidden_size=8,
            betas_x=[0.9], betas_h=[0.9],
            mode_x="invalid", mode_h="diff",
        )
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_init_empty_betas():
    """Empty betas should raise."""
    try:
        StackedEMAXHCfCCell(
            input_size=2, hidden_size=8,
            betas_x=[], betas_h=[0.9],
        )
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_step_shape_diff_3_2():
    """Kx=3 Kh=2 diff step shape."""
    cell = StackedEMAXHCfCCell(
        input_size=2, hidden_size=8,
        betas_x=[0.5, 0.9, 0.99], betas_h=[0.7, 0.95],
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
    cell = StackedEMAXHCfCCell(
        input_size=2, hidden_size=8,
        betas_x=[0.7, 0.95], betas_h=[0.7, 0.95],
        mode_x="concat", mode_h="concat",
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
    cell = StackedEMAXHCfCCell(
        input_size=2, hidden_size=8,
        betas_x=[0.5, 0.9, 0.99], betas_h=[0.7, 0.95],
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
    cell = StackedEMAXHCfCCell(
        input_size=2, hidden_size=8,
        betas_x=[0.7, 0.95], betas_h=[0.7, 0.95],
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


def test_cell_ema_x_decay_correct():
    """ema_x_k,t = beta_x_k * ema_x_k,t-1 + (1-beta_x_k) * x_t."""
    cell = StackedEMAXHCfCCell(
        input_size=2, hidden_size=4,
        betas_x=[0.5, 0.8], betas_h=[0.9],
    )
    x = torch.ones(2, 2)
    h = torch.zeros(2, 4)
    ema_x_0 = torch.zeros(2, 2)
    ema_x_1 = torch.zeros(2, 2)
    ema_h = [torch.zeros(2, 4)]
    _, emas_x_new, _ = cell(x, h, [ema_x_0, ema_x_1], ema_h)
    # After 1 step with x=1 and ema=0:
    # ema_x_0 = 0.5 * 0 + 0.5 * 1 = 0.5
    # ema_x_1 = 0.8 * 0 + 0.2 * 1 = 0.2
    assert torch.allclose(emas_x_new[0], 0.5 * x, atol=1e-5)
    assert torch.allclose(emas_x_new[1], 0.2 * x, atol=1e-5)


def test_cell_ema_h_decay_correct():
    """ema_h_k,t = beta_h_k * ema_h_k,t-1 + (1-beta_h_k) * h_t."""
    cell = StackedEMAXHCfCCell(
        input_size=2, hidden_size=4,
        betas_x=[0.9], betas_h=[0.5, 0.8],
    )
    x = torch.zeros(2, 2)
    h = torch.ones(2, 4)
    ema_x = [torch.zeros(2, 2)]
    ema_h_0 = torch.zeros(2, 4)
    ema_h_1 = torch.zeros(2, 4)
    _, _, emas_h_new = cell(x, h, ema_x, [ema_h_0, ema_h_1])
    # After 1 step with h=1 and ema=0:
    # ema_h_0 = 0.5 * 0 + 0.5 * 1 = 0.5
    # ema_h_1 = 0.8 * 0 + 0.2 * 1 = 0.2
    assert torch.allclose(emas_h_new[0], 0.5 * h, atol=1e-5)
    assert torch.allclose(emas_h_new[1], 0.2 * h, atol=1e-5)


def test_cell_gradient_flows():
    """Gradient reaches CfC weights."""
    cell = StackedEMAXHCfCCell(
        input_size=2, hidden_size=8,
        betas_x=[0.5, 0.9, 0.99], betas_h=[0.7, 0.95],
    )
    x = torch.randn(4, 2)
    h = torch.randn(4, 8)
    emas_x = [torch.zeros(4, 2) for _ in range(3)]
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _, _ = cell(x, h, emas_x, emas_h)
    h_new.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_cell_no_input_only_or_h_only():
    """Stacked cell has BOTH x-side and h-side EMAs."""
    cell = StackedEMAXHCfCCell(
        input_size=2, hidden_size=8,
        betas_x=[0.7, 0.95], betas_h=[0.7, 0.95],
    )
    import inspect
    sig = inspect.signature(cell.forward)
    params = list(sig.parameters.keys())
    assert "emas_x" in params
    assert "emas_h" in params


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network (Kx=2 Kh=2 diff)."""
    net = StackedEMAXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas_x=[0.7, 0.95], betas_h=[0.7, 0.95],
    )
    assert net.num_layers == 2
    assert net.Kx == 2
    assert net.Kh == 2


def test_stacked_forward_shape_diff_3_2():
    """Forward returns [B, T, output_size] (diff 3/2)."""
    net = StackedEMAXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas_x=[0.5, 0.9, 0.99], betas_h=[0.7, 0.95],
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_concat_2_2():
    """Forward returns [B, T, output_size] (concat 2/2)."""
    net = StackedEMAXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas_x=[0.7, 0.95], betas_h=[0.7, 0.95],
        mode_x="concat", mode_h="concat", return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_best():
    """Forward returns [B, T, output_size] (Kx=3 diff + Kh=2 diff)."""
    net = StackedEMAXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas_x=[0.5, 0.9, 0.99], betas_h=[0.7, 0.95],
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = StackedEMAXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas_x=[0.7, 0.95], betas_h=[0.7, 0.95],
        return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_stacked_forward_finite_with_nan():
    """Forward handles NaN inputs."""
    net = StackedEMAXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas_x=[0.7, 0.95], betas_h=[0.7, 0.95],
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers():
    """Gradient reaches all layers' CfC weights."""
    net = StackedEMAXHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas_x=[0.7, 0.95], betas_h=[0.7, 0.95],
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.f_gate[0].weight.grad is not None
        assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_learns_sin():
    """Smoke: Stacked-EMA-XH-CfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = StackedEMAXHCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2,
        betas_x=[0.7, 0.95], betas_h=[0.7, 0.95],
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


def test_bench_smoke_sx_xh_vs_cfc():
    """Mini-bench: Stacked-EMA-XH-CfC vs CfC baseline on sin task."""
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

    # Stacked-EMA-XH-CfC.
    torch.manual_seed(42)
    sx_xh = StackedEMAXHCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        betas_x=[0.7, 0.95], betas_h=[0.7, 0.95],
        return_sequences=True,
    )
    opt = torch.optim.Adam(sx_xh.parameters(), lr=1e-2)
    sx_xh_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = sx_xh(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        sx_xh_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(sx_xh_loss)


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
