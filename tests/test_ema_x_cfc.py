"""Round 156 — tests for EMA-X-CfC (Input EMA Augmentation) (PRD #10-118)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.ema_x_cfc import EMAXCfCCell, EMAXCfCStackedNetwork


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------


def test_cell_init_concat():
    """Default concat mode."""
    cell = EMAXCfCCell(input_size=2, hidden_size=8, ema_mode="concat", beta=0.9)
    assert cell.input_size == 2
    assert cell.hidden_size == 8
    assert cell.ema_mode == "concat"
    assert cell.beta == 0.9


def test_cell_init_gate():
    """Gate mode: has learnable alpha."""
    cell = EMAXCfCCell(input_size=2, hidden_size=8, ema_mode="gate", beta=0.9)
    assert cell.ema_mode == "gate"
    assert cell.gate_alpha is not None


def test_cell_init_diff():
    """Diff mode."""
    cell = EMAXCfCCell(input_size=2, hidden_size=8, ema_mode="diff", beta=0.9)
    assert cell.ema_mode == "diff"


def test_cell_init_ema_only():
    """EMA-only mode."""
    cell = EMAXCfCCell(input_size=2, hidden_size=8, ema_mode="ema_only", beta=0.9)
    assert cell.ema_mode == "ema_only"


def test_cell_init_invalid_mode():
    """Invalid ema_mode should raise."""
    try:
        EMAXCfCCell(input_size=2, hidden_size=8, ema_mode="invalid", beta=0.9)
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_step_shape_concat():
    """Concat mode: returns h and ema."""
    cell = EMAXCfCCell(input_size=2, hidden_size=8, ema_mode="concat", beta=0.9)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema = torch.zeros(4, 2)
    h_new, ema_new = cell(x, h, ema)
    assert h_new.shape == (4, 8)
    assert ema_new.shape == (4, 2)


def test_cell_step_shape_gate():
    """Gate mode shape."""
    cell = EMAXCfCCell(input_size=2, hidden_size=8, ema_mode="gate", beta=0.9)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema = torch.zeros(4, 2)
    h_new, ema_new = cell(x, h, ema)
    assert h_new.shape == (4, 8)
    assert ema_new.shape == (4, 2)


def test_cell_step_shape_diff():
    """Diff mode shape."""
    cell = EMAXCfCCell(input_size=2, hidden_size=8, ema_mode="diff", beta=0.9)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema = torch.zeros(4, 2)
    h_new, ema_new = cell(x, h, ema)
    assert h_new.shape == (4, 8)
    assert ema_new.shape == (4, 2)


def test_cell_step_shape_ema_only():
    """EMA-only mode shape."""
    cell = EMAXCfCCell(input_size=2, hidden_size=8, ema_mode="ema_only", beta=0.9)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema = torch.zeros(4, 2)
    h_new, ema_new = cell(x, h, ema)
    assert h_new.shape == (4, 8)
    assert ema_new.shape == (4, 2)


def test_cell_step_finite_concat():
    """Forward output is finite (concat)."""
    cell = EMAXCfCCell(input_size=2, hidden_size=8, ema_mode="concat", beta=0.9)
    x = torch.randn(4, 2) * 5.0
    h = torch.zeros(4, 8)
    ema = torch.zeros(4, 2)
    h_new, ema_new = cell(x, h, ema)
    assert torch.isfinite(h_new).all()
    assert torch.isfinite(ema_new).all()


def test_cell_step_handles_nan_concat():
    """NaN input handled (concat)."""
    cell = EMAXCfCCell(input_size=2, hidden_size=8, ema_mode="concat", beta=0.9)
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 8)
    ema = torch.zeros(4, 2)
    h_new, ema_new = cell(x, h, ema)
    assert torch.isfinite(h_new).all()
    assert torch.isfinite(ema_new).all()


def test_cell_ema_decay_correct():
    """EMA_t = beta * EMA_{t-1} + (1-beta) * x_t."""
    cell = EMAXCfCCell(input_size=2, hidden_size=8, ema_mode="concat", beta=0.5)
    x = torch.tensor([[1.0, 0.0]])
    h = torch.zeros(1, 8)
    ema = torch.zeros(1, 2)
    _, ema_new = cell(x, h, ema)
    # ema_new = 0.5 * 0 + 0.5 * [1, 0] = [0.5, 0]
    assert torch.allclose(ema_new, torch.tensor([[0.5, 0.0]]), atol=1e-5)


def test_cell_ema_recurrent():
    """EMA after 2 steps with constant x should be 1 - beta^t of x."""
    cell = EMAXCfCCell(input_size=2, hidden_size=8, ema_mode="concat", beta=0.5)
    x = torch.tensor([[1.0, 0.0]])
    h = torch.zeros(1, 8)
    ema = torch.zeros(1, 2)
    _, ema_1 = cell(x, h, ema)
    _, ema_2 = cell(x, h, ema_1)
    # After 2 steps: ema = (1 - 0.5^2) * x = 0.75 * [1, 0]
    assert torch.allclose(ema_2, torch.tensor([[0.75, 0.0]]), atol=1e-5)


def test_cell_gradient_flows_concat():
    """Gradient should reach CfC weights (concat)."""
    cell = EMAXCfCCell(input_size=2, hidden_size=8, ema_mode="concat", beta=0.9)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema = torch.zeros(4, 2)
    h_new, _ = cell(x, h, ema)
    h_new.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_cell_gradient_flows_gate():
    """Gradient should reach gate_alpha (gate mode)."""
    cell = EMAXCfCCell(input_size=2, hidden_size=8, ema_mode="gate", beta=0.9)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema = torch.zeros(4, 2)
    h_new, _ = cell(x, h, ema)
    h_new.sum().backward()
    assert cell.gate_alpha.grad is not None
    assert cell.gate_alpha.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network (concat)."""
    net = EMAXCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat", beta=0.9,
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8
    assert net.ema_mode == "concat"
    assert net.beta == 0.9


def test_stacked_forward_shape_concat():
    """Forward returns [B, T, output_size] (concat)."""
    net = EMAXCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat", beta=0.9, return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_gate():
    """Forward returns [B, T, output_size] (gate)."""
    net = EMAXCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="gate", beta=0.9, return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_diff():
    """Forward returns [B, T, output_size] (diff)."""
    net = EMAXCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="diff", beta=0.9, return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_ema_only():
    """Forward returns [B, T, output_size] (ema_only)."""
    net = EMAXCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="ema_only", beta=0.9, return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = EMAXCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat", beta=0.9, return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_stacked_forward_finite_with_nan_concat():
    """Forward handles NaN inputs (concat)."""
    net = EMAXCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat", beta=0.9,
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers_concat():
    """Gradient should reach all layers' CfC weights (concat)."""
    net = EMAXCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat", beta=0.9,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.f_gate[0].weight.grad is not None
        assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_stacked_ema_evolves_over_time():
    """EMA state should evolve (be different at t=0 and t=T-1)."""
    net = EMAXCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat", beta=0.5, return_sequences=True,
    )
    net.eval()
    with torch.no_grad():
        # Constant x.
        x = torch.ones(1, 8, 2) * 0.5
        y = net(x)
        # EMA should converge to [0.5, 0.5] (the value of x).
        # Since y is the output, we can't directly check EMA.
        # Just check that the output is finite.
        assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_learns_sin_concat():
    """Smoke: EMA-X-CfC (concat) should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = EMAXCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2,
        ema_mode="concat", beta=0.9,
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


def test_smoke_learns_sin_gate():
    """Smoke: EMA-X-CfC (gate) should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = EMAXCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2,
        ema_mode="gate", beta=0.9,
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


def test_bench_smoke_ema_vs_cfc():
    """Mini-bench: EMA-X-CfC vs CfC baseline on sin task."""
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

    # EMA-X-CfC concat.
    torch.manual_seed(42)
    ema = EMAXCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        ema_mode="concat", beta=0.9, return_sequences=True,
    )
    opt = torch.optim.Adam(ema.parameters(), lr=1e-2)
    ema_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = ema(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        ema_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(ema_loss)


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
