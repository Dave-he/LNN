"""Round 159 — tests for EMA-H-CfC (Hidden State EMA Augmentation) (PRD #10-121)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.ema_h_cfc import EMAHCfCCell, EMAHCfCStackedNetwork


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------


def test_cell_init_concat():
    """Default concat mode."""
    cell = EMAHCfCCell(input_size=2, hidden_size=8, ema_mode="concat", beta=0.9)
    assert cell.input_size == 2
    assert cell.hidden_size == 8
    assert cell.ema_mode == "concat"
    assert cell.beta == 0.9


def test_cell_init_gate():
    """Gate mode: has learnable alpha."""
    cell = EMAHCfCCell(input_size=2, hidden_size=8, ema_mode="gate", beta=0.9)
    assert cell.ema_mode == "gate"
    assert cell.gate_alpha is not None


def test_cell_init_diff():
    """Diff mode."""
    cell = EMAHCfCCell(input_size=2, hidden_size=8, ema_mode="diff", beta=0.9)
    assert cell.ema_mode == "diff"


def test_cell_init_ema_only():
    """EMA-only mode."""
    cell = EMAHCfCCell(input_size=2, hidden_size=8, ema_mode="ema_only", beta=0.9)
    assert cell.ema_mode == "ema_only"


def test_cell_init_invalid_mode():
    """Invalid ema_mode should raise."""
    try:
        EMAHCfCCell(input_size=2, hidden_size=8, ema_mode="invalid", beta=0.9)
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_step_shape_concat():
    """Concat mode: returns h and ema_h."""
    cell = EMAHCfCCell(input_size=2, hidden_size=8, ema_mode="concat", beta=0.9)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema_h = torch.zeros(4, 8)
    h_new, ema_h_new = cell(x, h, ema_h)
    assert h_new.shape == (4, 8)
    assert ema_h_new.shape == (4, 8)


def test_cell_step_shape_gate():
    """Gate mode shape."""
    cell = EMAHCfCCell(input_size=2, hidden_size=8, ema_mode="gate", beta=0.9)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema_h = torch.zeros(4, 8)
    h_new, ema_h_new = cell(x, h, ema_h)
    assert h_new.shape == (4, 8)
    assert ema_h_new.shape == (4, 8)


def test_cell_step_shape_diff():
    """Diff mode shape."""
    cell = EMAHCfCCell(input_size=2, hidden_size=8, ema_mode="diff", beta=0.9)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema_h = torch.zeros(4, 8)
    h_new, ema_h_new = cell(x, h, ema_h)
    assert h_new.shape == (4, 8)
    assert ema_h_new.shape == (4, 8)


def test_cell_step_shape_ema_only():
    """EMA-only mode shape."""
    cell = EMAHCfCCell(input_size=2, hidden_size=8, ema_mode="ema_only", beta=0.9)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema_h = torch.zeros(4, 8)
    h_new, ema_h_new = cell(x, h, ema_h)
    assert h_new.shape == (4, 8)
    assert ema_h_new.shape == (4, 8)


def test_cell_step_finite_concat():
    """Forward output is finite (concat)."""
    cell = EMAHCfCCell(input_size=2, hidden_size=8, ema_mode="concat", beta=0.9)
    x = torch.randn(4, 2) * 5.0
    h = torch.zeros(4, 8)
    ema_h = torch.zeros(4, 8)
    h_new, ema_h_new = cell(x, h, ema_h)
    assert torch.isfinite(h_new).all()
    assert torch.isfinite(ema_h_new).all()


def test_cell_step_handles_nan_concat():
    """NaN input handled (concat)."""
    cell = EMAHCfCCell(input_size=2, hidden_size=8, ema_mode="concat", beta=0.9)
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 8)
    ema_h = torch.zeros(4, 8)
    h_new, ema_h_new = cell(x, h, ema_h)
    assert torch.isfinite(h_new).all()
    assert torch.isfinite(ema_h_new).all()


def test_cell_ema_decay_correct():
    """EMA_h_t = beta * EMA_h_{t-1} + (1-beta) * h_t."""
    cell = EMAHCfCCell(input_size=2, hidden_size=8, ema_mode="concat", beta=0.5)
    x = torch.randn(2, 2)
    h = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    ema_h = torch.zeros(2, 8)
    _, ema_h_new = cell(x, h, ema_h)
    # ema_h_new = 0.5 * 0 + 0.5 * h = 0.5 * h
    assert torch.allclose(ema_h_new, 0.5 * h, atol=1e-5)


def test_cell_ema_recurrent():
    """EMA_h after 2 steps with constant h should be 1 - beta^t of h."""
    cell = EMAHCfCCell(input_size=2, hidden_size=4, ema_mode="concat", beta=0.5)
    x = torch.randn(2, 2)
    h = torch.ones(2, 4)
    ema_h = torch.zeros(2, 4)
    _, ema_h_1 = cell(x, h, ema_h)
    _, ema_h_2 = cell(x, h, ema_h_1)
    # After 2 steps: ema_h = (1 - 0.5^2) * h = 0.75 * h
    assert torch.allclose(ema_h_2, 0.75 * h, atol=1e-5)


def test_cell_gradient_flows_concat():
    """Gradient should reach CfC weights (concat)."""
    cell = EMAHCfCCell(input_size=2, hidden_size=8, ema_mode="concat", beta=0.9)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema_h = torch.zeros(4, 8)
    h_new, _ = cell(x, h, ema_h)
    h_new.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_cell_gradient_flows_gate():
    """Gradient should reach gate_alpha (gate mode).

    We need h != ema_h for the gradient to flow through gate_alpha,
    since aug_h = sigmoid(alpha)*h + (1-sigmoid(alpha))*ema_h
    and ∂aug_h/∂alpha = sigmoid*(1-sigmoid)*(h - ema_h).
    """
    cell = EMAHCfCCell(input_size=2, hidden_size=8, ema_mode="gate", beta=0.5)
    x = torch.randn(4, 2)
    h = torch.randn(4, 8)  # non-zero so h - ema_h != 0
    ema_h = torch.zeros(4, 8)
    h_new, _ = cell(x, h, ema_h)
    h_new.sum().backward()
    assert cell.gate_alpha.grad is not None
    assert cell.gate_alpha.grad.abs().sum().item() > 0


def test_cell_no_input_ema():
    """EMA-H-CfC has no input-side EMA — only hidden state EMA."""
    cell = EMAHCfCCell(input_size=2, hidden_size=8, ema_mode="concat", beta=0.9)
    # No input_ema or ema_x (unlike ema_x_cfc)
    # The cell signature is (x, h, ema_h), not (x, h, ema, ...)
    import inspect
    sig = inspect.signature(cell.forward)
    params = list(sig.parameters.keys())
    assert "ema_h" in params
    assert "ema" not in params  # no input-side ema


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network (concat)."""
    net = EMAHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat", beta=0.9,
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8
    assert net.ema_mode == "concat"
    assert net.beta == 0.9


def test_stacked_forward_shape_concat():
    """Forward returns [B, T, output_size] (concat)."""
    net = EMAHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat", beta=0.9, return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_gate():
    """Forward returns [B, T, output_size] (gate)."""
    net = EMAHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="gate", beta=0.9, return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_diff():
    """Forward returns [B, T, output_size] (diff)."""
    net = EMAHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="diff", beta=0.9, return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_ema_only():
    """Forward returns [B, T, output_size] (ema_only)."""
    net = EMAHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="ema_only", beta=0.9, return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = EMAHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat", beta=0.9, return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_stacked_forward_finite_with_nan_concat():
    """Forward handles NaN inputs (concat)."""
    net = EMAHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat", beta=0.9,
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers_concat():
    """Gradient should reach all layers' CfC weights (concat)."""
    net = EMAHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat", beta=0.9,
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


def test_smoke_learns_sin_concat():
    """Smoke: EMA-H-CfC (concat) should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = EMAHCfCStackedNetwork(
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


def test_smoke_learns_sin_diff():
    """Smoke: EMA-H-CfC (diff) should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = EMAHCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2,
        ema_mode="diff", beta=0.9,
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


def test_bench_smoke_ema_h_vs_cfc():
    """Mini-bench: EMA-H-CfC vs CfC baseline on sin task."""
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

    # EMA-H-CfC concat.
    torch.manual_seed(42)
    eh = EMAHCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        ema_mode="concat", beta=0.9, return_sequences=True,
    )
    opt = torch.optim.Adam(eh.parameters(), lr=1e-2)
    eh_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = eh(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        eh_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(eh_loss)


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
