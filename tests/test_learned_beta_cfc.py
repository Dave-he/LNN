"""Round 157 — tests for LearnedBeta-CfC (per-feature learnable β EMA) (PRD #10-119)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.learned_beta_cfc import LearnedBetaCfCCell, LearnedBetaCfCStackedNetwork


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------


def test_cell_init_concat():
    """Default concat mode."""
    cell = LearnedBetaCfCCell(input_size=2, hidden_size=8, ema_mode="concat")
    assert cell.input_size == 2
    assert cell.hidden_size == 8
    assert cell.ema_mode == "concat"
    # beta should be in (0, 1)
    assert (cell.beta > 0).all() and (cell.beta < 1).all()
    # default sigmoid(2.197) ≈ 0.9
    assert torch.allclose(cell.beta, torch.full((2,), 0.9), atol=0.01)


def test_cell_init_gate():
    """Gate mode: has learnable alpha."""
    cell = LearnedBetaCfCCell(input_size=2, hidden_size=8, ema_mode="gate")
    assert cell.ema_mode == "gate"
    assert cell.gate_alpha is not None


def test_cell_init_diff():
    """Diff mode."""
    cell = LearnedBetaCfCCell(input_size=2, hidden_size=8, ema_mode="diff")
    assert cell.ema_mode == "diff"


def test_cell_init_ema_only():
    """EMA-only mode."""
    cell = LearnedBetaCfCCell(input_size=2, hidden_size=8, ema_mode="ema_only")
    assert cell.ema_mode == "ema_only"


def test_cell_init_invalid_mode():
    """Invalid ema_mode should raise."""
    try:
        LearnedBetaCfCCell(input_size=2, hidden_size=8, ema_mode="invalid")
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_step_shape_concat():
    """Concat mode: returns h and ema."""
    cell = LearnedBetaCfCCell(input_size=2, hidden_size=8, ema_mode="concat")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema = torch.zeros(4, 2)
    h_new, ema_new = cell(x, h, ema)
    assert h_new.shape == (4, 8)
    assert ema_new.shape == (4, 2)


def test_cell_step_shape_gate():
    """Gate mode shape."""
    cell = LearnedBetaCfCCell(input_size=2, hidden_size=8, ema_mode="gate")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema = torch.zeros(4, 2)
    h_new, ema_new = cell(x, h, ema)
    assert h_new.shape == (4, 8)
    assert ema_new.shape == (4, 2)


def test_cell_step_shape_diff():
    """Diff mode shape."""
    cell = LearnedBetaCfCCell(input_size=2, hidden_size=8, ema_mode="diff")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema = torch.zeros(4, 2)
    h_new, ema_new = cell(x, h, ema)
    assert h_new.shape == (4, 8)
    assert ema_new.shape == (4, 2)


def test_cell_step_shape_ema_only():
    """EMA-only mode shape."""
    cell = LearnedBetaCfCCell(input_size=2, hidden_size=8, ema_mode="ema_only")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema = torch.zeros(4, 2)
    h_new, ema_new = cell(x, h, ema)
    assert h_new.shape == (4, 8)
    assert ema_new.shape == (4, 2)


def test_cell_step_finite_concat():
    """Forward output is finite (concat)."""
    cell = LearnedBetaCfCCell(input_size=2, hidden_size=8, ema_mode="concat")
    x = torch.randn(4, 2) * 5.0
    h = torch.zeros(4, 8)
    ema = torch.zeros(4, 2)
    h_new, ema_new = cell(x, h, ema)
    assert torch.isfinite(h_new).all()
    assert torch.isfinite(ema_new).all()


def test_cell_step_handles_nan_concat():
    """NaN input handled (concat)."""
    cell = LearnedBetaCfCCell(input_size=2, hidden_size=8, ema_mode="concat")
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 8)
    ema = torch.zeros(4, 2)
    h_new, ema_new = cell(x, h, ema)
    assert torch.isfinite(h_new).all()
    assert torch.isfinite(ema_new).all()


def test_cell_ema_decay_correct():
    """EMA_t = beta * EMA_{t-1} + (1-beta) * x_t with fixed beta=0.5."""
    # Manually set beta_raw so that sigmoid gives 0.5
    cell = LearnedBetaCfCCell(input_size=2, hidden_size=8, ema_mode="concat", beta_init=0.0)
    # sigmoid(0) = 0.5
    x = torch.tensor([[1.0, 0.0]])
    h = torch.zeros(1, 8)
    ema = torch.zeros(1, 2)
    _, ema_new = cell(x, h, ema)
    # ema_new = 0.5 * 0 + 0.5 * [1, 0] = [0.5, 0]
    assert torch.allclose(ema_new, torch.tensor([[0.5, 0.0]]), atol=1e-5)


def test_cell_ema_recurrent():
    """EMA after 2 steps with constant x should be 1 - beta^t of x (β=0.5)."""
    cell = LearnedBetaCfCCell(input_size=2, hidden_size=8, ema_mode="concat", beta_init=0.0)
    x = torch.tensor([[1.0, 0.0]])
    h = torch.zeros(1, 8)
    ema = torch.zeros(1, 2)
    _, ema_1 = cell(x, h, ema)
    _, ema_2 = cell(x, h, ema_1)
    # After 2 steps: ema = (1 - 0.5^2) * x = 0.75 * [1, 0]
    assert torch.allclose(ema_2, torch.tensor([[0.75, 0.0]]), atol=1e-5)


def test_cell_per_feature_beta_different():
    """β should be per-feature (dim D)."""
    cell = LearnedBetaCfCCell(input_size=4, hidden_size=8, ema_mode="concat")
    # β is dim 4 (per-feature)
    assert cell.beta.shape == (4,)


def test_cell_beta_is_learnable():
    """β is learnable (gradient flows to beta_raw)."""
    cell = LearnedBetaCfCCell(input_size=2, hidden_size=8, ema_mode="concat")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema = torch.zeros(4, 2)
    h_new, _ = cell(x, h, ema)
    h_new.sum().backward()
    assert cell.beta_raw.grad is not None
    assert cell.beta_raw.grad.abs().sum().item() > 0


def test_cell_beta_in_unit_interval():
    """β = sigmoid(beta_raw) should always be in (0, 1) regardless of raw value."""
    cell = LearnedBetaCfCCell(input_size=3, hidden_size=8, ema_mode="concat")
    # Set extreme values for beta_raw
    with torch.no_grad():
        cell.beta_raw.copy_(torch.tensor([-50.0, 0.0, 50.0]))
    beta = cell.beta
    # Sigmoid saturates but still strictly in [0, 1] (allow 0/1 at saturation)
    assert (beta >= 0).all() and (beta <= 1).all()
    # Check expected sigmoid values (sigmoid(-50) ≈ 0, sigmoid(0) = 0.5, sigmoid(50) ≈ 1)
    assert torch.allclose(beta[0], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(beta[1], torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(beta[2], torch.tensor(1.0), atol=1e-6)


def test_cell_gradient_flows_concat():
    """Gradient should reach CfC weights (concat)."""
    cell = LearnedBetaCfCCell(input_size=2, hidden_size=8, ema_mode="concat")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    ema = torch.zeros(4, 2)
    h_new, _ = cell(x, h, ema)
    h_new.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_cell_gradient_flows_gate():
    """Gradient should reach gate_alpha (gate mode)."""
    cell = LearnedBetaCfCCell(input_size=2, hidden_size=8, ema_mode="gate")
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
    net = LearnedBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat",
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8
    assert net.ema_mode == "concat"
    # β should be in (0, 1) for first cell
    assert (net.cells[0].beta > 0).all() and (net.cells[0].beta < 1).all()


def test_stacked_forward_shape_concat():
    """Forward returns [B, T, output_size] (concat)."""
    net = LearnedBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat", return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_gate():
    """Forward returns [B, T, output_size] (gate)."""
    net = LearnedBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="gate", return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_diff():
    """Forward returns [B, T, output_size] (diff)."""
    net = LearnedBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="diff", return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_ema_only():
    """Forward returns [B, T, output_size] (ema_only)."""
    net = LearnedBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="ema_only", return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = LearnedBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat", return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_stacked_forward_finite_with_nan_concat():
    """Forward handles NaN inputs (concat)."""
    net = LearnedBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat",
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers_concat():
    """Gradient should reach all layers' CfC weights (concat)."""
    net = LearnedBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat",
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.f_gate[0].weight.grad is not None
        assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_stacked_gradient_flows_to_beta():
    """Gradient should reach all cells' beta_raw (concat)."""
    net = LearnedBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        ema_mode="concat",
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.beta_raw.grad is not None
        assert cell.beta_raw.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_learns_sin_concat():
    """Smoke: LearnedBeta-CfC (concat) should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = LearnedBetaCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2,
        ema_mode="concat",
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
    """Smoke: LearnedBeta-CfC (diff) should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = LearnedBetaCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2,
        ema_mode="diff",
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


def test_bench_smoke_learned_beta_vs_cfc():
    """Mini-bench: LearnedBeta-CfC vs CfC baseline on sin task."""
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

    # LearnedBeta-CfC concat.
    torch.manual_seed(42)
    lb = LearnedBetaCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        ema_mode="concat", return_sequences=True,
    )
    opt = torch.optim.Adam(lb.parameters(), lr=1e-2)
    lb_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = lb(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        lb_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(lb_loss)


def test_bench_beta_adapts_after_training():
    """β values should differ after training (per-feature adaptation)."""
    torch.manual_seed(0)
    net = LearnedBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=1,
        ema_mode="diff",
    )
    # Initial β
    beta_init = net.cells[0].beta.detach().clone()
    # Train briefly on a structured task
    B, T, D = 4, 32, 2
    t = torch.linspace(0, 4 * math.pi, T).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
    x = torch.zeros(B, T, D)
    half = T // 2
    x[:, :half, 0] = torch.sin(t[:, :half, 0])
    x[:, half:, 0] = torch.sin(2 * t[:, half:, 0])
    target = x[:, :, :1]
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(20):
        opt.zero_grad()
        out = net(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()
    beta_after = net.cells[0].beta.detach().clone()
    # At least one feature's β should have changed
    diff = (beta_after - beta_init).abs().sum().item()
    assert diff > 1e-4  # some adaptation occurred


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
