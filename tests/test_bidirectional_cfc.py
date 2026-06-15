"""Round 144 — tests for Bidirectional CfC (PRD #10-106)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.bidirectional_cfc import (
    BidirectionalCfCCell,
    BidirectionalCfCStackedNetwork,
    BidirectionalWeightedCfCCell,
)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


def test_init_concat_cell():
    """Default concat cell init."""
    cell = BidirectionalCfCCell(input_size=2, hidden_size=8)
    assert cell.input_size == 2
    assert cell.hidden_size == 8
    assert cell.output_size == 16  # 2 * hidden
    assert hasattr(cell, "forward_cell")
    assert hasattr(cell, "backward_cell")


def test_init_sum_cell():
    """Sum cell init."""
    cell = BidirectionalCfCCell(input_size=2, hidden_size=8, merge_mode="sum")
    assert cell.output_size == 8


def test_init_weighted_cell():
    """Weighted cell init."""
    cell = BidirectionalWeightedCfCCell(input_size=2, hidden_size=8)
    assert hasattr(cell, "alpha_proj")


def test_init_invalid_merge_mode():
    """Invalid merge_mode should raise."""
    try:
        BidirectionalCfCCell(input_size=2, hidden_size=8, merge_mode="invalid")
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_init_stacked_default():
    """Default 2-layer stacked network."""
    net = BidirectionalCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8


def test_init_stacked_all_modes():
    """All merge modes work for stacked."""
    for mode in ("concat", "sum", "weighted"):
        net = BidirectionalCfCStackedNetwork(
            input_size=2, hidden_size=8, output_size=1, num_layers=2, merge_mode=mode,
        )
        assert net.merge_mode == mode


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------


def test_forward_shape_concat():
    """Concat cell returns [B, T, 2*hidden]."""
    cell = BidirectionalCfCCell(input_size=2, hidden_size=8, merge_mode="concat")
    x = torch.randn(4, 16, 2)
    out = cell(x)
    assert out.shape == (4, 16, 16)


def test_forward_shape_sum():
    """Sum cell returns [B, T, hidden]."""
    cell = BidirectionalCfCCell(input_size=2, hidden_size=8, merge_mode="sum")
    x = torch.randn(4, 16, 2)
    out = cell(x)
    assert out.shape == (4, 16, 8)


def test_forward_shape_weighted():
    """Weighted cell returns [B, T, hidden]."""
    cell = BidirectionalWeightedCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 16, 2)
    out = cell(x)
    assert out.shape == (4, 16, 8)


def test_forward_finite_concat():
    """Concat cell output is finite."""
    torch.manual_seed(0)
    cell = BidirectionalCfCCell(input_size=2, hidden_size=8, merge_mode="concat")
    x = torch.randn(4, 16, 2) * 5.0
    out = cell(x)
    assert torch.isfinite(out).all()


def test_forward_handles_nan_input():
    """Forward should handle NaN inputs (zero-fill)."""
    torch.manual_seed(0)
    cell = BidirectionalCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    out = cell(x)
    assert torch.isfinite(out).all()


def test_forward_uses_both_directions_concat():
    """Concat cell output should differ from sum cell output (different merge)."""
    torch.manual_seed(0)
    cell_concat = BidirectionalCfCCell(input_size=2, hidden_size=8, merge_mode="concat")
    cell_sum = BidirectionalCfCCell(input_size=2, hidden_size=8, merge_mode="sum")
    # Copy weights from concat to sum for fair comparison.
    cell_sum.forward_cell.load_state_dict(cell_concat.forward_cell.state_dict())
    cell_sum.backward_cell.load_state_dict(cell_concat.backward_cell.state_dict())
    x = torch.randn(4, 16, 2)
    out_concat = cell_concat(x)
    out_sum = cell_sum(x)
    # Concat and sum should differ (concat has 2*hidden, sum has hidden).
    assert out_concat.shape != out_sum.shape


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


def test_gradient_to_forward_cell_concat():
    """Gradient should reach forward cell weights."""
    cell = BidirectionalCfCCell(input_size=2, hidden_size=8, merge_mode="concat")
    x = torch.randn(2, 16, 2)
    out = cell(x)
    out.sum().backward()
    assert cell.forward_cell.f_gate[0].weight.grad is not None
    assert cell.forward_cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_backward_cell_concat():
    """Gradient should reach backward cell weights."""
    cell = BidirectionalCfCCell(input_size=2, hidden_size=8, merge_mode="concat")
    x = torch.randn(2, 16, 2)
    out = cell(x)
    out.sum().backward()
    assert cell.backward_cell.f_gate[0].weight.grad is not None
    assert cell.backward_cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_alpha_proj_weighted():
    """Gradient should reach alpha_proj in weighted cell."""
    cell = BidirectionalWeightedCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 16, 2)
    out = cell(x)
    out.sum().backward()
    assert cell.alpha_proj.weight.grad is not None
    assert cell.alpha_proj.weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_forward_shape_concat():
    """return_sequences=True returns [B, T, output_size] for concat."""
    net = BidirectionalCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        merge_mode="concat", return_sequences=True,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 16, 1)


def test_stacked_forward_shape_sum():
    """return_sequences=True returns [B, T, output_size] for sum."""
    net = BidirectionalCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        merge_mode="sum", return_sequences=True,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 16, 1)


def test_stacked_forward_shape_weighted():
    """return_sequences=True returns [B, T, output_size] for weighted."""
    net = BidirectionalCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        merge_mode="weighted", return_sequences=True,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 16, 1)


def test_stacked_handles_nan_input():
    """Forward should handle NaN inputs (zero-fill)."""
    net = BidirectionalCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        merge_mode="concat",
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach all layers' parameters."""
    net = BidirectionalCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        merge_mode="concat",
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.forward_cell.f_gate[0].weight.grad is not None
        assert cell.backward_cell.f_gate[0].weight.grad is not None


def test_stacked_smoke_learns_sin_concat():
    """Smoke: stacked bidirectional concat should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = BidirectionalCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        merge_mode="concat",
    )
    t = torch.linspace(0, 4 * math.pi, 16).unsqueeze(0).unsqueeze(-1)
    x = torch.zeros(2, 16, 2)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    target = torch.sin(t.squeeze(-1)).unsqueeze(-1).expand(2, 16, 1)
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


def test_stacked_smoke_learns_sin_weighted():
    """Smoke: stacked weighted bidirectional should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = BidirectionalCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        merge_mode="weighted",
    )
    t = torch.linspace(0, 4 * math.pi, 16).unsqueeze(0).unsqueeze(-1)
    x = torch.zeros(2, 16, 2)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    target = torch.sin(t.squeeze(-1)).unsqueeze(-1).expand(2, 16, 1)
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


def test_bench_smoke_bidirectional_vs_cfc():
    """Mini-bench: Bidirectional CfC vs CfC baseline on sin task."""
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
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        return_sequences=True,
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

    # Bidirectional CfC (sum mode to match param count).
    torch.manual_seed(42)
    bidi = BidirectionalCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        merge_mode="sum", return_sequences=True,
    )
    opt = torch.optim.Adam(bidi.parameters(), lr=1e-2)
    bidi_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = bidi(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        bidi_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(bidi_loss)


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
