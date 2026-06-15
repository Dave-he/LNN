"""Round 134 — tests for Gated Input Skip CfC cell and stacked network (PRD #10-96)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.gated_input_skip_cfc import (
    GatedInputSkipCfCCell,
    GatedInputSkipCfCStackedNetwork,
)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


def test_init_default():
    """Default init: skip_init_scale=0.1, gate_init_bias=0.0."""
    cell = GatedInputSkipCfCCell(input_size=2, hidden_size=8)
    assert cell.skip_init_scale == 0.1
    assert cell.gate_init_bias == 0.0


def test_init_skip_proj_shape():
    """W_skip should be [hidden_size, input_size]."""
    cell = GatedInputSkipCfCCell(input_size=3, hidden_size=10)
    assert cell.skip_proj.weight.shape == (10, 3)


def test_init_skip_proj_small_norm():
    """W_skip should be initialized with small std (0.1 by default)."""
    cell = GatedInputSkipCfCCell(input_size=2, hidden_size=8)
    norm = cell.skip_proj.weight.norm().item()
    # For [8, 2] tensor with std 0.1, expected norm ≈ sqrt(16 * 0.01) = 0.4.
    assert norm < 1.0, f"W_skip norm too large: {norm}"


def test_init_gate_bias_zero():
    """Default gate_init_bias=0.0 -> gate starts at sigmoid(0) = 0.5."""
    cell = GatedInputSkipCfCCell(input_size=2, hidden_size=8)
    assert abs(cell.gate_proj.bias.mean().item()) < 1e-6


def test_init_gate_bias_positive():
    """Positive gate_init_bias -> gate starts > 0.5."""
    cell = GatedInputSkipCfCCell(input_size=2, hidden_size=8, gate_init_bias=2.0)
    # sigmoid(2.0) ≈ 0.88
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    cell(x, h)
    # The actual gate output depends on x and h, but the bias is +2.0.
    assert cell.gate_proj.bias.mean().item() > 1.5


def test_init_time_scale():
    """Time scale parameter should be initialized to time_scale_init."""
    cell = GatedInputSkipCfCCell(input_size=2, hidden_size=8, time_scale_init=2.5)
    assert torch.allclose(cell.time_scale, torch.full((8,), 2.5))


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------


def test_forward_shape():
    """Forward should return [B, hidden_size]."""
    cell = GatedInputSkipCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert h_new.shape == (4, 8)


def test_forward_skip_zero_initial():
    """When W_skip is zero, gate*skip = 0, so cell behaves like standard CfC."""
    torch.manual_seed(0)
    cell = GatedInputSkipCfCCell(input_size=2, hidden_size=4)
    # Zero out the skip projection.
    with torch.no_grad():
        cell.skip_proj.weight.zero_()
    x = torch.randn(2, 2)
    h = torch.zeros(2, 4)
    h_new = cell(x, h)
    # With skip_proj.weight = 0, the skip term is 0, so h_final = h_new_cfc.
    # Just verify shape and finiteness.
    assert h_new.shape == (2, 4)
    assert torch.isfinite(h_new).all()


def test_forward_skip_only_changes_with_skip_proj():
    """Compare h_new with skip_proj.weight = 0 vs nonzero."""
    torch.manual_seed(0)
    cell_a = GatedInputSkipCfCCell(input_size=2, hidden_size=4)
    torch.manual_seed(0)
    cell_b = GatedInputSkipCfCCell(input_size=2, hidden_size=4)
    # Zero out skip_proj in cell_a.
    with torch.no_grad():
        cell_a.skip_proj.weight.zero_()
        cell_a.skip_proj.weight.requires_grad_(False)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 4)
    h_a = cell_a(x, h)
    h_b = cell_b(x, h)
    # With the same random init for f_gate/g_branch/h_branch, h_a and h_b
    # should differ only due to the skip term.
    assert not torch.allclose(h_a, h_b)


def test_forward_gate_only_changes_with_gate_proj():
    """Compare h_new with gate_proj.bias = -10 vs +10."""
    torch.manual_seed(0)
    cell_a = GatedInputSkipCfCCell(input_size=2, hidden_size=4)
    torch.manual_seed(0)
    cell_b = GatedInputSkipCfCCell(input_size=2, hidden_size=4)
    # Force gate to be 0 in cell_a (sigmoid(-10) ≈ 0).
    with torch.no_grad():
        cell_a.gate_proj.bias.fill_(-10.0)
    # Force gate to be 1 in cell_b (sigmoid(+10) ≈ 1).
    with torch.no_grad():
        cell_b.gate_proj.bias.fill_(10.0)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 4)
    h_a = cell_a(x, h)
    h_b = cell_b(x, h)
    # With the same skip_proj and gate_proj weights, the difference
    # h_b - h_a should be approximately equal to skip (gate=1 - gate=0 = 1).
    assert not torch.allclose(h_a, h_b)


def test_forward_stability_100_steps():
    """No NaN/Inf in 100 sequential forward steps."""
    torch.manual_seed(0)
    cell = GatedInputSkipCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    for _ in range(100):
        h = cell(x, h)
    assert torch.isfinite(h).all()


def test_forward_gate_activation_in_range():
    """The gate output (sigmoid) should be in [0, 1]."""
    cell = GatedInputSkipCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2) * 5.0   # large input
    h = torch.randn(4, 8) * 5.0   # large hidden
    h_new = cell(x, h)
    # Gate must produce values in [0, 1] (sigmoid).
    # We don't have direct access to gate here, but h_new must be finite.
    assert torch.isfinite(h_new).all()
    # The gate diagnostic should be in [0, 1].
    assert 0.0 <= cell._last_gate_mean <= 1.0


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


def test_gradient_to_W_skip():
    """Gradient should reach the W_skip parameters."""
    cell = GatedInputSkipCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.skip_proj.weight.grad is not None
    assert cell.skip_proj.weight.grad.abs().sum().item() > 0


def test_gradient_to_W_gate():
    """Gradient should reach the gate projection weights."""
    cell = GatedInputSkipCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.gate_proj.weight.grad is not None
    assert cell.gate_proj.weight.grad.abs().sum().item() > 0


def test_gradient_to_W_f():
    """Gradient should reach the CfC f_gate weights."""
    cell = GatedInputSkipCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_W_g():
    """Gradient should reach the CfC g_branch weights."""
    cell = GatedInputSkipCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.g_branch[0].weight.grad is not None
    assert cell.g_branch[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_W_h():
    """Gradient should reach the CfC h_branch weights."""
    cell = GatedInputSkipCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.h_branch[0].weight.grad is not None
    assert cell.h_branch[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_time_scale():
    """Gradient should reach the CfC time_scale parameter."""
    cell = GatedInputSkipCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.time_scale.grad is not None
    assert cell.time_scale.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network."""
    net = GatedInputSkipCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8
    assert net.output_size == 1


def test_stacked_forward_shape_sequences():
    """return_sequences=True returns [B, T, output_size]."""
    net = GatedInputSkipCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=True,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 16, 1)


def test_stacked_forward_shape_last_step():
    """return_sequences=False returns [B, output_size]."""
    net = GatedInputSkipCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=False,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 1)


def test_stacked_handles_nan_input():
    """Forward should handle NaN inputs (zero-fill)."""
    net = GatedInputSkipCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach all layers' skip and gate parameters."""
    net = GatedInputSkipCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.skip_proj.weight.grad is not None
        assert cell.skip_proj.weight.grad.abs().sum().item() > 0
        assert cell.gate_proj.weight.grad is not None
        assert cell.gate_proj.weight.grad.abs().sum().item() > 0


def test_stacked_smoke_learns_sin():
    """Smoke: stacked GIS-CfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = GatedInputSkipCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
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
    assert final_loss is not None
    assert math.isfinite(final_loss), f"loss blew up: {final_loss}"
    assert final_loss < 5.0, f"loss too high after 50 steps: {final_loss}"
    # Verify loss decreased.
    assert final_loss < initial_loss, (
        f"loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# Bench-style smoke test
# ---------------------------------------------------------------------------


def test_bench_smoke_gis_vs_cfc():
    """Mini-bench: GIS-CfC vs CfC baseline on sin task."""
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

    # GIS-CfC.
    torch.manual_seed(42)
    gis = GatedInputSkipCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        return_sequences=True,
    )
    opt = torch.optim.Adam(gis.parameters(), lr=1e-2)
    gis_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = gis(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        gis_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(gis_loss)


# ---------------------------------------------------------------------------
# Diagnostic tests
# ---------------------------------------------------------------------------


def test_skip_norm_changes_with_init_scale():
    """W_skip norm should be larger with larger skip_init_scale."""
    cell_small = GatedInputSkipCfCCell(input_size=2, hidden_size=8, skip_init_scale=0.01)
    cell_large = GatedInputSkipCfCCell(input_size=2, hidden_size=8, skip_init_scale=1.0)
    assert cell_large.skip_norm() > cell_small.skip_norm()


def test_gate_responds_to_input():
    """Gate activation should differ for different inputs."""
    cell = GatedInputSkipCfCCell(input_size=2, hidden_size=8)
    h = torch.zeros(2, 8)
    cell(torch.zeros(2, 2), h)
    gate_zero = cell._last_gate_mean
    cell(torch.ones(2, 2) * 5.0, h)
    gate_large = cell._last_gate_mean
    # Different inputs should produce different gates.
    assert gate_zero != gate_large


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
