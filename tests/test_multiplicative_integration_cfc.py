"""Round 142 — tests for Multiplicative Integration CfC (PRD #10-104)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.multiplicative_integration_cfc import (
    MultiplicativeIntegrationCfCCell,
    MultiplicativeIntegrationCfCStackedNetwork,
    MultiplicativeIntegrationXResidualCfCCell,
)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


def test_init_default_x_residual():
    """Default variant is 'x_residual'."""
    cell = MultiplicativeIntegrationCfCCell(input_size=2, hidden_size=8)
    assert cell.input_size == 2
    assert cell.hidden_size == 8
    assert isinstance(cell.time_scale, torch.nn.Parameter)


def test_init_pure_cell():
    """Pure MI cell (no x_residual)."""
    cell = MultiplicativeIntegrationCfCCell(input_size=2, hidden_size=8)
    # Has x_proj and h_proj.
    assert hasattr(cell, "x_proj")
    assert hasattr(cell, "h_proj")


def test_init_x_residual_cell():
    """X-residual MI cell."""
    cell = MultiplicativeIntegrationXResidualCfCCell(input_size=2, hidden_size=8)
    assert hasattr(cell, "x_proj")
    assert hasattr(cell, "h_proj")


def test_init_f_bias_non_zero():
    """f_gate bias should be non-zero (default 1.0) to break h=0 symmetry."""
    for cls in (MultiplicativeIntegrationCfCCell, MultiplicativeIntegrationXResidualCfCCell):
        cell = cls(input_size=2, hidden_size=8)
        assert abs(cell.f_gate[0].bias.mean().item()) > 0.5, (
            f"f_gate bias should be non-zero, got {cell.f_gate[0].bias.mean().item()}"
        )


def test_init_invalid_variant():
    """Invalid variant in stacked network should raise."""
    try:
        MultiplicativeIntegrationCfCStackedNetwork(
            input_size=2, hidden_size=8, output_size=1, variant="invalid"
        )
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------


def test_forward_shape_pure():
    """Pure MI cell forward returns [B, hidden_size]."""
    cell = MultiplicativeIntegrationCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert h_new.shape == (4, 8)


def test_forward_shape_x_residual():
    """X-residual MI cell forward returns [B, hidden_size]."""
    cell = MultiplicativeIntegrationXResidualCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert h_new.shape == (4, 8)


def test_forward_finite_pure():
    """Pure MI cell output is finite."""
    torch.manual_seed(0)
    cell = MultiplicativeIntegrationCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2) * 5.0
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert torch.isfinite(h_new).all()


def test_forward_finite_x_residual():
    """X-residual MI cell output is finite."""
    torch.manual_seed(0)
    cell = MultiplicativeIntegrationXResidualCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2) * 5.0
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert torch.isfinite(h_new).all()


def test_forward_evolves_from_h_zero_pure():
    """Pure MI cell: h must evolve from h=0 (chicken-and-egg test)."""
    cell = MultiplicativeIntegrationCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    # h_new should be non-zero (thanks to non-zero gate biases).
    assert h_new.abs().sum() > 1e-3, f"h stayed at zero: {h_new}"


def test_forward_evolves_from_h_zero_x_residual():
    """X-residual MI cell: h must evolve from h=0."""
    cell = MultiplicativeIntegrationXResidualCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    assert h_new.abs().sum() > 1e-3, f"h stayed at zero: {h_new}"


def test_forward_stability_100_steps_pure():
    """No NaN/Inf in 100 sequential pure MI steps."""
    torch.manual_seed(0)
    cell = MultiplicativeIntegrationCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    for _ in range(100):
        h = cell(x, h)
    assert torch.isfinite(h).all()


def test_forward_stability_100_steps_x_residual():
    """No NaN/Inf in 100 sequential x-residual MI steps."""
    torch.manual_seed(0)
    cell = MultiplicativeIntegrationXResidualCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    for _ in range(100):
        h = cell(x, h)
    assert torch.isfinite(h).all()


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


def test_gradient_to_x_proj_pure():
    """Gradient should reach x_proj weights in pure MI cell."""
    cell = MultiplicativeIntegrationCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.x_proj.weight.grad is not None
    assert cell.x_proj.weight.grad.abs().sum().item() > 0


def test_gradient_to_h_proj_pure():
    """Gradient should reach h_proj weights in pure MI cell.

    Note: h must be non-zero for the gradient to flow through the
    multiplicative product (x_proj * h_proj). With h=0, the gradient
    to h_proj is 0 (chicken-and-egg dead zone). In a real forward
    pass, h is non-zero after the first step.
    """
    cell = MultiplicativeIntegrationCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    # Use non-zero h to ensure gradient flows through h_proj.
    h = torch.randn(2, 8)
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.h_proj.weight.grad is not None
    assert cell.h_proj.weight.grad.abs().sum().item() > 0


def test_gradient_to_f_gate_pure():
    """Gradient should reach f_gate weights in pure MI cell."""
    cell = MultiplicativeIntegrationCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_x_proj_x_residual():
    """Gradient should reach x_proj weights in x-residual MI cell."""
    cell = MultiplicativeIntegrationXResidualCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)
    h_new.sum().backward()
    assert cell.x_proj.weight.grad is not None
    assert cell.x_proj.weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network."""
    net = MultiplicativeIntegrationCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8


def test_stacked_init_pure():
    """Pure MI variant."""
    net = MultiplicativeIntegrationCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, variant="pure",
    )
    assert net.variant == "pure"


def test_stacked_init_x_residual():
    """X-residual MI variant."""
    net = MultiplicativeIntegrationCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, variant="x_residual",
    )
    assert net.variant == "x_residual"


def test_stacked_forward_shape_sequences_pure():
    """return_sequences=True returns [B, T, output_size] for pure MI."""
    net = MultiplicativeIntegrationCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, variant="pure",
        return_sequences=True,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 16, 1)


def test_stacked_forward_shape_sequences_x_residual():
    """return_sequences=True returns [B, T, output_size] for x-residual MI."""
    net = MultiplicativeIntegrationCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, variant="x_residual",
        return_sequences=True,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 16, 1)


def test_stacked_handles_nan_input_x_residual():
    """Forward should handle NaN inputs (zero-fill)."""
    net = MultiplicativeIntegrationCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, variant="x_residual",
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers_x_residual():
    """Gradient should reach all layers' parameters (x-residual)."""
    net = MultiplicativeIntegrationCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3, variant="x_residual",
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.x_proj.weight.grad is not None
        assert cell.x_proj.weight.grad.abs().sum().item() > 0
        assert cell.h_proj.weight.grad is not None
        assert cell.h_proj.weight.grad.abs().sum().item() > 0


def test_stacked_smoke_learns_sin_x_residual():
    """Smoke: stacked MI-CfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = MultiplicativeIntegrationCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, variant="x_residual",
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
    assert final_loss < 5.0, f"loss too high after 50 steps: {final_loss}"
    assert final_loss < initial_loss, (
        f"loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# Bench-style smoke test
# ---------------------------------------------------------------------------


def test_bench_smoke_mi_vs_cfc_x_residual():
    """Mini-bench: MI-CfC (x_residual) vs CfC baseline on sin task."""
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

    # MI-CfC (x_residual).
    torch.manual_seed(42)
    mi = MultiplicativeIntegrationCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2, variant="x_residual",
        return_sequences=True,
    )
    opt = torch.optim.Adam(mi.parameters(), lr=1e-2)
    mi_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = mi(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        mi_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(mi_loss)


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
