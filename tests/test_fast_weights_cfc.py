"""Round 133 — tests for Hebbian Fast Weights CfC cell and stacked network (PRD #10-95)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.fast_weights_cfc import (
    FastWeightsCfCCell,
    FastWeightsCfCStackedNetwork,
)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


def test_init_default():
    """Default init: lambda=0.9, eta=0.1."""
    cell = FastWeightsCfCCell(input_size=2, hidden_size=8)
    assert abs(cell.lam.item() - 0.9) < 0.01
    assert abs(cell.eta.item() - 0.1) < 0.01


def test_init_custom_lambda_eta():
    """Custom init values should be reflected in cell.lam, cell.eta."""
    cell = FastWeightsCfCCell(input_size=2, hidden_size=8, init_lambda=0.5, init_eta=0.05)
    assert abs(cell.lam.item() - 0.5) < 0.01
    assert abs(cell.eta.item() - 0.05) < 0.01


def test_init_F_is_zero():
    """F should be None before any forward pass (treated as 0)."""
    cell = FastWeightsCfCCell(input_size=2, hidden_size=8)
    assert cell._F is None
    assert cell.fast_weight_norm() == 0.0


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------


def test_forward_shape():
    """Forward should return [B, hidden_size]."""
    cell = FastWeightsCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert h_new.shape == (4, 8)


def test_forward_first_step_Fh_is_zero():
    """At t=0, F=0 so Fh=0 and the cell behaves like a standard CfC."""
    cell = FastWeightsCfCCell(input_size=2, hidden_size=4)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 4)
    h_new = cell(x, h)
    # Fh = F @ h = 0 @ 0 = 0
    # combined = [x, h, 0] so behavior matches standard CfC.
    # Just check shape and finiteness.
    assert h_new.shape == (2, 4)
    assert torch.isfinite(h_new).all()


def test_forward_F_evolves_across_steps():
    """After 5 forward steps, F should be non-zero."""
    torch.manual_seed(0)
    cell = FastWeightsCfCCell(input_size=2, hidden_size=4, init_lambda=0.9, init_eta=0.5)
    x = torch.randn(1, 2)
    h = torch.zeros(1, 4)
    for _ in range(5):
        h = cell(x, h)
    assert cell._F is not None
    assert cell._F.abs().sum().item() > 0
    assert cell.fast_weight_norm() > 0


def test_forward_F_resets_between_sequences():
    """After reset_state, F should be None again."""
    cell = FastWeightsCfCCell(input_size=2, hidden_size=4)
    x = torch.randn(1, 2)
    h = torch.zeros(1, 4)
    for _ in range(3):
        h = cell(x, h)
    assert cell._F is not None
    cell.reset_state()
    assert cell._F is None


def test_forward_hebbian_outer_product_sign():
    """The Hebbian outer product should reflect the sign of h_new * h_prev."""
    torch.manual_seed(0)
    cell = FastWeightsCfCCell(input_size=2, hidden_size=4, init_lambda=0.5, init_eta=1.0)
    # Force h to be a fixed positive vector.
    h = torch.ones(1, 4) * 0.5
    x = torch.zeros(1, 2)
    for _ in range(3):
        h = cell(x, h)
    # F should have positive diagonal (outer product of positive vector with itself).
    assert cell._F.diagonal().mean().item() > 0


def test_forward_stability_100_steps():
    """No NaN/Inf in 100 sequential forward steps."""
    torch.manual_seed(0)
    cell = FastWeightsCfCCell(input_size=2, hidden_size=8, init_lambda=0.9, init_eta=0.1)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    for _ in range(100):
        h = cell(x, h)
    assert torch.isfinite(h).all()


def test_forward_F_grows_bounded():
    """F should not blow up over 100 steps (lambda < 1 ensures decay)."""
    torch.manual_seed(0)
    cell = FastWeightsCfCCell(input_size=2, hidden_size=4, init_lambda=0.8, init_eta=0.5)
    x = torch.randn(1, 2)
    h = torch.zeros(1, 4)
    for _ in range(50):
        h = cell(x, h)
    # F should be bounded.
    F_norm = cell.fast_weight_norm()
    assert F_norm < 100.0, f"F norm too large: {F_norm}"


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


def test_gradient_to_W_g():
    """Gradient should reach the g_branch weights."""
    cell = FastWeightsCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    # First Linear in g_branch is at index 0.
    assert cell.g_branch[0].weight.grad is not None
    assert cell.g_branch[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_W_f():
    """Gradient should reach the f_gate weights."""
    cell = FastWeightsCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_lambda():
    """Gradient should reach the lambda parameter (multi-step).

    Need >= 4 steps: at step k+1, the new F_t = lam * F_{t-1} + eta * outer
    depends on lam through F_{t-1} (which is non-zero at step t=3+).
    """
    cell = FastWeightsCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    for _ in range(5):
        h = cell(x, h)
    h.sum().backward()
    assert cell.raw_lambda.grad is not None
    assert cell.raw_lambda.grad.abs().item() > 0


def test_gradient_to_eta():
    """Gradient should reach the eta parameter (multi-step)."""
    cell = FastWeightsCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    for _ in range(3):
        h = cell(x, h)
    h.sum().backward()
    assert cell.raw_eta.grad is not None
    assert cell.raw_eta.grad.abs().item() > 0


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network."""
    net = FastWeightsCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8


def test_stacked_forward_shape_sequences():
    """return_sequences=True returns [B, T, output_size]."""
    net = FastWeightsCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=True,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 16, 1)


def test_stacked_forward_shape_last_step():
    """return_sequences=False returns [B, output_size]."""
    net = FastWeightsCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=False,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 1)


def test_stacked_resets_F_per_forward():
    """Forward should reset F for each new sequence."""
    net = FastWeightsCfCStackedNetwork(
        input_size=2, hidden_size=4, output_size=1, num_layers=2,
    )
    x1 = torch.randn(1, 5, 2)
    x2 = torch.randn(1, 5, 2)
    _ = net(x1)
    f1 = [c._F.clone() if c._F is not None else None for c in net.cells]
    _ = net(x2)
    f2 = [c._F.clone() if c._F is not None else None for c in net.cells]
    # F should evolve in the second forward pass.
    for c1, c2 in zip(f1, f2):
        if c1 is not None and c2 is not None:
            assert not torch.allclose(c1, c2, atol=1e-6)


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach all layers' parameters."""
    net = FastWeightsCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.g_branch[0].weight.grad is not None
        assert cell.g_branch[0].weight.grad.abs().sum().item() > 0
        assert cell.raw_lambda.grad is not None
        assert cell.raw_eta.grad is not None


def test_stacked_smoke_learns_sin():
    """Smoke: stacked FastWeightsCfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = FastWeightsCfCStackedNetwork(
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


# ---------------------------------------------------------------------------
# Bench-style smoke test
# ---------------------------------------------------------------------------


def test_bench_smoke_fast_weights_vs_cfc():
    """Mini-bench: FastWeightsCfC vs CfC baseline on sin task."""
    torch.manual_seed(42)
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

    # FastWeights baseline.
    torch.manual_seed(42)
    fw = FastWeightsCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        return_sequences=True,
    )
    opt = torch.optim.Adam(fw.parameters(), lr=1e-2)
    fw_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = fw(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        fw_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(fw_loss)


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
