"""Round 135 — tests for Layer Normalization CfC cell and stacked network (PRD #10-97)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.layer_norm_cfc import (
    LayerNormCfCCell,
    LayerNormCfCStackedNetwork,
)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


def test_init_default():
    """Default init: time_scale=1.0, ln_eps=1e-5."""
    cell = LayerNormCfCCell(input_size=2, hidden_size=8)
    assert cell.ln_eps == 1e-5


def test_init_gamma_one_beta_zero():
    """LN gamma should be init to 1.0 and beta to 0.0 (identity at start)."""
    cell = LayerNormCfCCell(input_size=2, hidden_size=8)
    # nn.LayerNorm initializes weight=1, bias=0.
    assert torch.allclose(cell.layer_norm.weight, torch.ones_like(cell.layer_norm.weight))
    assert torch.allclose(cell.layer_norm.bias, torch.zeros_like(cell.layer_norm.bias))


def test_init_time_scale():
    """Time scale parameter should be initialized to time_scale_init."""
    cell = LayerNormCfCCell(input_size=2, hidden_size=8, time_scale_init=2.5)
    assert torch.allclose(cell.time_scale, torch.full((8,), 2.5))


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------


def test_forward_shape():
    """Forward should return [B, hidden_size]."""
    cell = LayerNormCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_new = cell(x, h)
    assert h_new.shape == (4, 8)


def test_forward_finite():
    """Forward output should be finite."""
    cell = LayerNormCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2) * 100.0   # large input
    h = torch.randn(4, 8) * 100.0   # large hidden
    h_new = cell(x, h)
    assert torch.isfinite(h_new).all()


def test_forward_stability_100_steps():
    """No NaN/Inf in 100 sequential forward steps."""
    torch.manual_seed(0)
    cell = LayerNormCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    for _ in range(100):
        h = cell(x, h)
    assert torch.isfinite(h).all()


def test_forward_ln_normalizes_input():
    """Verify LN normalizes the combined input."""
    cell = LayerNormCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2) * 5.0
    h = torch.randn(4, 8) * 5.0
    combined = torch.cat([x, h], dim=-1)
    combined_normed = cell.layer_norm(combined)
    # After LN, each row should have mean ≈ 0, var ≈ 1.
    row_means = combined_normed.mean(dim=-1)
    row_vars = combined_normed.var(dim=-1, unbiased=False)
    assert torch.allclose(row_means, torch.zeros_like(row_means), atol=1e-5)
    assert torch.allclose(row_vars, torch.ones_like(row_vars), atol=1e-3)


def test_forward_ln_identity_at_init():
    """With gamma=1, beta=0, LN(combined) should match identity (modulo normalize)."""
    cell = LayerNormCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.randn(2, 8)
    combined = torch.cat([x, h], dim=-1)
    # At init, gamma=1, beta=0, so LN = (combined - mean) / std.
    expected = (combined - combined.mean(dim=-1, keepdim=True)) / (
        combined.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
    )
    actual = cell.layer_norm(combined)
    assert torch.allclose(actual, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


def test_gradient_to_W_f():
    """Gradient should reach the CfC f_gate weights."""
    cell = LayerNormCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_W_g():
    """Gradient should reach the CfC g_branch weights."""
    cell = LayerNormCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.g_branch[0].weight.grad is not None
    assert cell.g_branch[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_W_h():
    """Gradient should reach the CfC h_branch weights."""
    cell = LayerNormCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.h_branch[0].weight.grad is not None
    assert cell.h_branch[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_ln_gamma():
    """Gradient should reach the LN gamma parameter."""
    cell = LayerNormCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.randn(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.layer_norm.weight.grad is not None
    assert cell.layer_norm.weight.grad.abs().sum().item() > 0


def test_gradient_to_ln_beta():
    """Gradient should reach the LN beta parameter."""
    cell = LayerNormCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.randn(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.layer_norm.bias.grad is not None
    assert cell.layer_norm.bias.grad.abs().sum().item() > 0


def test_gradient_to_time_scale():
    """Gradient should reach the CfC time_scale parameter."""
    cell = LayerNormCfCCell(input_size=2, hidden_size=8)
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
    net = LayerNormCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8
    assert net.output_size == 1


def test_stacked_forward_shape_sequences():
    """return_sequences=True returns [B, T, output_size]."""
    net = LayerNormCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=True,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 16, 1)


def test_stacked_forward_shape_last_step():
    """return_sequences=False returns [B, output_size]."""
    net = LayerNormCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=False,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 1)


def test_stacked_handles_nan_input():
    """Forward should handle NaN inputs (zero-fill)."""
    net = LayerNormCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach all layers' LN parameters."""
    net = LayerNormCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.layer_norm.weight.grad is not None
        assert cell.layer_norm.weight.grad.abs().sum().item() > 0
        assert cell.layer_norm.bias.grad is not None
        assert cell.layer_norm.bias.grad.abs().sum().item() > 0


def test_stacked_smoke_learns_sin():
    """Smoke: stacked LN-CfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = LayerNormCfCStackedNetwork(
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


def test_bench_smoke_ln_vs_cfc():
    """Mini-bench: LN-CfC vs CfC baseline on sin task."""
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

    # LN-CfC.
    torch.manual_seed(42)
    ln_cfc = LayerNormCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        return_sequences=True,
    )
    opt = torch.optim.Adam(ln_cfc.parameters(), lr=1e-2)
    ln_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = ln_cfc(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        ln_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(ln_loss)


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
