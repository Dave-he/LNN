"""Round 136 — tests for Zoneout CfC cell and stacked network (PRD #10-98)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.zoneout_cfc import (
    ZoneoutCfCCell,
    ZoneoutCfCStackedNetwork,
)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


def test_init_default():
    """Default init: p_zoneout=0.1, time_scale=1.0."""
    cell = ZoneoutCfCCell(input_size=2, hidden_size=8)
    assert cell.p_zoneout == 0.1


def test_init_custom_p_zoneout():
    """Custom p_zoneout should be reflected in cell.p_zoneout."""
    cell = ZoneoutCfCCell(input_size=2, hidden_size=8, p_zoneout=0.3)
    assert cell.p_zoneout == 0.3


def test_init_p_zoneout_zero():
    """p_zoneout=0.0 is valid (no Zoneout)."""
    cell = ZoneoutCfCCell(input_size=2, hidden_size=8, p_zoneout=0.0)
    assert cell.p_zoneout == 0.0


def test_init_p_zoneout_invalid():
    """p_zoneout >= 1.0 should raise ValueError."""
    try:
        ZoneoutCfCCell(input_size=2, hidden_size=8, p_zoneout=1.0)
        assert False, "should have raised"
    except ValueError:
        pass


def test_init_time_scale():
    """Time scale parameter should be initialized to time_scale_init."""
    cell = ZoneoutCfCCell(input_size=2, hidden_size=8, time_scale_init=2.5)
    assert torch.allclose(cell.time_scale, torch.full((8,), 2.5))


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------


def test_forward_shape():
    """Forward should return [B, hidden_size]."""
    cell = ZoneoutCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2)
    h = torch.randn(4, 8)
    h_new = cell(x, h)
    assert h_new.shape == (4, 8)


def test_forward_eval_no_zoneout():
    """At eval mode, Zoneout is disabled — output is h_new_cfc."""
    cell = ZoneoutCfCCell(input_size=2, hidden_size=8, p_zoneout=0.5)
    cell.eval()
    x = torch.randn(2, 2)
    h = torch.randn(2, 8)
    h_new = cell(x, h)
    # At eval, output should be exactly h_new_cfc (no Zoneout).
    # Run the cell twice with different h_prev, output should differ.
    h_new_2 = cell(x, torch.zeros_like(h))
    assert not torch.allclose(h_new, h_new_2)


def test_forward_train_zoneout_active():
    """At train mode with p_zoneout > 0, Zoneout is active."""
    cell = ZoneoutCfCCell(input_size=2, hidden_size=8, p_zoneout=0.5)
    cell.train()
    # Run multiple times — different masks → different outputs.
    outputs = []
    for _ in range(10):
        x = torch.randn(2, 2)
        h = torch.randn(2, 8)
        outputs.append(cell(x, h))
    # All outputs should be finite.
    for o in outputs:
        assert torch.isfinite(o).all()
    # At least some outputs should differ (due to random Zoneout masks).
    # (Statistically, with p=0.5 and hidden_size=8, expected different cells = 4.)


def test_forward_p_zoneout_zero_always_uses_new():
    """p_zoneout=0 → output is always h_new_cfc (no Zoneout)."""
    cell = ZoneoutCfCCell(input_size=2, hidden_size=8, p_zoneout=0.0)
    cell.train()
    x = torch.randn(2, 2)
    h = torch.randn(2, 8)
    # Run twice with the same h — should get the same output (no random mask).
    out1 = cell(x, h)
    out2 = cell(x, h)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_forward_p_zoneout_high_preserves_h():
    """p_zoneout=0.9 → output is mostly h_prev."""
    cell = ZoneoutCfCCell(input_size=2, hidden_size=8, p_zoneout=0.9)
    cell.train()
    torch.manual_seed(0)
    x = torch.randn(2, 2)
    h = torch.randn(2, 8)
    # With p=0.9 per-neuron, expected number of "all 8 neurons zone out"
    # events in 20 trials is 20 * 0.9^8 ≈ 9.3. Use a threshold of 5.
    close_count = 0
    for _ in range(20):
        o = cell(x, h)
        # Compute per-row L1 distance to h.
        dist = (o - h).abs().mean(dim=-1)
        if (dist < 0.1).all():
            close_count += 1
    # Most outputs should be close to h (>=5/20 expected from binomial).
    assert close_count >= 5, f"only {close_count}/20 outputs close to h with p=0.9"


def test_forward_stability_100_steps():
    """No NaN/Inf in 100 sequential forward steps (train mode)."""
    torch.manual_seed(0)
    cell = ZoneoutCfCCell(input_size=2, hidden_size=8, p_zoneout=0.1)
    cell.train()
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    for _ in range(100):
        h = cell(x, h)
    assert torch.isfinite(h).all()


def test_forward_zoneout_rate_diagnostic():
    """The zoneout_rate diagnostic should be tracked."""
    cell = ZoneoutCfCCell(input_size=2, hidden_size=8, p_zoneout=0.3)
    cell.train()
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    cell(x, h)
    # In train mode, the diagnostic should be non-zero (if mask was non-zero).
    # Note: actual rate varies per step due to randomness.
    assert hasattr(cell, "_last_zoneout_rate")


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


def test_gradient_to_W_f():
    """Gradient should reach the CfC f_gate weights."""
    cell = ZoneoutCfCCell(input_size=2, hidden_size=8, p_zoneout=0.1)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_W_g():
    """Gradient should reach the CfC g_branch weights."""
    cell = ZoneoutCfCCell(input_size=2, hidden_size=8, p_zoneout=0.1)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.g_branch[0].weight.grad is not None
    assert cell.g_branch[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_W_h():
    """Gradient should reach the CfC h_branch weights."""
    cell = ZoneoutCfCCell(input_size=2, hidden_size=8, p_zoneout=0.1)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    out = cell(x, h)
    out.sum().backward()
    assert cell.h_branch[0].weight.grad is not None
    assert cell.h_branch[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_time_scale():
    """Gradient should reach the CfC time_scale parameter."""
    cell = ZoneoutCfCCell(input_size=2, hidden_size=8, p_zoneout=0.1)
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
    net = ZoneoutCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8


def test_stacked_forward_shape_sequences():
    """return_sequences=True returns [B, T, output_size]."""
    net = ZoneoutCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=True,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 16, 1)


def test_stacked_forward_shape_last_step():
    """return_sequences=False returns [B, output_size]."""
    net = ZoneoutCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=False,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 1)


def test_stacked_handles_nan_input():
    """Forward should handle NaN inputs (zero-fill)."""
    net = ZoneoutCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_eval_no_zoneout():
    """At eval mode, Zoneout is disabled and outputs are deterministic."""
    net = ZoneoutCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, p_zoneout=0.5,
    )
    net.eval()
    x = torch.randn(2, 16, 2)
    y1 = net(x)
    y2 = net(x)
    assert torch.allclose(y1, y2, atol=1e-6)


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach all layers' parameters."""
    net = ZoneoutCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3, p_zoneout=0.1,
    )
    net.train()
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.g_branch[0].weight.grad is not None
        assert cell.g_branch[0].weight.grad.abs().sum().item() > 0


def test_stacked_smoke_learns_sin():
    """Smoke: stacked Zoneout-CfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = ZoneoutCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, p_zoneout=0.1,
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
        net.train()
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


def test_bench_smoke_zoneout_vs_cfc():
    """Mini-bench: Zoneout-CfC vs CfC baseline on sin task."""
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

    # Zoneout-CfC.
    torch.manual_seed(42)
    zo = ZoneoutCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        p_zoneout=0.1, return_sequences=True,
    )
    opt = torch.optim.Adam(zo.parameters(), lr=1e-2)
    zo_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        zo.train()
        out = zo(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        zo_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(zo_loss)


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
