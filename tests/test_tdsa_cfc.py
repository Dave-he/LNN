"""Round 152 — tests for Time-Domain Self-Attention CfC (TDSA-CfC) (PRD #10-114)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.tdsa_cfc import (
    TimeDomainSelfAttentionCfCCell,
    TimeDomainSelfAttentionCfCStackedNetwork,
)


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------


def test_cell_init_default():
    """Default 1 head, causal mask, attn_dim=input_size."""
    cell = TimeDomainSelfAttentionCfCCell(input_size=4, hidden_size=8)
    assert cell.input_size == 4
    assert cell.hidden_size == 8
    assert cell.num_heads == 1
    assert cell.attn_dim == 4
    assert cell.head_dim == 4
    assert cell.causal is True


def test_cell_init_multi_head():
    """Multi-head init."""
    cell = TimeDomainSelfAttentionCfCCell(input_size=4, hidden_size=8, num_heads=2)
    assert cell.num_heads == 2
    assert cell.head_dim == 2


def test_cell_init_custom_attn_dim():
    """Custom attn_dim."""
    cell = TimeDomainSelfAttentionCfCCell(input_size=4, hidden_size=8, attn_dim=8)
    assert cell.attn_dim == 8
    assert cell.head_dim == 8


def test_cell_init_invalid_attn_dim():
    """Invalid attn_dim should raise."""
    try:
        TimeDomainSelfAttentionCfCCell(input_size=4, hidden_size=8, num_heads=3, attn_dim=4)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_cell_init_noncausal():
    """Non-causal init."""
    cell = TimeDomainSelfAttentionCfCCell(input_size=4, hidden_size=8, causal=False)
    assert cell.causal is False


def test_cell_forward_shape():
    """Forward returns [B, T, hidden_size]."""
    cell = TimeDomainSelfAttentionCfCCell(input_size=4, hidden_size=8)
    x = torch.randn(2, 16, 4)
    out = cell(x)
    assert out.shape == (2, 16, 8)


def test_cell_forward_finite():
    """Forward output is finite."""
    cell = TimeDomainSelfAttentionCfCCell(input_size=4, hidden_size=8)
    x = torch.randn(2, 16, 4) * 5.0
    out = cell(x)
    assert torch.isfinite(out).all()


def test_cell_forward_handles_nan():
    """NaN input handled."""
    cell = TimeDomainSelfAttentionCfCCell(input_size=4, hidden_size=8)
    x = torch.randn(2, 16, 4)
    x[0, 5, 0] = float("nan")
    out = cell(x)
    assert torch.isfinite(out).all()


def test_cell_gradient_flows_to_qkv():
    """Gradient should reach q, k, v, o projections."""
    cell = TimeDomainSelfAttentionCfCCell(input_size=4, hidden_size=8)
    x = torch.randn(2, 16, 4)
    out = cell(x)
    out.sum().backward()
    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        proj = getattr(cell, proj_name)
        assert proj.weight.grad is not None
        assert proj.weight.grad.abs().sum().item() > 0


def test_cell_gradient_flows_to_cfc():
    """Gradient should reach CfC weights."""
    cell = TimeDomainSelfAttentionCfCCell(input_size=4, hidden_size=8)
    x = torch.randn(2, 16, 4)
    out = cell(x)
    out.sum().backward()
    assert cell.cfc.f_gate[0].weight.grad is not None
    assert cell.cfc.f_gate[0].weight.grad.abs().sum().item() > 0


def test_cell_causality():
    """Causal: changing future input should not affect past output."""
    cell = TimeDomainSelfAttentionCfCCell(input_size=4, hidden_size=8, causal=True)
    cell.eval()
    with torch.no_grad():
        x1 = torch.randn(1, 16, 4)
        x2 = x1.clone()
        x2[0, 10:, :] = torch.randn(6, 4)  # change future
        out1 = cell(x1)
        out2 = cell(x2)
        # Output at t < 10 should be identical (causal).
        for t in range(10):
            assert torch.allclose(out1[0, t, :], out2[0, t, :], atol=1e-5), f"Non-causal at t={t}"


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network."""
    net = TimeDomainSelfAttentionCfCStackedNetwork(
        input_size=4, hidden_size=8, output_size=1, num_layers=2,
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8


def test_stacked_init_multi_head():
    """Multi-head stacked network."""
    net = TimeDomainSelfAttentionCfCStackedNetwork(
        input_size=4, hidden_size=8, output_size=1, num_layers=2, num_heads=2,
    )
    assert net.cells[0].num_heads == 2


def test_stacked_forward_shape():
    """Forward returns [B, T, output_size]."""
    net = TimeDomainSelfAttentionCfCStackedNetwork(
        input_size=4, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=True,
    )
    x = torch.randn(2, 16, 4)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = TimeDomainSelfAttentionCfCStackedNetwork(
        input_size=4, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=False,
    )
    x = torch.randn(2, 16, 4)
    y = net(x)
    assert y.shape == (2, 1)


def test_stacked_forward_finite_with_nan():
    """Forward handles NaN inputs."""
    net = TimeDomainSelfAttentionCfCStackedNetwork(
        input_size=4, hidden_size=8, output_size=1, num_layers=2,
    )
    x = torch.randn(2, 16, 4)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach all layers' Q/K/V/O and CfC weights."""
    net = TimeDomainSelfAttentionCfCStackedNetwork(
        input_size=4, hidden_size=8, output_size=1, num_layers=2,
    )
    x = torch.randn(2, 16, 4)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(cell, proj_name)
            assert proj.weight.grad is not None
            assert proj.weight.grad.abs().sum().item() > 0
        assert cell.cfc.f_gate[0].weight.grad is not None


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_learns_sin():
    """Smoke: TDSA-CfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = TimeDomainSelfAttentionCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2,
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


def test_smoke_learns_structured():
    """Smoke: TDSA-CfC should reduce loss on structured task."""
    torch.manual_seed(0)
    net = TimeDomainSelfAttentionCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2,
    )
    t = torch.linspace(0, 4 * math.pi, 16).unsqueeze(0).unsqueeze(-1).expand(2, 16, 1)
    x = torch.zeros(2, 16, 2)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    target = torch.sin(2 * t.squeeze(-1)).unsqueeze(-1)
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


def test_bench_smoke_tdsa_vs_cfc():
    """Mini-bench: TDSA-CfC vs CfC baseline on sin task."""
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

    # TDSA-CfC.
    torch.manual_seed(42)
    tdsa = TimeDomainSelfAttentionCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        return_sequences=True,
    )
    opt = torch.optim.Adam(tdsa.parameters(), lr=1e-2)
    tdsa_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = tdsa(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        tdsa_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(tdsa_loss)


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
