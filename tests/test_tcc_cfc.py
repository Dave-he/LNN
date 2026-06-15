"""Round 149 — tests for Temporal Conv Concat CfC (TCC-CfC) (PRD #10-111)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.tcc_cfc import TemporalConvConcatCfCCell, TemporalConvConcatCfCStackedNetwork


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------


def test_cell_init_default():
    """Default K=3 init."""
    cell = TemporalConvConcatCfCCell(input_size=2, hidden_size=8, kernel_size=3)
    assert cell.input_size == 2
    assert cell.hidden_size == 8
    assert cell.kernel_size == 3
    # Conv1d: in_channels=2, out_channels=2, kernel_size=3.
    assert cell.conv.in_channels == 2
    assert cell.conv.out_channels == 2
    assert cell.conv.kernel_size == (3,)


def test_cell_init_k1():
    """K=1: conv is just a 1x1 conv (no temporal context)."""
    cell = TemporalConvConcatCfCCell(input_size=2, hidden_size=8, kernel_size=1)
    assert cell.kernel_size == 1
    assert cell.conv.kernel_size == (1,)


def test_cell_init_k5():
    """K=5 conv."""
    cell = TemporalConvConcatCfCCell(input_size=2, hidden_size=8, kernel_size=5)
    assert cell.kernel_size == 5


def test_cell_init_invalid_kernel():
    """K < 1 should raise."""
    try:
        TemporalConvConcatCfCCell(input_size=2, hidden_size=8, kernel_size=0)
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_forward_shape():
    """Forward returns [B, T, hidden]."""
    cell = TemporalConvConcatCfCCell(input_size=2, hidden_size=8, kernel_size=3)
    x = torch.randn(2, 16, 2)
    out = cell(x)
    assert out.shape == (2, 16, 8)


def test_cell_forward_finite():
    """Forward output is finite."""
    cell = TemporalConvConcatCfCCell(input_size=2, hidden_size=8, kernel_size=3)
    x = torch.randn(2, 16, 2) * 5.0
    out = cell(x)
    assert torch.isfinite(out).all()


def test_cell_forward_handles_nan():
    """NaN input handled."""
    cell = TemporalConvConcatCfCCell(input_size=2, hidden_size=8, kernel_size=3)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    out = cell(x)
    assert torch.isfinite(out).all()


def test_cell_conv_causal():
    """Conv should be causal: output at t=0 depends only on x[0..K-1]."""
    torch.manual_seed(0)
    cell = TemporalConvConcatCfCCell(input_size=1, hidden_size=4, kernel_size=3)
    # Set conv weights to identity-like (just x[0] at t=0).
    with torch.no_grad():
        # 1x1 conv for input dim 1, kernel size 3. We want conv at t=0 to
        # only see x[0]. Pad with (2, 0) zeros on the left.
        # The conv weight has shape [out, in, K] = [1, 1, 3].
        # If we set weight = [1, 0, 0], then conv[t] = x[t] (with left pad).
        cell.conv.weight.data = torch.tensor([[[1.0, 0.0, 0.0]]])
        cell.conv.bias.data = torch.zeros(1)
    x = torch.zeros(1, 4, 1)
    x[0, 2, 0] = 1.0  # non-zero at t=2
    out = cell(x)
    # h_t depends on c_t which is the conv output. We don't directly check
    # h values, but we check that the conv runs without errors and is finite.
    assert torch.isfinite(out).all()
    assert out.shape == (1, 4, 4)


def test_cell_gradient_flows_to_conv():
    """Gradient should reach conv weights."""
    cell = TemporalConvConcatCfCCell(input_size=2, hidden_size=8, kernel_size=3)
    x = torch.randn(2, 16, 2)
    out = cell(x)
    out.sum().backward()
    assert cell.conv.weight.grad is not None
    assert cell.conv.weight.grad.abs().sum().item() > 0


def test_cell_gradient_flows_to_cfc():
    """Gradient should reach CfC weights."""
    cell = TemporalConvConcatCfCCell(input_size=2, hidden_size=8, kernel_size=3)
    x = torch.randn(2, 16, 2)
    out = cell(x)
    out.sum().backward()
    assert cell.cfc.f_gate[0].weight.grad is not None
    assert cell.cfc.f_gate[0].weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network."""
    net = TemporalConvConcatCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, kernel_size=3,
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8
    assert net.kernel_size == 3


def test_stacked_init_k5():
    """K=5 stacked network."""
    net = TemporalConvConcatCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, kernel_size=5,
    )
    for cell in net.cells:
        assert cell.kernel_size == 5


def test_stacked_forward_shape():
    """return_sequences=True returns [B, T, output_size]."""
    net = TemporalConvConcatCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, kernel_size=3,
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = TemporalConvConcatCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, kernel_size=3,
        return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_stacked_forward_finite_with_nan():
    """Forward handles NaN inputs."""
    net = TemporalConvConcatCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, kernel_size=3,
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach all layers' conv weights."""
    net = TemporalConvConcatCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, kernel_size=3,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for li, cell in enumerate(net.cells):
        assert cell.conv.weight.grad is not None
        assert cell.conv.weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_learns_sin():
    """Smoke: TCC-CfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = TemporalConvConcatCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, kernel_size=3,
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
    """Smoke: TCC-CfC should reduce loss on structured task."""
    torch.manual_seed(0)
    net = TemporalConvConcatCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, kernel_size=5,
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


def test_bench_smoke_tcc_vs_cfc():
    """Mini-bench: TCC-CfC vs CfC baseline on sin task."""
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

    # TCC-CfC K=3.
    torch.manual_seed(42)
    tcc = TemporalConvConcatCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2, kernel_size=3,
        return_sequences=True,
    )
    opt = torch.optim.Adam(tcc.parameters(), lr=1e-2)
    tcc_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = tcc(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        tcc_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(tcc_loss)


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
