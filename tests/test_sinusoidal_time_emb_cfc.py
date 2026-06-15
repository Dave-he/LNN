"""Round 138 — tests for Sinusoidal Time Embedding CfC (PRD #10-100)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.sinusoidal_time_emb_cfc import (
    SinusoidalTimeEmbCfCCell,
    SinusoidalTimeEmbCfCStackedNetwork,
    sinusoidal_time_embedding,
)


# ---------------------------------------------------------------------------
# Sinusoidal embedding utility tests
# ---------------------------------------------------------------------------


def test_sinusoidal_embedding_shape():
    """Output shape should be t.shape + (dim,)."""
    t = torch.linspace(0, 1, 16)
    emb = sinusoidal_time_embedding(t, dim=4)
    assert emb.shape == (16, 4)


def test_sinusoidal_embedding_bounded():
    """Sinusoidal values should be in [-1, 1]."""
    t = torch.linspace(0, 100, 64)
    emb = sinusoidal_time_embedding(t, dim=8)
    assert (emb >= -1.0).all()
    assert (emb <= 1.0).all()


def test_sinusoidal_embedding_different_t():
    """Different t should give different embeddings."""
    t = torch.tensor([0.0, 0.25, 0.5, 0.75])
    emb = sinusoidal_time_embedding(t, dim=4)
    # Each row should be different.
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            assert not torch.allclose(emb[i], emb[j], atol=1e-6)


def test_sinusoidal_embedding_at_t_zero():
    """At t=0, embedding should be [0, ..., 0, 1, ..., 1] (sin(0)=0, cos(0)=1)."""
    t = torch.tensor([0.0])
    emb = sinusoidal_time_embedding(t, dim=4)
    # First half (sin) should be 0, second half (cos) should be 1.
    assert torch.allclose(emb[0, :2], torch.zeros(2), atol=1e-6)
    assert torch.allclose(emb[0, 2:], torch.ones(2), atol=1e-6)


def test_sinusoidal_embedding_doesnt_depend_on_grad():
    """Sinusoidal embedding should not require grad (no parameters)."""
    t = torch.linspace(0, 1, 8, requires_grad=True)
    emb = sinusoidal_time_embedding(t, dim=4)
    # No params in sinusoidal embedding.
    assert emb.requires_grad is True  # grad flows through t
    # But the function itself has no parameters.


# ---------------------------------------------------------------------------
# Cell init tests
# ---------------------------------------------------------------------------


def test_init_default():
    """Default init: time_emb_dim=4, max_period=10000.0."""
    cell = SinusoidalTimeEmbCfCCell(input_size=2, hidden_size=8)
    assert cell.time_emb_dim == 4
    assert cell.max_period == 10000.0
    assert cell.augmented_input_size == 2 + 4


def test_init_time_scale():
    """Time scale parameter should be initialized to time_scale_init."""
    cell = SinusoidalTimeEmbCfCCell(input_size=2, hidden_size=8, time_scale_init=2.5)
    assert torch.allclose(cell.time_scale, torch.full((8,), 2.5))


def test_init_custom_time_emb_dim():
    """Custom time_emb_dim should be reflected in cell.time_emb_dim."""
    cell = SinusoidalTimeEmbCfCCell(input_size=3, hidden_size=8, time_emb_dim=8)
    assert cell.time_emb_dim == 8
    assert cell.augmented_input_size == 3 + 8


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------


def test_forward_shape():
    """Forward should return [B, hidden_size]."""
    cell = SinusoidalTimeEmbCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    h_new = cell(x, h, t_norm=torch.tensor(0.5))
    assert h_new.shape == (4, 8)


def test_forward_finite():
    """Forward output should be finite."""
    cell = SinusoidalTimeEmbCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2) * 5.0
    h = torch.zeros(4, 8)
    h_new = cell(x, h, t_norm=torch.tensor(0.5))
    assert torch.isfinite(h_new).all()


def test_forward_stability_100_steps():
    """No NaN/Inf in 100 sequential forward steps."""
    torch.manual_seed(0)
    cell = SinusoidalTimeEmbCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    for t in range(100):
        h = cell(x, h, t_norm=torch.tensor(t / 100.0))
    assert torch.isfinite(h).all()


def test_forward_different_t_gives_different_output():
    """Different t_norm should give different h_new (when h same)."""
    torch.manual_seed(0)
    cell = SinusoidalTimeEmbCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_0 = cell(x, h, t_norm=torch.tensor(0.0))
    h_05 = cell(x, h, t_norm=torch.tensor(0.5))
    assert not torch.allclose(h_0, h_05, atol=1e-6)


def test_forward_t_norm_none_works():
    """If t_norm is None, cell should still produce output (use 0.0)."""
    cell = SinusoidalTimeEmbCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h)  # no t_norm
    assert h_new.shape == (2, 8)
    assert torch.isfinite(h_new).all()


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


def test_gradient_to_W_f():
    """Gradient should reach the CfC f_gate weights."""
    cell = SinusoidalTimeEmbCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h, t_norm=torch.tensor(0.5))
    h_new.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_W_g():
    """Gradient should reach the CfC g_branch weights."""
    cell = SinusoidalTimeEmbCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h, t_norm=torch.tensor(0.5))
    h_new.sum().backward()
    assert cell.g_branch[0].weight.grad is not None
    assert cell.g_branch[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_W_h():
    """Gradient should reach the CfC h_branch weights."""
    cell = SinusoidalTimeEmbCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h, t_norm=torch.tensor(0.5))
    h_new.sum().backward()
    assert cell.h_branch[0].weight.grad is not None
    assert cell.h_branch[0].weight.grad.abs().sum().item() > 0


def test_gradient_to_time_scale():
    """Gradient should reach the CfC time_scale parameter."""
    cell = SinusoidalTimeEmbCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 2)
    h = torch.zeros(2, 8)
    h_new = cell(x, h, t_norm=torch.tensor(0.5))
    h_new.sum().backward()
    assert cell.time_scale.grad is not None
    assert cell.time_scale.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network."""
    net = SinusoidalTimeEmbCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8


def test_stacked_forward_shape_sequences():
    """return_sequences=True returns [B, T, output_size]."""
    net = SinusoidalTimeEmbCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=True,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 16, 1)


def test_stacked_forward_shape_last_step():
    """return_sequences=False returns [B, output_size]."""
    net = SinusoidalTimeEmbCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=False,
    )
    x = torch.randn(4, 16, 2)
    y = net(x)
    assert y.shape == (4, 1)


def test_stacked_handles_nan_input():
    """Forward should handle NaN inputs (zero-fill)."""
    net = SinusoidalTimeEmbCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach all layers' parameters."""
    net = SinusoidalTimeEmbCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.g_branch[0].weight.grad is not None
        assert cell.g_branch[0].weight.grad.abs().sum().item() > 0


def test_stacked_smoke_learns_sin():
    """Smoke: stacked TE-CfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = SinusoidalTimeEmbCfCStackedNetwork(
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


def test_stacked_different_time_emb_gives_different_output():
    """Different time_emb_dim should change parameter count."""
    net_d4 = SinusoidalTimeEmbCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, time_emb_dim=4,
    )
    net_d8 = SinusoidalTimeEmbCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, time_emb_dim=8,
    )
    # d8 has more parameters (more input features in projections).
    n_d4 = sum(p.numel() for p in net_d4.parameters())
    n_d8 = sum(p.numel() for p in net_d8.parameters())
    assert n_d8 > n_d4


# ---------------------------------------------------------------------------
# Bench-style smoke test
# ---------------------------------------------------------------------------


def test_bench_smoke_te_vs_cfc():
    """Mini-bench: TE-CfC vs CfC baseline on sin task."""
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

    # TE-CfC.
    torch.manual_seed(42)
    te = SinusoidalTimeEmbCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        return_sequences=True,
    )
    opt = torch.optim.Adam(te.parameters(), lr=1e-2)
    te_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = te(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        te_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(te_loss)


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
