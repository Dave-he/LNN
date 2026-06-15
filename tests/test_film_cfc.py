"""Round 153 — tests for FiLM-CfC (Feature-wise Linear Modulation) (PRD #10-115)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.film_cfc import FiLMCfCCell, FiLMCfCStackedNetwork


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------


def test_cell_init_global():
    """Default global ctx_mode init."""
    cell = FiLMCfCCell(input_size=2, hidden_size=8, ctx_mode="global")
    assert cell.input_size == 2
    assert cell.hidden_size == 8
    assert cell.ctx_mode == "global"


def test_cell_init_self():
    """Self ctx_mode init."""
    cell = FiLMCfCCell(input_size=2, hidden_size=8, ctx_mode="self")
    assert cell.ctx_mode == "self"


def test_cell_init_concat():
    """Concat ctx_mode init."""
    cell = FiLMCfCCell(input_size=2, hidden_size=8, ctx_mode="concat")
    assert cell.ctx_mode == "concat"


def test_cell_init_invalid_ctx_mode():
    """Invalid ctx_mode should raise."""
    try:
        FiLMCfCCell(input_size=2, hidden_size=8, ctx_mode="invalid")
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_forward_shape_global():
    """Global ctx_mode returns [B, T, hidden_size]."""
    cell = FiLMCfCCell(input_size=2, hidden_size=8, ctx_mode="global")
    x = torch.randn(2, 16, 2)
    out = cell(x)
    assert out.shape == (2, 16, 8)


def test_cell_forward_shape_self():
    """Self ctx_mode returns [B, T, hidden_size]."""
    cell = FiLMCfCCell(input_size=2, hidden_size=8, ctx_mode="self")
    x = torch.randn(2, 16, 2)
    out = cell(x)
    assert out.shape == (2, 16, 8)


def test_cell_forward_shape_concat():
    """Concat ctx_mode returns [B, T, hidden_size]."""
    cell = FiLMCfCCell(input_size=2, hidden_size=8, ctx_mode="concat")
    x = torch.randn(2, 16, 2)
    out = cell(x)
    assert out.shape == (2, 16, 8)


def test_cell_forward_finite_global():
    """Forward output is finite (global)."""
    cell = FiLMCfCCell(input_size=2, hidden_size=8, ctx_mode="global")
    x = torch.randn(2, 16, 2) * 5.0
    out = cell(x)
    assert torch.isfinite(out).all()


def test_cell_forward_finite_self():
    """Forward output is finite (self)."""
    cell = FiLMCfCCell(input_size=2, hidden_size=8, ctx_mode="self")
    x = torch.randn(2, 16, 2) * 5.0
    out = cell(x)
    assert torch.isfinite(out).all()


def test_cell_forward_handles_nan():
    """NaN input handled (global)."""
    cell = FiLMCfCCell(input_size=2, hidden_size=8, ctx_mode="global")
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    out = cell(x)
    assert torch.isfinite(out).all()


def test_cell_gradient_flows_to_gamma_beta():
    """Gradient should reach gamma_proj and beta_proj (global)."""
    cell = FiLMCfCCell(input_size=2, hidden_size=8, ctx_mode="global")
    x = torch.randn(2, 16, 2)
    out = cell(x)
    out.sum().backward()
    assert cell.gamma_proj.weight.grad is not None
    assert cell.gamma_proj.weight.grad.abs().sum().item() > 0
    assert cell.beta_proj.weight.grad is not None
    assert cell.beta_proj.weight.grad.abs().sum().item() > 0


def test_cell_gradient_flows_to_cfc():
    """Gradient should reach CfC weights."""
    cell = FiLMCfCCell(input_size=2, hidden_size=8, ctx_mode="global")
    x = torch.randn(2, 16, 2)
    out = cell(x)
    out.sum().backward()
    assert cell.cfc.f_gate[0].weight.grad is not None
    assert cell.cfc.f_gate[0].weight.grad.abs().sum().item() > 0


def test_cell_self_modulation_varies_with_x():
    """Self ctx_mode: different x should produce different gamma/beta."""
    cell = FiLMCfCCell(input_size=2, hidden_size=8, ctx_mode="self")
    x1 = torch.zeros(1, 4, 2)
    x2 = torch.ones(1, 4, 2) * 5.0
    out1 = cell(x1)
    out2 = cell(x2)
    # Self modulation: different x → different output.
    assert not torch.allclose(out1, out2)


def test_cell_global_modulation_constant_across_steps():
    """Global ctx_mode: same global ctx → same gamma/beta across timesteps."""
    cell = FiLMCfCCell(input_size=2, hidden_size=8, ctx_mode="global")
    cell.eval()
    with torch.no_grad():
        # x with same global mean but different local structure.
        x1 = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])
        x2 = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
        # Global means: x1 = [0.5, 0.5], x2 = [0.5, 0.5] — same.
        # So gamma and beta should be the same.
        gamma1 = cell.gamma_proj(x1.mean(dim=1, keepdim=True).expand(-1, 4, -1))
        gamma2 = cell.gamma_proj(x2.mean(dim=1, keepdim=True).expand(-1, 4, -1))
        assert torch.allclose(gamma1, gamma2)


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network (global)."""
    net = FiLMCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, ctx_mode="global",
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8
    assert net.ctx_mode == "global"


def test_stacked_forward_shape_global():
    """Forward returns [B, T, output_size] (global)."""
    net = FiLMCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, ctx_mode="global",
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_self():
    """Forward returns [B, T, output_size] (self)."""
    net = FiLMCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, ctx_mode="self",
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = FiLMCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, ctx_mode="global",
        return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_stacked_forward_finite_with_nan():
    """Forward handles NaN inputs."""
    net = FiLMCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, ctx_mode="global",
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach all layers' gamma, beta, and CfC weights."""
    net = FiLMCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, ctx_mode="global",
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.gamma_proj.weight.grad is not None
        assert cell.gamma_proj.weight.grad.abs().sum().item() > 0
        assert cell.beta_proj.weight.grad is not None
        assert cell.beta_proj.weight.grad.abs().sum().item() > 0
        assert cell.cfc.f_gate[0].weight.grad is not None


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_learns_sin():
    """Smoke: FiLM-CfC should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = FiLMCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, ctx_mode="global",
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
    """Smoke: FiLM-CfC should reduce loss on structured task."""
    torch.manual_seed(0)
    net = FiLMCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2, ctx_mode="global",
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


def test_bench_smoke_film_vs_cfc():
    """Mini-bench: FiLM-CfC vs CfC baseline on sin task."""
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

    # FiLM-CfC.
    torch.manual_seed(42)
    film = FiLMCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2, ctx_mode="global",
        return_sequences=True,
    )
    opt = torch.optim.Adam(film.parameters(), lr=1e-2)
    film_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = film(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        film_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(film_loss)


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
