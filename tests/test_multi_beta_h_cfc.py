"""Round 160 — tests for MultiBeta-H-CfC (Multi-Scale Hidden State EMA) (PRD #10-122)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.multi_beta_h_cfc import MultiBetaHCfCCell, MultiBetaHCfCStackedNetwork


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------


def test_cell_init_diff_2():
    """K=2 diff mode."""
    cell = MultiBetaHCfCCell(
        input_size=2, hidden_size=8, betas=[0.7, 0.95], mode="diff",
    )
    assert cell.K == 2
    assert cell.betas == [0.7, 0.95]
    assert cell.mode == "diff"


def test_cell_init_concat_3():
    """K=3 concat mode."""
    cell = MultiBetaHCfCCell(
        input_size=2, hidden_size=8, betas=[0.5, 0.9, 0.99], mode="concat",
    )
    assert cell.K == 3
    assert cell.betas == [0.5, 0.9, 0.99]
    assert cell.mode == "concat"


def test_cell_init_invalid_mode():
    """Invalid mode should raise."""
    try:
        MultiBetaHCfCCell(
            input_size=2, hidden_size=8, betas=[0.7, 0.95], mode="invalid",
        )
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_init_empty_betas():
    """Empty betas should raise."""
    try:
        MultiBetaHCfCCell(
            input_size=2, hidden_size=8, betas=[], mode="diff",
        )
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_step_shape_diff_2():
    """K=2 diff step shape."""
    cell = MultiBetaHCfCCell(
        input_size=2, hidden_size=8, betas=[0.7, 0.95], mode="diff",
    )
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, emas_h_new = cell(x, h, emas_h)
    assert h_new.shape == (4, 8)
    assert len(emas_h_new) == 2
    for e in emas_h_new:
        assert e.shape == (4, 8)


def test_cell_step_shape_concat_3():
    """K=3 concat step shape."""
    cell = MultiBetaHCfCCell(
        input_size=2, hidden_size=8, betas=[0.5, 0.9, 0.99], mode="concat",
    )
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    emas_h = [torch.zeros(4, 8) for _ in range(3)]
    h_new, emas_h_new = cell(x, h, emas_h)
    assert h_new.shape == (4, 8)
    assert len(emas_h_new) == 3


def test_cell_step_finite_diff_2():
    """Diff K=2 forward is finite."""
    cell = MultiBetaHCfCCell(
        input_size=2, hidden_size=8, betas=[0.7, 0.95], mode="diff",
    )
    x = torch.randn(4, 2) * 5.0
    h = torch.randn(4, 8)
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, emas_h_new = cell(x, h, emas_h)
    assert torch.isfinite(h_new).all()
    for e in emas_h_new:
        assert torch.isfinite(e).all()


def test_cell_step_handles_nan():
    """NaN input handled."""
    cell = MultiBetaHCfCCell(
        input_size=2, hidden_size=8, betas=[0.5, 0.9, 0.99], mode="diff",
    )
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.randn(4, 8)
    h[1, 0] = float("nan")
    emas_h = [torch.zeros(4, 8) for _ in range(3)]
    h_new, emas_h_new = cell(x, h, emas_h)
    assert torch.isfinite(h_new).all()
    for e in emas_h_new:
        assert torch.isfinite(e).all()


def test_cell_ema_h_decay_correct():
    """ema_h_k,t = beta_k * ema_h_k,t-1 + (1-beta_k) * h_t."""
    cell = MultiBetaHCfCCell(
        input_size=2, hidden_size=4, betas=[0.5, 0.8], mode="diff",
    )
    x = torch.randn(2, 2)
    h = torch.ones(2, 4)
    ema_h_0 = torch.zeros(2, 4)
    ema_h_1 = torch.zeros(2, 4)
    _, emas_h_new = cell(x, h, [ema_h_0, ema_h_1])
    # After 1 step with h=1 and ema=0:
    # ema_h_0 = 0.5 * 0 + 0.5 * 1 = 0.5
    # ema_h_1 = 0.8 * 0 + 0.2 * 1 = 0.2
    assert torch.allclose(emas_h_new[0], 0.5 * h, atol=1e-5)
    assert torch.allclose(emas_h_new[1], 0.2 * h, atol=1e-5)


def test_cell_ema_h_recurrent():
    """ema_h after 2 steps with constant h should be 1 - beta^t of h."""
    cell = MultiBetaHCfCCell(
        input_size=2, hidden_size=4, betas=[0.5], mode="diff",
    )
    x = torch.randn(2, 2)
    h = torch.ones(2, 4)
    ema_h_0 = torch.zeros(2, 4)
    _, ema_h_1 = cell(x, h, [ema_h_0])
    _, ema_h_2 = cell(x, h, ema_h_1)
    # After 2 steps: ema_h = (1 - 0.5^2) * h = 0.75 * h
    assert torch.allclose(ema_h_2[0], 0.75 * h, atol=1e-5)


def test_cell_gradient_flows_diff_2():
    """Gradient reaches CfC weights (diff K=2)."""
    cell = MultiBetaHCfCCell(
        input_size=2, hidden_size=8, betas=[0.7, 0.95], mode="diff",
    )
    x = torch.randn(4, 2)
    h = torch.randn(4, 8)
    emas_h = [torch.zeros(4, 8) for _ in range(2)]
    h_new, _ = cell(x, h, emas_h)
    h_new.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_cell_gradient_flows_concat_3():
    """Gradient reaches CfC weights (concat K=3)."""
    cell = MultiBetaHCfCCell(
        input_size=2, hidden_size=8, betas=[0.5, 0.9, 0.99], mode="concat",
    )
    x = torch.randn(4, 2)
    h = torch.randn(4, 8)
    emas_h = [torch.zeros(4, 8) for _ in range(3)]
    h_new, _ = cell(x, h, emas_h)
    h_new.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_cell_no_input_ema():
    """MultiBeta-H-CfC has no input-side EMA — only h-side."""
    cell = MultiBetaHCfCCell(
        input_size=2, hidden_size=8, betas=[0.7, 0.95], mode="diff",
    )
    import inspect
    sig = inspect.signature(cell.forward)
    params = list(sig.parameters.keys())
    assert "emas_h" in params
    assert "emas" not in params  # no input-side emas (multi_beta_cfc has 'emas')


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network (K=2 diff)."""
    net = MultiBetaHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas=[0.7, 0.95], mode="diff",
    )
    assert net.num_layers == 2
    assert net.K == 2
    assert net.mode == "diff"
    assert net.betas == [0.7, 0.95]


def test_stacked_forward_shape_diff_2():
    """Forward returns [B, T, output_size] (diff K=2)."""
    net = MultiBetaHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas=[0.7, 0.95], mode="diff", return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_concat_3():
    """Forward returns [B, T, output_size] (concat K=3)."""
    net = MultiBetaHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas=[0.5, 0.9, 0.99], mode="concat", return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_diff_3():
    """Forward returns [B, T, output_size] (diff K=3)."""
    net = MultiBetaHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas=[0.5, 0.9, 0.99], mode="diff", return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_concat_2():
    """Forward returns [B, T, output_size] (concat K=2)."""
    net = MultiBetaHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas=[0.7, 0.95], mode="concat", return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = MultiBetaHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas=[0.7, 0.95], mode="diff", return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_stacked_forward_finite_with_nan():
    """Forward handles NaN inputs."""
    net = MultiBetaHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas=[0.7, 0.95], mode="diff",
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers():
    """Gradient reaches all layers' CfC weights."""
    net = MultiBetaHCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas=[0.7, 0.95], mode="diff",
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.f_gate[0].weight.grad is not None
        assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_learns_sin_diff_2():
    """Smoke: MultiBeta-H-CfC (diff K=2) should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = MultiBetaHCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2,
        betas=[0.7, 0.95], mode="diff",
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


def test_smoke_learns_sin_concat_3():
    """Smoke: MultiBeta-H-CfC (concat K=3) should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = MultiBetaHCfCStackedNetwork(
        input_size=2, hidden_size=12, output_size=1, num_layers=2,
        betas=[0.5, 0.9, 0.99], mode="concat",
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


# ---------------------------------------------------------------------------
# Bench-style smoke test
# ---------------------------------------------------------------------------


def test_bench_smoke_mbh_vs_cfc():
    """Mini-bench: MultiBeta-H-CfC vs CfC baseline on sin task."""
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

    # MultiBeta-H-CfC diff K=2.
    torch.manual_seed(42)
    mbh = MultiBetaHCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        betas=[0.7, 0.95], mode="diff", return_sequences=True,
    )
    opt = torch.optim.Adam(mbh.parameters(), lr=1e-2)
    mbh_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = mbh(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        mbh_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(mbh_loss)


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
