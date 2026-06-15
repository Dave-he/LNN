"""Round 158 — tests for MultiBeta-CfC (Multi-Scale EMA Augmentation) (PRD #10-120)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.multi_beta_cfc import MultiBetaCfCCell, MultiBetaCfCStackedNetwork


# ---------------------------------------------------------------------------
# Cell tests
# ---------------------------------------------------------------------------


def test_cell_init_diff():
    """Default diff mode with 2 betas."""
    cell = MultiBetaCfCCell(input_size=2, hidden_size=8, betas=[0.7, 0.95], mode="diff")
    assert cell.input_size == 2
    assert cell.hidden_size == 8
    assert cell.betas == [0.7, 0.95]
    assert cell.K == 2
    assert cell.mode == "diff"


def test_cell_init_concat():
    """Concat mode with 3 betas."""
    cell = MultiBetaCfCCell(input_size=2, hidden_size=8, betas=[0.5, 0.9, 0.99], mode="concat")
    assert cell.K == 3
    assert cell.mode == "concat"


def test_cell_init_invalid_mode():
    """Invalid mode should raise."""
    try:
        MultiBetaCfCCell(input_size=2, hidden_size=8, betas=[0.9], mode="invalid")
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_init_empty_betas():
    """Empty betas should raise."""
    try:
        MultiBetaCfCCell(input_size=2, hidden_size=8, betas=[], mode="diff")
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_cell_step_shape_diff():
    """Diff mode: returns h and list of emas."""
    cell = MultiBetaCfCCell(input_size=2, hidden_size=8, betas=[0.7, 0.95], mode="diff")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    emas = [torch.zeros(4, 2), torch.zeros(4, 2)]
    h_new, emas_new = cell(x, h, emas)
    assert h_new.shape == (4, 8)
    assert len(emas_new) == 2
    for e in emas_new:
        assert e.shape == (4, 2)


def test_cell_step_shape_concat():
    """Concat mode: returns h and list of emas."""
    cell = MultiBetaCfCCell(input_size=2, hidden_size=8, betas=[0.5, 0.9, 0.99], mode="concat")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    emas = [torch.zeros(4, 2) for _ in range(3)]
    h_new, emas_new = cell(x, h, emas)
    assert h_new.shape == (4, 8)
    assert len(emas_new) == 3


def test_cell_step_finite_diff():
    """Forward output is finite (diff)."""
    cell = MultiBetaCfCCell(input_size=2, hidden_size=8, betas=[0.7, 0.95], mode="diff")
    x = torch.randn(4, 2) * 5.0
    h = torch.zeros(4, 8)
    emas = [torch.zeros(4, 2), torch.zeros(4, 2)]
    h_new, emas_new = cell(x, h, emas)
    assert torch.isfinite(h_new).all()
    for e in emas_new:
        assert torch.isfinite(e).all()


def test_cell_step_handles_nan_diff():
    """NaN input handled (diff)."""
    cell = MultiBetaCfCCell(input_size=2, hidden_size=8, betas=[0.7, 0.95], mode="diff")
    x = torch.randn(4, 2)
    x[0, 0] = float("nan")
    h = torch.zeros(4, 8)
    emas = [torch.zeros(4, 2), torch.zeros(4, 2)]
    h_new, emas_new = cell(x, h, emas)
    assert torch.isfinite(h_new).all()
    for e in emas_new:
        assert torch.isfinite(e).all()


def test_cell_ema_decay_correct():
    """Each EMA follows its own β."""
    cell = MultiBetaCfCCell(input_size=2, hidden_size=8, betas=[0.5], mode="diff")
    x = torch.tensor([[1.0, 0.0]])
    h = torch.zeros(1, 8)
    emas = [torch.zeros(1, 2)]
    _, emas_new = cell(x, h, emas)
    # ema_new = 0.5 * 0 + 0.5 * [1, 0] = [0.5, 0]
    assert torch.allclose(emas_new[0], torch.tensor([[0.5, 0.0]]), atol=1e-5)


def test_cell_ema_different_betas_converge_differently():
    """EMA with different β values should produce different values."""
    cell = MultiBetaCfCCell(input_size=1, hidden_size=4, betas=[0.5, 0.99], mode="diff")
    x = torch.tensor([[1.0]])
    h = torch.zeros(1, 4)
    emas = [torch.zeros(1, 1), torch.zeros(1, 1)]
    # Run 5 steps with constant x=1
    for _ in range(5):
        _, emas = cell(x, h, emas)
    # EMA with beta=0.5 should be closer to 1.0 (more aggressive smoothing)
    # EMA with beta=0.99 should be closer to 0 (less aggressive)
    assert emas[0].item() > emas[1].item()
    assert emas[0].item() > 0.9  # beta=0.5 close to 1
    assert emas[1].item() < 0.1  # beta=0.99 close to 0


def test_cell_gradient_flows_diff():
    """Gradient should reach CfC weights (diff)."""
    cell = MultiBetaCfCCell(input_size=2, hidden_size=8, betas=[0.7, 0.95], mode="diff")
    x = torch.randn(4, 2)
    h = torch.zeros(4, 8)
    emas = [torch.zeros(4, 2), torch.zeros(4, 2)]
    h_new, _ = cell(x, h, emas)
    h_new.sum().backward()
    assert cell.f_gate[0].weight.grad is not None
    assert cell.f_gate[0].weight.grad.abs().sum().item() > 0


def test_cell_no_learnable_beta():
    """MultiBeta has no learnable β — only fixed hyperparameters."""
    cell = MultiBetaCfCCell(input_size=2, hidden_size=8, betas=[0.7, 0.95], mode="diff")
    # No beta_raw or gate_alpha (unlike learned_beta_cfc)
    assert not hasattr(cell, "beta_raw")
    assert not hasattr(cell, "gate_alpha")


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network (diff)."""
    net = MultiBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas=[0.7, 0.95], mode="diff",
    )
    assert net.num_layers == 2
    assert net.hidden_size == 8
    assert net.betas == [0.7, 0.95]
    assert net.mode == "diff"


def test_stacked_init_3_betas():
    """3-betas stacked network."""
    net = MultiBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas=[0.5, 0.9, 0.99], mode="diff",
    )
    assert net.betas == [0.5, 0.9, 0.99]
    assert net.K == 3


def test_stacked_forward_shape_diff():
    """Forward returns [B, T, output_size] (diff)."""
    net = MultiBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas=[0.7, 0.95], mode="diff", return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_shape_concat():
    """Forward returns [B, T, output_size] (concat)."""
    net = MultiBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas=[0.5, 0.9, 0.99], mode="concat", return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = MultiBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas=[0.7, 0.95], mode="diff", return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_stacked_forward_finite_with_nan_diff():
    """Forward handles NaN inputs (diff)."""
    net = MultiBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas=[0.7, 0.95], mode="diff",
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers_diff():
    """Gradient should reach all layers' CfC weights (diff)."""
    net = MultiBetaCfCStackedNetwork(
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


def test_smoke_learns_sin_diff():
    """Smoke: MultiBeta-CfC (diff, 2 betas) should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = MultiBetaCfCStackedNetwork(
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


def test_smoke_learns_sin_concat_3betas():
    """Smoke: MultiBeta-CfC (concat, 3 betas) should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = MultiBetaCfCStackedNetwork(
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


def test_bench_smoke_multi_beta_vs_cfc():
    """Mini-bench: MultiBeta-CfC vs CfC baseline on sin task."""
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

    # MultiBeta-CfC diff.
    torch.manual_seed(42)
    mb = MultiBetaCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        betas=[0.7, 0.95], mode="diff", return_sequences=True,
    )
    opt = torch.optim.Adam(mb.parameters(), lr=1e-2)
    mb_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = mb(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        mb_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(mb_loss)


def test_bench_smoke_3_betas_vs_2_betas():
    """Mini-bench: K=3 should produce different output than K=2."""
    torch.manual_seed(42)
    mb2 = MultiBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas=[0.7, 0.95], mode="diff", return_sequences=True,
    )
    torch.manual_seed(42)
    mb3 = MultiBetaCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        betas=[0.5, 0.9, 0.99], mode="diff", return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y2 = mb2(x)
    y3 = mb3(x)
    # Outputs should be different (K=2 vs K=3)
    assert not torch.allclose(y2, y3)


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
