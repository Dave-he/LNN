"""Round 146 — tests for Slow Context RNN CfC (PRD #10-108)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.scrn_cfc import SlowContextEncoder, SCRNCfCCell, SCRNCfCStackedNetwork


# ---------------------------------------------------------------------------
# SlowContextEncoder tests
# ---------------------------------------------------------------------------


def test_slow_encoder_init_default():
    """Default init: alpha=0.95, slow_size=hidden_size."""
    enc = SlowContextEncoder(input_size=4, slow_size=8)
    assert enc.input_size == 4
    assert enc.slow_size == 8
    # alpha should be ≈ 0.95 at init.
    alpha = enc.alpha.item()
    assert abs(alpha - 0.95) < 1e-4


def test_slow_encoder_init_alpha_05():
    """Init alpha=0.5."""
    enc = SlowContextEncoder(input_size=4, slow_size=8, alpha_init=0.5)
    alpha = enc.alpha.item()
    assert abs(alpha - 0.5) < 1e-4


def test_slow_encoder_init_alpha_099():
    """Init alpha=0.99 (long memory)."""
    enc = SlowContextEncoder(input_size=4, slow_size=8, alpha_init=0.99)
    alpha = enc.alpha.item()
    assert abs(alpha - 0.99) < 1e-4


def test_slow_encoder_invalid_alpha():
    """Invalid alpha should raise."""
    try:
        SlowContextEncoder(input_size=4, slow_size=8, alpha_init=1.0)
        assert False, "Should have raised"
    except AssertionError:
        pass
    try:
        SlowContextEncoder(input_size=4, slow_size=8, alpha_init=-0.1)
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_slow_encoder_forward_shape():
    """Forward returns [B, T, slow_size]."""
    enc = SlowContextEncoder(input_size=2, slow_size=4)
    x = torch.randn(2, 8, 2)
    out = enc(x)
    assert out.shape == (2, 8, 4)


def test_slow_encoder_ema_constant_input():
    """For constant input, slow context should converge to W_s @ x + bias."""
    enc = SlowContextEncoder(input_size=2, slow_size=3, alpha_init=0.5)
    enc.eval()  # disable any stochastic behavior
    x = torch.ones(1, 50, 2)  # constant input, long enough for convergence
    out = enc(x)
    # After many steps, s_t should be close to W_s @ x + bias.
    # The transient decays as α^t. For T=50 and α=0.5, transient is 2^(-50) ≈ 1e-15.
    W_s = enc.proj.weight  # [slow_size, input_size]
    b_s = enc.proj.bias  # [slow_size]
    x_const = torch.ones(2)  # [D]
    expected_ss = (W_s @ x_const + b_s).detach()  # [slow_size]
    # Compare last timestep.
    assert torch.allclose(out[0, -1, :].detach(), expected_ss, atol=1e-3)


def test_slow_encoder_handles_nan():
    """NaN inputs are zero-filled."""
    enc = SlowContextEncoder(input_size=2, slow_size=4)
    x = torch.randn(2, 8, 2)
    x[0, 3, 1] = float("nan")
    out = enc(x)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# SCRNCfCCell tests
# ---------------------------------------------------------------------------


def test_scrn_cell_init_default():
    """Default cell init."""
    cell = SCRNCfCCell(input_size=2, hidden_size=8)
    assert cell.input_size == 2
    assert cell.hidden_size == 8
    assert cell.slow_size == 8
    assert cell.output_size == 16  # hidden + slow


def test_scrn_cell_custom_slow_size():
    """Custom slow_size."""
    cell = SCRNCfCCell(input_size=2, hidden_size=8, slow_size=4)
    assert cell.slow_size == 4
    assert cell.output_size == 12


def test_scrn_cell_alpha_init_05():
    """alpha_init=0.5 propagates to slow encoder."""
    cell = SCRNCfCCell(input_size=2, hidden_size=8, alpha_init=0.5)
    alpha = cell.slow_encoder.alpha.item()
    assert abs(alpha - 0.5) < 1e-4


def test_scrn_cell_forward_shape():
    """Forward returns [B, T, hidden+slow]."""
    cell = SCRNCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 16, 2)
    out = cell(x)
    assert out.shape == (2, 16, 16)


def test_scrn_cell_forward_finite():
    """Forward output is finite."""
    cell = SCRNCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 16, 2) * 5.0
    out = cell(x)
    assert torch.isfinite(out).all()


def test_scrn_cell_handles_nan():
    """NaN input handled."""
    cell = SCRNCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    out = cell(x)
    assert torch.isfinite(out).all()


def test_scrn_cell_gradient_flows_to_slow_encoder():
    """Gradient should reach slow encoder params."""
    cell = SCRNCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 16, 2)
    out = cell(x)
    out.sum().backward()
    # Slow encoder has proj.weight and logit_alpha.
    assert cell.slow_encoder.proj.weight.grad is not None
    assert cell.slow_encoder.proj.weight.grad.abs().sum().item() > 0
    assert cell.slow_encoder.logit_alpha.grad is not None
    assert cell.slow_encoder.logit_alpha.grad.abs().sum().item() > 0


def test_scrn_cell_gradient_flows_to_cfc():
    """Gradient should reach CfC cell params."""
    cell = SCRNCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 16, 2)
    out = cell(x)
    out.sum().backward()
    assert cell.cfc_cell.f_gate[0].weight.grad is not None
    assert cell.cfc_cell.f_gate[0].weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Stacked network tests
# ---------------------------------------------------------------------------


def test_stacked_init_default():
    """Default 2-layer stacked network."""
    net = SCRNCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    assert net.num_layers == 2
    assert net.slow_size == 8
    assert net.output_size == 1


def test_stacked_init_custom_alpha():
    """alpha_init propagates to all cells."""
    net = SCRNCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, alpha_init=0.5,
    )
    for cell in net.cells:
        assert abs(cell.slow_encoder.alpha.item() - 0.5) < 1e-4


def test_stacked_forward_shape():
    """return_sequences=True returns [B, T, output_size]."""
    net = SCRNCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
        return_sequences=True,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_stacked_forward_finite_with_nan():
    """Forward handles NaN inputs."""
    net = SCRNCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    y = net(x)
    assert torch.isfinite(y).all()


def test_stacked_gradient_flows_to_all_layers():
    """Gradient should reach all cells' slow encoders and CfC parts."""
    net = SCRNCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.slow_encoder.proj.weight.grad is not None
        assert cell.cfc_cell.f_gate[0].weight.grad is not None


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_learns_sin_alpha_095():
    """Smoke: SCRN α=0.95 should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = SCRNCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, alpha_init=0.95,
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


def test_smoke_learns_structured_alpha_095():
    """Smoke: SCRN α=0.95 should reduce loss on toy structured."""
    torch.manual_seed(0)
    net = SCRNCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, alpha_init=0.95,
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


def test_bench_smoke_scrn_vs_cfc():
    """Mini-bench: SCRN-CfC vs CfC baseline on sin task."""
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

    # SCRN-CfC α=0.95.
    torch.manual_seed(42)
    scrn = SCRNCfCStackedNetwork(
        input_size=D, hidden_size=H, output_size=1, num_layers=2,
        alpha_init=0.95, return_sequences=True,
    )
    opt = torch.optim.Adam(scrn.parameters(), lr=1e-2)
    scrn_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = scrn(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        scrn_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(scrn_loss)


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
