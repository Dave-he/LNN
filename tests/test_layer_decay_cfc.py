"""Round 167 — tests for LayerDecay-CfC (per-layer β schedule) (PRD #10-129)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.layer_decay_cfc import (
    LayerDecayCfCCell,
    LayerDecayCfCStackedNetwork,
    make_layer_beta_schedule,
    make_ld_constant,
    make_ld_linear_k5,
    make_ld_reverse_k5,
    make_ld_linear_slow,
    make_ld_linear_fast,
    make_ld_linear_relu,
)


# ---------------------------------------------------------------------------
# Tests for schedule
# ---------------------------------------------------------------------------


def test_schedule_constant():
    """constant mode returns same betas for all layers."""
    s = make_layer_beta_schedule([0.7, 0.95], num_layers=3, mode="constant")
    assert s == [[0.7, 0.95], [0.7, 0.95], [0.7, 0.95]]


def test_schedule_linear():
    """linear mode ramps betas from min to max."""
    s = make_layer_beta_schedule([0.5, 0.99], num_layers=3, mode="linear")
    assert s[0] == [0.5, 0.5]
    assert s[1] == [0.745, 0.745]
    assert s[2] == [0.99, 0.99]


def test_schedule_reverse():
    """reverse mode ramps betas from max to min."""
    s = make_layer_beta_schedule([0.99, 0.5], num_layers=3, mode="reverse")
    # Hmm actually schedule takes min(betas_h) and max(betas_h)
    # so min=0.5, max=0.99 with reverse means high at l=0.
    assert s[0] == [0.99, 0.99]
    assert s[2] == [0.5, 0.5]


def test_schedule_single_layer():
    """num_layers=1 returns single list regardless of mode."""
    s = make_layer_beta_schedule([0.7, 0.95], num_layers=1, mode="linear")
    assert s == [[0.7, 0.95]]


# ---------------------------------------------------------------------------
# Tests for cell + network
# ---------------------------------------------------------------------------


def test_cell_init():
    """Cell initializes with correct shape."""
    cell = LayerDecayCfCCell(input_size=2, hidden_size=8, Kx=3, betas_h=[0.7, 0.95])
    assert cell.Kx == 3
    assert cell.Kh == 2
    assert cell.betas_h == [0.7, 0.95]
    assert cell.beta_x_raw.shape == (3, 2)


def test_cell_forward():
    """Cell forward returns tuple (h_next, new_ema_x_list, new_ema_h_list)."""
    cell = LayerDecayCfCCell(input_size=2, hidden_size=8, Kx=3, betas_h=[0.7, 0.95])
    x_t = torch.randn(2, 2)
    h_t = torch.randn(2, 8)
    ema_x = [torch.zeros(2, 2) for _ in range(3)]
    ema_h = [torch.zeros(2, 8) for _ in range(2)]
    layer_betas = [0.7, 0.95]
    h_next, new_ema_x, new_ema_h = cell(x_t, h_t, ema_x, ema_h, layer_betas)
    assert h_next.shape == (2, 8)
    assert len(new_ema_x) == 3
    assert len(new_ema_h) == 2
    assert new_ema_x[0].shape == (2, 2)
    assert new_ema_h[0].shape == (2, 8)


def test_network_factory_linear():
    """make_ld_linear_k5 factory creates 3-layer linear β network."""
    net = make_ld_linear_k5(input_size=2, hidden_size=8, output_size=1)
    assert net.num_layers == 3
    assert net.mode == "linear"
    assert net.Kx == 5
    assert net.Kh == 2
    # Layer betas should differ across layers (linear ramp).
    assert net.layer_betas_h[0][0] < net.layer_betas_h[1][0] < net.layer_betas_h[2][0]


def test_network_factory_constant():
    """make_ld_constant creates constant β network."""
    net = make_ld_constant(input_size=2, hidden_size=8, output_size=1)
    assert net.mode == "constant"
    assert net.layer_betas_h[0] == net.layer_betas_h[1] == net.layer_betas_h[2]


def test_network_forward_shape():
    """Network forward returns [B, T, output_size]."""
    net = make_ld_linear_k5(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 16, 1)


def test_network_no_sequences():
    """return_sequences=False returns [B, output_size]."""
    net = LayerDecayCfCStackedNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=3,
        Kx=5, betas_h=[0.7, 0.95], mode="linear", return_sequences=False,
    )
    x = torch.randn(2, 16, 2)
    y = net(x)
    assert y.shape == (2, 1)


def test_network_gradient_flows_all_layers():
    """Gradient reaches all layers."""
    net = make_ld_linear_k5(input_size=2, hidden_size=8, output_size=1)
    x = torch.randn(2, 16, 2)
    y = net(x)
    y.sum().backward()
    for cell in net.cells:
        assert cell.beta_x_raw.grad is not None


def test_smoke_learns_sin():
    """Linear β network should reduce loss on toy sin."""
    torch.manual_seed(0)
    net = make_ld_linear_k5(input_size=2, hidden_size=12, output_size=1)
    t = torch.linspace(0, 4 * math.pi, 16).unsqueeze(0).unsqueeze(-1).expand(2, 16, 1)
    x = torch.zeros(2, 16, 2)
    x[..., 0] = torch.sin(t.squeeze(-1))
    x[..., 1] = torch.cos(t.squeeze(-1))
    target = torch.sin(t.squeeze(-1)).unsqueeze(-1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    initial_loss = None
    final_loss = 0.0
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
    assert initial_loss is not None
    assert math.isfinite(final_loss)
    assert final_loss < initial_loss


def test_linear_vs_reverse_differs():
    """Linear and reverse should give different layer β."""
    net_l = make_ld_linear_k5(input_size=2, hidden_size=8, output_size=1)
    net_r = make_ld_reverse_k5(input_size=2, hidden_size=8, output_size=1)
    # Layer 0 of linear should be at min (0.5), reverse at max (0.99).
    assert net_l.layer_betas_h[0][0] == 0.5
    assert net_r.layer_betas_h[0][0] == 0.99
    assert net_l.layer_betas_h[-1][0] == 0.99
    assert net_r.layer_betas_h[-1][0] == 0.5


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
