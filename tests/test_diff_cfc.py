"""Round 145 — tests for Difference Features CfC (PRD #10-107)."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.diff_cfc import DifferenceInputEncoder, DiffCfCNetwork


# ---------------------------------------------------------------------------
# Encoder tests
# ---------------------------------------------------------------------------


def test_encoder_concat_1_output_dim():
    """concat mode with n_diff=1 has 2*D dim."""
    enc = DifferenceInputEncoder(input_size=4, n_diff=1, mode="concat")
    assert enc.output_size == 8


def test_encoder_concat_2_output_dim():
    """concat mode with n_diff=2 has 3*D dim."""
    enc = DifferenceInputEncoder(input_size=4, n_diff=2, mode="concat")
    assert enc.output_size == 12


def test_encoder_concat_0_output_dim():
    """concat mode with n_diff=0 has 1*D dim (x only)."""
    enc = DifferenceInputEncoder(input_size=4, n_diff=0, mode="concat")
    assert enc.output_size == 4


def test_encoder_diff_only_1_output_dim():
    """diff_only mode with n_diff=1 has 1*D dim."""
    enc = DifferenceInputEncoder(input_size=4, n_diff=1, mode="diff_only")
    assert enc.output_size == 4


def test_encoder_diff_only_2_output_dim():
    """diff_only mode with n_diff=2 has 2*D dim."""
    enc = DifferenceInputEncoder(input_size=4, n_diff=2, mode="diff_only")
    assert enc.output_size == 8


def test_encoder_invalid_mode():
    """Invalid mode raises."""
    try:
        DifferenceInputEncoder(input_size=4, n_diff=1, mode="invalid")
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_encoder_invalid_ndiff_diff_only():
    """n_diff=0 with diff_only raises."""
    try:
        DifferenceInputEncoder(input_size=4, n_diff=0, mode="diff_only")
        assert False, "Should have raised"
    except ValueError:
        pass


def test_encoder_invalid_negative_ndiff():
    """n_diff < 0 raises."""
    try:
        DifferenceInputEncoder(input_size=4, n_diff=-1, mode="concat")
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_encoder_concat_1_first_order_diff():
    """Δx_t = x_t - x_{t-1}, with Δx_0 = 0."""
    enc = DifferenceInputEncoder(input_size=2, n_diff=1, mode="concat")
    x = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]])  # [1, 3, 2]
    out = enc(x)
    # Δx_0 = 0, Δx_1 = 2,3, Δx_2 = 4,6
    # Output: [x, Δx]
    expected = torch.tensor([[[1.0, 2.0, 0.0, 0.0],
                              [3.0, 5.0, 2.0, 3.0],
                              [7.0, 11.0, 4.0, 6.0]]])
    assert torch.allclose(out, expected)


def test_encoder_concat_2_second_order_diff():
    """Δ²x = Δ(Δx), with Δ²x_0 = 0 and Δ²x_1 = Δx_1 - 0 = Δx_1."""
    enc = DifferenceInputEncoder(input_size=2, n_diff=2, mode="concat")
    x = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]])  # [1, 3, 2]
    out = enc(x)
    # Δx = [0, 2, 4] for dim0, [0, 3, 6] for dim1
    # Δ²x = [0, 2, 2] for dim0, [0, 3, 3] for dim1
    # Output: [x, Δx, Δ²x]
    expected = torch.tensor([[[1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                              [3.0, 5.0, 2.0, 3.0, 2.0, 3.0],
                              [7.0, 11.0, 4.0, 6.0, 2.0, 3.0]]])
    assert torch.allclose(out, expected)


def test_encoder_handles_nan_input():
    """NaN in input is zero-filled before computing diffs."""
    enc = DifferenceInputEncoder(input_size=2, n_diff=1, mode="concat")
    x = torch.tensor([[[1.0, float("nan")], [3.0, 5.0]]])  # [1, 2, 2]
    out = enc(x)
    # NaN -> 0, so x_clean = [[1, 0], [3, 5]]
    # Δx = [0, 2] for dim0, [0, 5] for dim1
    # Output: [x_clean, Δx]
    assert torch.isfinite(out).all()
    expected = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                              [3.0, 5.0, 2.0, 5.0]]])
    assert torch.allclose(out, expected)


def test_encoder_diff_only_1():
    """diff_only with n_diff=1 outputs only Δx."""
    enc = DifferenceInputEncoder(input_size=2, n_diff=1, mode="diff_only")
    x = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]])
    out = enc(x)
    expected = torch.tensor([[[0.0, 0.0],
                              [2.0, 3.0],
                              [4.0, 6.0]]])
    assert torch.allclose(out, expected)


def test_encoder_diff_only_2():
    """diff_only with n_diff=2 outputs Δx and Δ²x."""
    enc = DifferenceInputEncoder(input_size=2, n_diff=2, mode="diff_only")
    x = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]])
    out = enc(x)
    # Δx = [0, 2, 4] for dim0, [0, 3, 6] for dim1
    # Δ²x = [0, 2, 2] for dim0, [0, 3, 3] for dim1
    expected = torch.tensor([[[0.0, 0.0, 0.0, 0.0],
                              [2.0, 3.0, 2.0, 3.0],
                              [4.0, 6.0, 2.0, 3.0]]])
    assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# DiffCfCNetwork tests
# ---------------------------------------------------------------------------


def test_network_init_concat_1():
    """Concat1 network has 2*D encoded dim."""
    net = DiffCfCNetwork(input_size=2, hidden_size=8, output_size=1, n_diff=1, mode="concat")
    assert net.encoder.output_size == 4


def test_network_init_concat_2():
    """Concat2 network has 3*D encoded dim."""
    net = DiffCfCNetwork(input_size=2, hidden_size=8, output_size=1, n_diff=2, mode="concat")
    assert net.encoder.output_size == 6


def test_network_init_diff_only():
    """Diff-only network has n_diff * D encoded dim."""
    net = DiffCfCNetwork(input_size=2, hidden_size=8, output_size=1, n_diff=2, mode="diff_only")
    assert net.encoder.output_size == 4


def test_network_forward_shape_concat_1():
    """Concat1 network returns [B, T, output_size]."""
    net = DiffCfCNetwork(input_size=2, hidden_size=8, output_size=1, n_diff=1, mode="concat")
    x = torch.randn(4, 16, 2)
    out = net(x)
    assert out.shape == (4, 16, 1)


def test_network_forward_shape_diff_only_2():
    """Diff-only n=2 network returns [B, T, output_size]."""
    net = DiffCfCNetwork(input_size=2, hidden_size=8, output_size=1, n_diff=2, mode="diff_only")
    x = torch.randn(4, 16, 2)
    out = net(x)
    assert out.shape == (4, 16, 1)


def test_network_forward_finite_with_nan():
    """Forward handles NaN inputs."""
    net = DiffCfCNetwork(input_size=2, hidden_size=8, output_size=1, n_diff=2, mode="concat")
    x = torch.randn(2, 16, 2)
    x[0, 5, 0] = float("nan")
    out = net(x)
    assert torch.isfinite(out).all()


def test_network_gradient_flows():
    """Gradient should reach CfC cell params."""
    net = DiffCfCNetwork(input_size=2, hidden_size=8, output_size=1, n_diff=1, mode="concat")
    x = torch.randn(2, 16, 2)
    out = net(x)
    out.sum().backward()
    # CfC cells[0] has f_gate[0].weight (Linear of input+hidden -> hidden).
    assert net.cfc.cells[0].f_gate[0].weight.grad is not None
    assert net.cfc.cells[0].f_gate[0].weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_learns_sin_concat_1():
    """Concat1 should reduce loss on toy sin (smoother features)."""
    torch.manual_seed(0)
    net = DiffCfCNetwork(input_size=2, hidden_size=8, output_size=1, n_diff=1, mode="concat")
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


def test_smoke_learns_structured_concat_1():
    """Concat1 should reduce loss on toy structured (regime boundary)."""
    torch.manual_seed(0)
    net = DiffCfCNetwork(input_size=2, hidden_size=8, output_size=1, n_diff=1, mode="concat")
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
# Bench-style smoke
# ---------------------------------------------------------------------------


def test_bench_smoke_diff_vs_cfc_on_sin():
    """Mini-bench: DiffCfC vs CfC on sin task."""
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

    # DiffCfC concat1.
    torch.manual_seed(42)
    diff_net = DiffCfCNetwork(
        input_size=D, hidden_size=H, output_size=1, n_diff=1, mode="concat", num_layers=2,
        return_sequences=True,
    )
    opt = torch.optim.Adam(diff_net.parameters(), lr=1e-2)
    diff_loss = 0.0
    for _ in range(5):
        opt.zero_grad()
        out = diff_net(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        diff_loss = loss.item()

    assert math.isfinite(cfc_loss)
    assert math.isfinite(diff_loss)


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
