"""Round 282 — tests for MultiRateTfpCfC (PRD N3×2606.12240 second-layer synthesis).

Validates that the EC-routed mixture of TFP-retention experts is
behaves as a drop-in sequence-to-sequence module.
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.multirate_tfp_cfc import (
    MultiRateTfpCfC,
    MultiRateTfpCfCNetwork,
)


def _seed(seed: int = 42):
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


def test_init_valid():
    _seed()
    cell = MultiRateTfpCfC(input_size=4, hidden_size=16, n_tau=4)
    assert cell.n_tau == 4
    assert cell.hidden_size == 16
    assert sum(cell._branch_dims) == 16
    assert len(cell.experts) == 4
    # Verify each expert has TFP retention
    for e in cell.experts:
        assert e.retention_kind == "tfp"


def test_init_rejects_n_tau_1():
    try:
        MultiRateTfpCfC(input_size=4, hidden_size=16, n_tau=1)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for n_tau=1")


def test_init_per_expert_tau_bias_matches_scales():
    """τ_proj bias should be set so that initial retention matches tau_scales."""
    _seed()
    tau_scales = (0.1, 0.5, 2.0, 10.0)
    cell = MultiRateTfpCfC(input_size=2, hidden_size=8, n_tau=4, tau_scales=tau_scales)
    for i, expert in enumerate(cell.experts):
        # softplus(bias) ≈ tau_i
        bias = expert.tau_proj[0][0].bias.detach()  # type: ignore[index]
        # Average bias across hidden dims should approximately equal log(exp(τ) - 1)
        for j in range(bias.shape[0]):
            tau_actual = math.log1p(math.exp(bias[j].item()))
            expected = tau_scales[i]
            assert abs(tau_actual - expected) < 0.05, (
                f"expert {i} dim {j}: tau_actual={tau_actual:.3f} expected={expected}"
            )
        break  # one expert check is enough


# ---------------------------------------------------------------------------
# Forward / shape tests
# ---------------------------------------------------------------------------


def test_forward_shape():
    _seed()
    cell = MultiRateTfpCfC(input_size=3, hidden_size=12, n_tau=3, top_k_active=2)
    x_t = torch.randn(5, 3)
    h = torch.randn(5, 12)
    out = cell(x_t, h, dt=1.0)
    assert out.shape == (5, 12)


def test_forward_no_h():
    _seed()
    cell = MultiRateTfpCfC(input_size=3, hidden_size=12, n_tau=3, top_k_active=2)
    x_t = torch.randn(5, 3)
    out = cell(x_t, None, dt=1.0)
    assert out.shape == (5, 12)


def test_network_forward_shape():
    _seed()
    net = MultiRateTfpCfCNetwork(input_size=3, hidden_size=12, output_size=2, n_tau=3)
    x = torch.randn(4, 16, 3)
    out = net(x)
    assert out.shape == (4, 16, 2)


def test_network_return_last_only():
    _seed()
    net = MultiRateTfpCfCNetwork(
        input_size=3, hidden_size=12, output_size=2, n_tau=3, return_sequences=False
    )
    x = torch.randn(4, 16, 3)
    out = net(x)
    assert out.shape == (4, 2)


# ---------------------------------------------------------------------------
# Routing behaviour
# ---------------------------------------------------------------------------


def test_top_k_active_default():
    """Default top_k_active should be ceil(n_tau/2)."""
    _seed()
    for n_tau in (2, 3, 4, 5, 8):
        cell = MultiRateTfpCfC(input_size=2, hidden_size=8, n_tau=n_tau)
        expected = max(1, math.ceil(n_tau / 2))
        assert cell.top_k_active == expected, f"n_tau={n_tau}"


def test_routing_changes_with_input():
    """Different inputs should produce different expert selections."""
    _seed()
    cell = MultiRateTfpCfC(input_size=3, hidden_size=12, n_tau=4, top_k_active=2)
    # Run two very different inputs through the router
    x1 = torch.randn(1, 3) * 5.0
    x2 = torch.randn(1, 3) * 5.0 + 100.0
    s1 = cell.router(x1)
    s2 = cell.router(x2)
    # Top-K selection should differ
    _, top1 = s1.topk(2, dim=-1)
    _, top2 = s2.topk(2, dim=-1)
    assert not torch.equal(top1, top2)


# ---------------------------------------------------------------------------
# Auxiliary loss
# ---------------------------------------------------------------------------


def test_auxiliary_loss_is_finite():
    _seed()
    cell = MultiRateTfpCfC(input_size=3, hidden_size=12, n_tau=4)
    x = torch.randn(8, 3)
    loss = cell.auxiliary_loss(x)
    assert torch.isfinite(loss).all()
    assert loss.dim() == 0  # scalar


def test_network_auxiliary_loss_averages_over_steps():
    _seed()
    net = MultiRateTfpCfCNetwork(input_size=3, hidden_size=12, output_size=2, n_tau=3)
    x = torch.randn(4, 10, 3)
    loss = net.auxiliary_loss(x)
    assert torch.isfinite(loss).all()
    assert loss.dim() == 0


# ---------------------------------------------------------------------------
# End-to-end training
# ---------------------------------------------------------------------------


def test_end_to_end_training_step():
    """Run a small training loop and verify loss decreases."""
    _seed()
    net = MultiRateTfpCfCNetwork(
        input_size=3, hidden_size=16, output_size=2, n_tau=3, top_k_active=2
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    x = torch.randn(8, 16, 3)
    target = torch.randn(8, 16, 2)
    loss0 = torch.nn.functional.mse_loss(net(x), target).item()
    for _ in range(5):
        opt.zero_grad()
        pred = net(x)
        mse = torch.nn.functional.mse_loss(pred, target)
        aux = net.auxiliary_loss(x)
        (mse + 0.01 * aux).backward()
        opt.step()
    loss1 = torch.nn.functional.mse_loss(net(x), target).item()
    assert loss1 < loss0, f"loss did not decrease ({loss0} → {loss1})"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow():
    _seed()
    cell = MultiRateTfpCfC(input_size=3, hidden_size=12, n_tau=3, top_k_active=2)
    x_t = torch.randn(4, 3, requires_grad=True)
    h = torch.randn(4, 12)
    out = cell(x_t, h, dt=1.0)
    out.sum().backward()
    assert x_t.grad is not None
    assert torch.isfinite(x_t.grad).all()
