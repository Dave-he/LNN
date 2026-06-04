"""Tests for lnn/core/riemannian_ltc.py (PRD §10 RLSTG stage B).

Verifies the minimum viable implementation of the RLSTG tangent-space
LTC pattern (arXiv 2601.14115v1 §3.2-3.3):

1. Init state is on the manifold
2. Forward shape
3. Gradients flow to learnable params
4. No NaN for small inputs (with the autograd-safe logmap0 trick)
5. Invalid manifold name raises
6. End-to-end forward + loss + backward + step

Geoopt 0.5.1 limits: full ``expmap`` / ``logmap`` at arbitrary points
require parallel transport which is not autograd-supported in 0.5.1.
We use origin-based ``expmap0`` / ``logmap0`` instead, which gives
autograd but restricts to ``x = origin`` (we treat every step as
starting from the origin, which is a stage-B smoke simplification).
"""

import pytest
import torch

from lnn.core.riemannian_ltc import (
    RiemannianLTC,
    RiemannianLTCNetwork,
    TangentSpaceLTC,
    _get_manifold,
)


# 1. Init state is on the Hyperboloid
def test_init_state_on_manifold():
    """`init_state` returns (1, 0, 0, ..., 0), which satisfies ⟨x, x⟩_L = -1."""
    layer = RiemannianLTC(input_size=4, hidden_size=8)
    x0 = layer.init_state(batch_size=3, device=torch.device("cpu"))
    assert x0.shape == (3, 9)  # ambient = hidden + 1
    assert torch.allclose(x0[:, 0], torch.ones(3))
    # Inner product: -x[0]^2 + sum(x[1:]^2) = -1 + 0 = -1
    inner = -x0[:, 0] ** 2 + (x0[:, 1:] ** 2).sum(dim=-1)
    assert torch.allclose(inner, -torch.ones(3), atol=1e-6), \
        f"Init state not on Hyperboloid: inner={inner.tolist()}"


# 2. Forward shape
def test_forward_shape_riemannian_ltc():
    """Single-layer forward returns [B, d+1] (ambient dim)."""
    layer = RiemannianLTC(input_size=4, hidden_size=8)
    x0 = layer.init_state(batch_size=2, device=torch.device("cpu"))
    u = torch.randn(2, 4)
    x1 = layer(x0, u)
    assert x1.shape == (2, 9)


def test_forward_shape_riemannian_ltc_network():
    """Stacked network returns correct output shape for both return_sequences modes."""
    m = RiemannianLTCNetwork(input_size=4, hidden_size=8, output_size=2, num_layers=1, return_sequences=True)
    x = torch.randn(2, 5, 4)
    y = m(x)
    assert y.shape == (2, 5, 2)

    m2 = RiemannianLTCNetwork(input_size=4, hidden_size=8, output_size=2, num_layers=1, return_sequences=False)
    y2 = m2(x)
    assert y2.shape == (2, 2)


# 3. Gradients flow to learnable parameters
def test_gradient_flows_to_input_proj_and_tangent_ltc():
    """Backward reaches input_proj.weight and tangent_ltc.W_h."""
    layer = RiemannianLTC(input_size=4, hidden_size=8)
    x0 = layer.init_state(batch_size=2, device=torch.device("cpu"))
    u = torch.randn(2, 4)
    x1 = layer(x0, u)
    loss = x1.pow(2).sum()
    loss.backward()
    assert layer.input_proj.weight.grad is not None
    assert layer.input_proj.weight.grad.abs().sum() > 0
    assert layer.tangent_ltc.W_h.weight.grad is not None
    assert layer.tangent_ltc.W_h.weight.grad.abs().sum() > 0
    assert layer.tangent_ltc.alpha.grad is not None


# 4. No NaN for small inputs (with autograd-safe logmap0 + clamp)
def test_no_nan_for_small_inputs():
    """With small inputs and default dt=0.001 + max_tangent_norm=1.0, no NaN."""
    torch.manual_seed(0)
    m = RiemannianLTCNetwork(input_size=4, hidden_size=8, output_size=2, num_layers=1)
    x = torch.randn(2, 5, 4) * 0.1
    y = m(x)
    assert not torch.isnan(y).any(), f"NaN in forward output: {y}"


# 5. Invalid manifold name raises
def test_invalid_manifold_raises():
    """_get_manifold rejects unknown manifold names."""
    with pytest.raises(ValueError, match="Unknown manifold"):
        _get_manifold("bogus_manifold")


# 6. End-to-end forward + loss + backward + step
def test_end_to_end_loss_backward_step():
    """Full training step: forward + MSE + backward + optimizer step, no NaN."""
    torch.manual_seed(0)
    m = RiemannianLTCNetwork(input_size=4, hidden_size=8, output_size=2, num_layers=1)
    optim = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randn(2, 5, 4) * 0.1
    target = torch.zeros(2, 5, 2)
    for _ in range(3):
        optim.zero_grad()
        y = m(x)
        loss = (y - target).pow(2).mean()
        loss.backward()
        optim.step()
    # Loss should be finite
    assert not torch.isnan(loss), f"Loss became NaN: {loss}"


# 7. TangentSpaceLTC basic
def test_tangent_space_ltc_basic():
    """TangentSpaceLTC forward: h + dt * (-α ⊙ h + tanh(W_h h + W_u u + b))."""
    torch.manual_seed(0)
    cell = TangentSpaceLTC(dim=8)
    h = torch.randn(2, 8)
    u = torch.randn(2, 8)
    h_new = cell(h, u, dt=0.01)
    assert h_new.shape == (2, 8)
    # h + dt * delta, so h_new should be close to h (dt=0.01)
    assert (h_new - h).abs().max() < 1.0


# 8. Multi-step stable (with the clamp)
def test_multi_step_stable():
    """10 forward steps with default clamps: no NaN explosion."""
    torch.manual_seed(0)
    layer = RiemannianLTC(input_size=4, hidden_size=8)
    x = layer.init_state(batch_size=2, device=torch.device("cpu"))
    u = torch.randn(2, 4) * 0.1
    for _ in range(10):
        x = layer(x, u)
    assert not torch.isnan(x).any(), f"NaN after 10 steps: {x}"
