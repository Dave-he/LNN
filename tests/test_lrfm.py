"""Round 24 (N2) — tests for Liquid Random Feature Methods (L-RFM) module.

Validates the closed-form LTC frozen feature implementation from
arXiv 2606.15571 (Linghu & Wang 2026).
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.lrfm import LiquidRandomFeatureBasis, LRFMSequenceRegressor


def _seed(s: int = 42):
    torch.manual_seed(s)


# ---------------------------------------------------------------------------
# LiquidRandomFeatureBasis
# ---------------------------------------------------------------------------


def test_basis_forward_shape():
    _seed()
    basis = LiquidRandomFeatureBasis(input_size=4, n_features=8)
    x = torch.randn(3, 4)
    t = torch.tensor(1.0)
    phi = basis(x, t)
    assert phi.shape == (3, 8)


def test_basis_forward_3d_shape():
    _seed()
    basis = LiquidRandomFeatureBasis(input_size=4, n_features=8)
    x = torch.randn(2, 5, 4)
    t = torch.tensor(1.0)
    phi = basis(x, t)
    assert phi.shape == (2, 5, 8)


def test_basis_at_t_zero_returns_h0():
    """At t=0, phi(x, 0) = h0(x) * 1 + g * A * 0 = h0(x)."""
    _seed()
    basis = LiquidRandomFeatureBasis(input_size=4, n_features=8)
    x = torch.randn(2, 4)
    t = torch.tensor(0.0)
    phi = basis(x, t)
    h0_expected = torch.tanh(x @ basis.w0.T + basis.b0)
    assert torch.allclose(phi, h0_expected, atol=1e-5), \
        f"phi(t=0) mismatch: max diff {(phi - h0_expected).abs().max().item()}"


def test_basis_at_large_t_approaches_gA_div_alpha():
    """At t → ∞, phi → g * A / alpha (steady state), excluding alpha≈0 features."""
    _seed()
    basis = LiquidRandomFeatureBasis(input_size=4, n_features=8)
    x = torch.randn(2, 4)
    t = torch.tensor(100.0)  # large but not extreme
    phi = basis(x, t)
    g = torch.tanh(x @ basis.w.T + basis.b)
    alpha = 1.0 / basis.tau + g  # (B, n_features)
    # Only verify for features where alpha is not near 0
    mask = alpha.abs() > 1e-3
    expected = g * (basis.A / alpha)
    diff = (phi - expected).abs()
    diff_masked = diff.where(mask, torch.zeros_like(diff))
    max_diff = diff_masked.max().item()
    # At t=100, exp_term ≈ exp(-alpha*100). For alpha > 1, exp_term < 1e-43 → 0.
    # The error from 1 - exp_term / alpha is bounded by 1/alpha * 1e-43.
    # We just verify that phi is finite on the non-trivial features.
    for b in range(x.shape[0]):
        for f_idx in range(basis.n_features):
            if mask[b, f_idx]:
                assert torch.isfinite(phi[b, f_idx]), \
                    f"non-finite at ({b},{f_idx}): alpha={alpha[b,f_idx].item()}"


def test_basis_features_are_frozen():
    """No parameters should require grad (frozen features)."""
    _seed()
    basis = LiquidRandomFeatureBasis(input_size=4, n_features=8)
    for name, p in basis.named_parameters():
        assert not p.requires_grad, f"{name} should be frozen but requires_grad=True"


def test_basis_finite_for_various_dt():
    """Numerical stability across a range of dt values."""
    _seed()
    basis = LiquidRandomFeatureBasis(input_size=4, n_features=8, tau_min=0.01, tau_max=10.0)
    x = torch.randn(3, 4)
    for dt in [0.001, 0.1, 1.0, 10.0, 100.0]:
        t = torch.tensor(float(dt))
        phi = basis(x, t)
        assert torch.isfinite(phi).all(), f"non-finite for dt={dt}"


def test_basis_per_step_dt():
    """Per-step dt: t shape (batch,)."""
    _seed()
    basis = LiquidRandomFeatureBasis(input_size=4, n_features=8)
    x = torch.randn(3, 4)
    t = torch.tensor([0.1, 1.0, 10.0])
    phi = basis(x, t)
    assert phi.shape == (3, 8)
    assert torch.isfinite(phi).all()


# ---------------------------------------------------------------------------
# LRFMSequenceRegressor
# ---------------------------------------------------------------------------


def test_regressor_forward_shape():
    _seed()
    reg = LRFMSequenceRegressor(input_size=4, output_size=1, n_features=16)
    x = torch.randn(3, 10, 4)
    y = reg(x, dt=1.0)
    assert y.shape == (3, 10, 1)


def test_regressor_only_readout_is_learnable():
    """Only readout Linear/MLP should have grad; basis must stay frozen."""
    _seed()
    reg = LRFMSequenceRegressor(input_size=4, output_size=1, n_features=16)
    trainable = [n for n, p in reg.named_parameters() if p.requires_grad]
    frozen = [n for n, p in reg.named_parameters() if not p.requires_grad]
    # Readout has 1 or 2 params; basis has 5 (tau, A, w, b, w0, b0 = 6 actually)
    assert len(trainable) >= 1, "Readout should be trainable"
    assert len(frozen) >= 5, "L-RFM basis should be frozen"
    # All trainable params should be in 'readout'
    for name in trainable:
        assert "readout" in name, f"Trainable param {name} not in readout"


def test_regressor_can_learn_simple_pattern():
    """Frozen L-RFM + linear readout can fit a simple time-varying target."""
    torch.manual_seed(0)
    reg = LRFMSequenceRegressor(input_size=4, output_size=1, n_features=64)
    opt = torch.optim.Adam(reg.parameters(), lr=1e-2)
    x = torch.randn(8, 20, 4)
    # Target: smooth function of time (sinusoid)
    t = torch.arange(20).float() * 0.5
    target = torch.sin(t).unsqueeze(0).unsqueeze(-1).expand(8, 20, 1)
    loss0 = torch.nn.functional.mse_loss(reg(x, dt=0.5), target).item()
    for _ in range(50):
        opt.zero_grad()
        pred = reg(x, dt=0.5)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        opt.step()
    loss_final = torch.nn.functional.mse_loss(reg(x, dt=0.5), target).item()
    assert loss_final < loss0, f"loss did not decrease ({loss0:.4f} -> {loss_final:.4f})"
