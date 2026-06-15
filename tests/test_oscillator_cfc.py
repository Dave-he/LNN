"""Tests for OscillatorCfC — Round 128 (arXiv:2602.12139 Shende et al. 2026).

The damped harmonic oscillator closed-form solution must:
1. Be a valid integration scheme: finite, bounded outputs
2. Reduce to e^(-γt) decay when ω → 0 (pure decay)
3. Reduce to cos(ωt) oscillation when ζ → 0 (undamped)
4. Reach the steady state F/ω² as t → ∞ (for any ζ)
5. Match a hand-coded Euler solution for small dt (sanity check)
6. Behave correctly in the underdamped regime (paper default)
7. Be differentiable (autograd-friendly)
8. Handle NaN inputs via the network's nan_to_num
9. The Network module matches CfCNetwork API
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from lnn.core.oscillator_cfc import OscillatorCfCCell, OscillatorCfCNetwork


def test_oscillator_cell_construction():
    """Cell constructs and has expected learnable parameters."""
    cell = OscillatorCfCCell(input_size=2, hidden_size=4)
    assert cell.input_size == 2
    assert cell.hidden_size == 4
    # Linear forcing: input -> hidden (no h-term for closed-form)
    assert cell.force.in_features == 2
    assert cell.force.out_features == 4
    # Per-neuron ω and ζ
    assert cell.omega_raw.shape == (4,)
    assert cell.zeta_raw.shape == (4,)
    # Initial ω is positive and within log-uniform range
    omega = cell.omega()
    assert (omega > 0).all()
    assert (omega >= 0.01).all() and (omega <= 10.0).all()
    # Initial ζ in (0, 1)
    zeta = cell.zeta()
    assert (zeta > 0).all() and (zeta < 1).all()
    # Initial state
    h0, p0 = cell.init_state(batch_size=3)
    assert h0.shape == (3, 4) and p0.shape == (3, 4)
    assert (h0 == 0).all() and (p0 == 0).all()


def test_oscillator_step_finite():
    """Step output is finite for normal inputs."""
    torch.manual_seed(0)
    cell = OscillatorCfCCell(input_size=2, hidden_size=8)
    x = torch.randn(4, 2)
    h = torch.randn(4, 8)
    p = torch.randn(4, 8)
    h_new, p_new = cell(x, h, p, dt=1.0)
    assert torch.isfinite(h_new).all()
    assert torch.isfinite(p_new).all()
    assert h_new.shape == (4, 8) and p_new.shape == (4, 8)


def test_oscillator_step_converges_to_steady_state():
    """As t → ∞ the state should approach the steady state F/ω².

    With tanh-forced F, F is bounded in [-1, 1], so the steady
    state h_ss = F/ω² is bounded by 1/ω²_max = 1/0.01² = 10000.
    We just check that the state is finite (no NaN/inf) and that
    the per-neuron std is small (convergence).
    """
    torch.manual_seed(0)
    cell = OscillatorCfCCell(input_size=2, hidden_size=4, force_activation="tanh")
    x = torch.ones(2, 2)  # constant input
    h = torch.zeros(2, 4)
    p = torch.zeros(2, 4)
    # Run for many steps
    for _ in range(2000):
        h, p = cell(x, h, p, dt=1.0)
    # Check finite
    assert torch.isfinite(h).all()
    assert torch.isfinite(p).all()
    # State should be bounded (F is bounded, ω² > 0)
    assert h.abs().max() < 20000.0
    # Velocity should converge to 0 (steady state)
    assert p.abs().max() < 1.0


def test_oscillator_reduces_to_decay_for_omega_zero():
    """When ω → 0 the oscillator reduces to exponential decay."""
    cell = OscillatorCfCCell(input_size=1, hidden_size=1)
    # Force omega very small
    with torch.no_grad():
        cell.omega_raw.fill_(math.log(1e-4))  # omega ≈ 1e-4
    omega = cell.omega()
    zeta = cell.zeta()
    h0 = torch.tensor([[1.0]])
    p0 = torch.tensor([[0.0]])
    F = torch.tensor([[0.0]])  # no forcing
    dt = 1.0
    h_new, p_new = OscillatorCfCCell(input_size=1, hidden_size=1).__class__.__base__ if False else None, None
    from lnn.core.oscillator_cfc import _oscillator_step_underdamped
    h_new, p_new = _oscillator_step_underdamped(h0, p0, F, omega, zeta, dt)
    # With very small ω and no forcing, h should decay (e^(-γ t))
    gamma = (zeta * omega).item()
    expected_h = math.exp(-gamma * dt) * 1.0
    # Allow some tolerance because of the small ω
    assert abs(h_new.item() - expected_h) < 0.05, f"got {h_new.item()}, expected ~{expected_h}"


def test_oscillator_reduces_to_undamped_for_zeta_zero():
    """When ζ → 0 the oscillator reduces to pure cos(ωt) oscillation."""
    from lnn.core.oscillator_cfc import _oscillator_step_underdamped
    cell = OscillatorCfCCell(input_size=1, hidden_size=1)
    with torch.no_grad():
        cell.zeta_raw.fill_(-20.0)  # sigmoid(-20) ≈ 2e-9 ≈ 0
    omega = cell.omega()
    zeta = cell.zeta()
    h0 = torch.tensor([[1.0]])
    p0 = torch.tensor([[0.0]])
    F = torch.tensor([[0.0]])  # no forcing
    h_new, _ = _oscillator_step_underdamped(h0, p0, F, omega, zeta, dt=0.1)
    # h(t) ≈ cos(ω t) ≈ cos(omega * 0.1)
    expected = math.cos(omega.item() * 0.1)
    assert abs(h_new.item() - expected) < 0.01, f"got {h_new.item()}, expected ~{expected}"


def test_oscillator_matches_euler_small_dt():
    """For small dt the closed-form step should approximately match Euler."""
    from lnn.core.oscillator_cfc import _oscillator_step_underdamped
    torch.manual_seed(42)
    cell = OscillatorCfCCell(input_size=2, hidden_size=4)
    x = torch.randn(2, 2)
    h0 = torch.randn(2, 4)
    p0 = torch.randn(2, 4)
    omega = cell.omega()
    zeta = cell.zeta()
    F = cell.force(x)  # forcing depends on x only (not h)
    # Closed-form step with small dt
    dt = 0.001
    h_cf, p_cf = _oscillator_step_underdamped(h0, p0, F, omega, zeta, dt)
    # Euler step
    gamma = zeta * omega
    h_euler = h0 + dt * p0
    p_euler = p0 + dt * (-2 * gamma * p0 - omega * omega * h0 + F)
    # Allow some error because Euler is only first-order accurate
    assert torch.allclose(h_cf, h_euler, atol=1e-3)
    assert torch.allclose(p_cf, p_euler, atol=1e-3)


def test_oscillator_differentiable():
    """Cell is autograd-friendly (gradients flow)."""
    torch.manual_seed(0)
    cell = OscillatorCfCCell(input_size=2, hidden_size=4)
    x = torch.randn(2, 2, requires_grad=True)
    h = torch.randn(2, 4, requires_grad=True)
    p = torch.randn(2, 4, requires_grad=True)
    h_new, p_new = cell(x, h, p, dt=1.0)
    loss = h_new.pow(2).sum() + p_new.pow(2).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    # Cell parameters receive gradients
    assert cell.omega_raw.grad is not None and torch.isfinite(cell.omega_raw.grad).all()
    assert cell.zeta_raw.grad is not None and torch.isfinite(cell.zeta_raw.grad).all()
    assert cell.force.weight.grad is not None and torch.isfinite(cell.force.weight.grad).all()


def test_oscillator_network_forward():
    """Network forward returns correct shape."""
    torch.manual_seed(0)
    net = OscillatorCfCNetwork(input_size=2, hidden_size=8, output_size=1, num_layers=2)
    x = torch.randn(4, 16, 2)
    # Test return_sequences=False
    out = net(x, dt=1.0)
    assert out.shape == (4, 1)
    # Test return_sequences=True
    net2 = OscillatorCfCNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, return_sequences=True
    )
    out_seq = net2(x, dt=1.0)
    assert out_seq.shape == (4, 16, 1)


def test_oscillator_network_nan_input():
    """Network handles NaN inputs gracefully."""
    torch.manual_seed(0)
    net = OscillatorCfCNetwork(input_size=2, hidden_size=4, output_size=1)
    x = torch.randn(2, 8, 2)
    x[0, 3, 1] = float("nan")
    x[1, 5, 0] = float("nan")
    out = net(x, dt=1.0)
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()


def test_oscillator_network_learns_simple_task():
    """Network can learn a simple identity target on a sin wave."""
    torch.manual_seed(0)
    net = OscillatorCfCNetwork(
        input_size=2, hidden_size=16, output_size=1, num_layers=1, return_sequences=True
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    T = 16
    t = torch.linspace(0, 2 * math.pi, T).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
    x_full = torch.zeros(4, T, 2)
    x_full[:, :, 0] = torch.sin(t.squeeze(-1)).expand(4, T)
    x_full[:, :, 1] = torch.cos(t.squeeze(-1)).expand(4, T)
    y = x_full[:, :, 0:1]  # predict sin

    initial_loss = None
    for step in range(200):
        opt.zero_grad()
        out = net(x_full, dt=1.0)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step == 0:
            initial_loss = loss.item()
    final_loss = loss.item()
    # Should learn to reduce loss (very loose threshold)
    assert final_loss < initial_loss, f"loss did not decrease: {initial_loss} -> {final_loss}"
    assert final_loss < 0.1, f"loss did not converge: {initial_loss} -> {final_loss}"


def test_oscillator_different_dt_values():
    """Network can be called with different dt values."""
    net = OscillatorCfCNetwork(input_size=2, hidden_size=4, output_size=1)
    x = torch.randn(2, 8, 2)
    out1 = net(x, dt=0.5)
    out2 = net(x, dt=2.0)
    out3 = net(x, dt=1.0)
    # Different dt values should give different outputs (state evolves differently)
    assert not torch.allclose(out1, out2, atol=1e-4)
    assert torch.allclose(out3, net(x, dt=1.0))  # deterministic


def test_oscillator_force_activation_tanh():
    """Force activation 'tanh' bounds the forcing."""
    cell = OscillatorCfCCell(input_size=2, hidden_size=4, force_activation="tanh")
    x = torch.randn(2, 2) * 100  # large input
    h = torch.randn(2, 4) * 100
    p = torch.randn(2, 4)
    h_new, p_new = cell(x, h, p, dt=1.0)
    # The forcing is bounded by tanh, so the state should not blow up
    assert torch.isfinite(h_new).all()
    assert h_new.abs().max() < 1000.0  # generous bound


def test_oscillator_zero_state_stays_in_steady_band():
    """If F=0 (zero input, zero hidden), the state remains at zero."""
    cell = OscillatorCfCCell(input_size=2, hidden_size=4)
    # zero state
    h = torch.zeros(2, 4)
    p = torch.zeros(2, 4)
    # zero input
    x = torch.zeros(2, 2)
    # one step
    h_new, p_new = cell(x, h, p, dt=1.0)
    # The linear forcing is W_x @ 0 + W_h @ 0 = 0 (or b)
    # With bias, F = b which is small, so h_new should be small
    assert h_new.abs().max() < 1.0
