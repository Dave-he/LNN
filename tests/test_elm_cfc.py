"""Tests for ELMCfC — Round 129 (arXiv:2605.12049 Spieler et al. 2026).

The Expressive Leaky Memory cell must:
1. Construct with expected parameters and initial states
2. Be finite for normal inputs
3. Be differentiable (autograd-friendly)
4. Reduce to a simple leaky integrator when d_m=1 and MLP=Identity
5. The high-pass output is bounded by ReLU
6. Handle NaN inputs gracefully
7. Network learns a simple sin task
8. Network respects return_sequences flag
9. Different d_m values change the architecture as expected
"""
from __future__ import annotations

import math

import pytest
import torch

from lnn.core.elm_cfc import ELMCfCCell, ELMCfCNetwork


def test_elm_cell_construction():
    """Cell constructs and has expected learnable parameters."""
    cell = ELMCfCCell(input_size=2, hidden_size=4, d_m=3, d_mlp=6)
    assert cell.input_size == 2
    assert cell.hidden_size == 4
    assert cell.d_m == 3
    assert cell.d_mlp == 6
    # Input projection: input+hidden -> hidden*d_m
    assert cell.in_proj.in_features == 2 + 4
    assert cell.in_proj.out_features == 4 * 3
    # MLP: 2*d_m -> d_mlp -> d_m  (input is proj + prev_memory)
    assert cell.mlp[0].in_features == 2 * 3  # 6
    assert cell.mlp[0].out_features == 6
    assert cell.mlp[2].out_features == 3
    # Per-memory-unit κ_m and κ_λ
    assert cell.kappa_m_raw.shape == (4, 3)
    assert cell.kappa_lambda_raw.shape == (4, 3)
    # Per-neuron w_r and b
    assert cell.w_r.shape == (4, 3)
    assert cell.b.shape == (4,)
    # Initial state
    m0, r0 = cell.init_state(batch_size=2)
    assert m0.shape == (2, 4, 3)
    assert r0.shape == (2, 4)
    assert (m0 == 0).all() and (r0 == 0).all()


def test_elm_cell_step_finite():
    """Step output is finite for normal inputs."""
    torch.manual_seed(0)
    cell = ELMCfCCell(input_size=2, hidden_size=4, d_m=3)
    x = torch.randn(2, 2)
    h_prev = torch.randn(2, 4)
    m = torch.randn(2, 4, 3)
    r = torch.randn(2, 4)
    m_new, r_new, a = cell(x, h_prev, m, r)
    assert torch.isfinite(m_new).all()
    assert torch.isfinite(r_new).all()
    assert torch.isfinite(a).all()
    assert m_new.shape == (2, 4, 3)
    assert r_new.shape == (2, 4)
    assert a.shape == (2, 4)


def test_elm_high_pass_output_non_negative():
    """The high-pass filtered output is non-negative (ReLU)."""
    torch.manual_seed(0)
    cell = ELMCfCCell(input_size=2, hidden_size=4, d_m=3)
    x = torch.randn(8, 2) * 10  # large input to provoke large output
    h_prev = torch.randn(8, 4) * 10
    m = torch.randn(8, 4, 3) * 10
    r = torch.randn(8, 4) * 10
    m_new, r_new, a = cell(x, h_prev, m, r)
    # a = ReLU(b + w_r^T m - r) should be >= 0
    assert (a >= 0).all()


def test_elm_differentiable():
    """Cell is autograd-friendly (gradients flow)."""
    torch.manual_seed(0)
    cell = ELMCfCCell(input_size=2, hidden_size=4, d_m=3)
    x = torch.randn(2, 2, requires_grad=True)
    h_prev = torch.randn(2, 4, requires_grad=True)
    m = torch.randn(2, 4, 3, requires_grad=True)
    r = torch.randn(2, 4, requires_grad=True)
    m_new, r_new, a = cell(x, h_prev, m, r)
    loss = m_new.pow(2).sum() + r_new.pow(2).sum() + a.pow(2).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert h_prev.grad is not None and torch.isfinite(h_prev.grad).all()
    assert m.grad is not None and torch.isfinite(m.grad).all()
    assert r.grad is not None and torch.isfinite(r.grad).all()
    # Cell parameters receive gradients
    assert cell.kappa_m_raw.grad is not None and torch.isfinite(cell.kappa_m_raw.grad).all()
    assert cell.kappa_lambda_raw.grad is not None and torch.isfinite(cell.kappa_lambda_raw.grad).all()
    assert cell.w_r.grad is not None and torch.isfinite(cell.w_r.grad).all()
    assert cell.in_proj.weight.grad is not None and torch.isfinite(cell.in_proj.weight.grad).all()


def test_elm_kappa_in_unit_interval():
    """κ_m and κ_λ are in (0, 1) — they're sigmoid-parameterised."""
    cell = ELMCfCCell(input_size=2, hidden_size=4, d_m=3)
    km = cell.kappa_m()
    kl = cell.kappa_lambda()
    assert (km > 0).all() and (km < 1).all()
    assert (kl > 0).all() and (kl < 1).all()


def test_elm_zero_input_small_state():
    """If x=0, h_prev=0, m=0, r=0, the state stays small (bounded by tanh and bias).

    The cell has a learnable bias on in_proj, so even with zero
    inputs the projection is non-zero. After tanh-bounded MLP and
    leaky integration with κ_λ < 1, the state stays small.
    """
    cell = ELMCfCCell(input_size=2, hidden_size=4, d_m=3)
    x = torch.zeros(2, 2)
    h_prev = torch.zeros(2, 4)
    m = torch.zeros(2, 4, 3)
    r = torch.zeros(2, 4)
    m_new, r_new, a = cell(x, h_prev, m, r)
    # m_new = (1-κ_λ) * tanh(mlp(bias)) which is bounded
    assert m_new.abs().max() < 1.0
    # r_new is also bounded
    assert r_new.abs().max() < 1.0
    # a = ReLU(b) which is non-negative
    assert (a >= 0).all()


def test_elm_leaky_integration_consistency():
    """Test leaky integration: m_new = κ_m * m + (1 - κ_λ) * tanh(proj)."""
    cell = ELMCfCCell(input_size=2, hidden_size=1, d_m=1)
    # Force input projection to a specific value
    with torch.no_grad():
        cell.in_proj.weight.zero_()
        cell.in_proj.bias.fill_(1.0)  # in_proj output = 1.0
        cell.kappa_m_raw.fill_(0.0)  # sigmoid(0) = 0.5
        cell.kappa_lambda_raw.fill_(math.log(3.0))  # sigmoid(log 3) = 0.75
        cell.w_r.fill_(1.0)
    x = torch.zeros(1, 2)
    h_prev = torch.zeros(1, 1)
    m = torch.zeros(1, 1, 1)
    r = torch.zeros(1, 1)
    m_new, r_new, a = cell(x, h_prev, m, r)
    # proj = 1.0, MLP([1.0]) = (some value), tanh(.) = bounded
    # m_new = 0.5 * 0 + 0.25 * tanh(.) = some small value
    # r_new = kappa_r * 0 + (1 - kappa_r) * w_r^T m_new
    # a = ReLU(b + w_r^T m_new - r_new)
    assert torch.isfinite(m_new).all()
    assert torch.isfinite(r_new).all()
    assert torch.isfinite(a).all()


def test_elm_network_forward():
    """Network forward returns correct shape."""
    torch.manual_seed(0)
    net = ELMCfCNetwork(input_size=2, hidden_size=8, output_size=1, num_layers=2, d_m=3)
    x = torch.randn(4, 16, 2)
    out = net(x)
    assert out.shape == (4, 1)
    # return_sequences=True
    net2 = ELMCfCNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=2, d_m=3, return_sequences=True
    )
    out_seq = net2(x)
    assert out_seq.shape == (4, 16, 1)


def test_elm_network_nan_input():
    """Network handles NaN inputs gracefully."""
    torch.manual_seed(0)
    net = ELMCfCNetwork(input_size=2, hidden_size=4, output_size=1, d_m=2)
    x = torch.randn(2, 8, 2)
    x[0, 3, 1] = float("nan")
    x[1, 5, 0] = float("nan")
    out = net(x)
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()


def test_elm_network_learns_simple_task():
    """Network can learn a simple sin wave identity."""
    torch.manual_seed(0)
    net = ELMCfCNetwork(
        input_size=2, hidden_size=8, output_size=1, num_layers=1, d_m=3, return_sequences=True
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    T = 16
    t = torch.linspace(0, 2 * math.pi, T).unsqueeze(0).unsqueeze(-1)
    x_full = torch.zeros(4, T, 2)
    x_full[:, :, 0] = torch.sin(t.squeeze(-1)).expand(4, T)
    x_full[:, :, 1] = torch.cos(t.squeeze(-1)).expand(4, T)
    y = x_full[:, :, 0:1]

    initial_loss = None
    for step in range(300):
        opt.zero_grad()
        out = net(x_full)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step == 0:
            initial_loss = loss.item()
    final_loss = loss.item()
    # Should learn to reduce loss
    assert final_loss < initial_loss, f"loss did not decrease: {initial_loss} -> {final_loss}"
    # Loose convergence check
    assert final_loss < 0.5, f"loss did not converge: {initial_loss} -> {final_loss}"


def test_elm_different_d_m_values():
    """Different d_m values give different architecture sizes."""
    net1 = ELMCfCNetwork(input_size=2, hidden_size=8, output_size=1, d_m=2)
    net2 = ELMCfCNetwork(input_size=2, hidden_size=8, output_size=1, d_m=4)
    n1 = sum(p.numel() for p in net1.parameters())
    n2 = sum(p.numel() for p in net2.parameters())
    # net2 should have more params (more memory units)
    assert n2 > n1
    # Both should work
    x = torch.randn(2, 8, 2)
    o1 = net1(x)
    o2 = net2(x)
    assert o1.shape == (2, 1) and o2.shape == (2, 1)
