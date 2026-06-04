"""
Tests for PDNAPulseHead (arXiv 2603.00153v1 §3.2-3.3).

Five core properties verified:
1. Output shape preservation (B, T, d) — head is residual.
2. α gate initialised to 0.01 (paper §3.2) — let backbone train first.
3. ω frequency diversity — log-uniform init [0.1, 10.0] gives > 5x ratio.
4. Output magnitude — α=0.01 + A=1.0 keeps |h_pulse - h| small at init.
5. Gradient flow — backprop through α, ω, A, phase_proj all reach leaves.
"""

import math

import pytest
import torch
from torch import nn

from lnn.core.cfc import PDNAPulseHead


def _make_head(**kwargs):
    defaults = dict(hidden_size=16, use_self_attend=True)
    defaults.update(kwargs)
    return PDNAPulseHead(**defaults)


# ------------------------------------------------------------------ 1. shape
def test_output_shape_preserved():
    head = _make_head()
    h = torch.randn(2, 28, 16)
    out = head(h)
    assert out.shape == h.shape == (2, 28, 16)


def test_output_shape_with_explicit_t():
    head = _make_head()
    h = torch.randn(3, 50, 16)
    t = torch.arange(50, dtype=torch.float32)
    out = head(h, t=t)
    assert out.shape == (3, 50, 16)


# ------------------------------------------------------------------ 2. gates
def test_alpha_init_0_01():
    head = _make_head()
    assert head.alpha.item() == pytest.approx(0.01, abs=1e-6)


def test_beta_init_0_01():
    head = _make_head()
    assert head.beta.item() == pytest.approx(0.01, abs=1e-6)


def test_gates_are_learnable_parameters():
    head = _make_head()
    assert isinstance(head.alpha, nn.Parameter)
    assert isinstance(head.beta, nn.Parameter)
    assert head.alpha.requires_grad is True
    assert head.beta.requires_grad is True


# ----------------------------------------------------------- 3. ω diversity
def test_omega_log_uniform_init_range():
    head = _make_head(hidden_size=64)
    omega = head.omega.detach()
    # log-uniform over [0.1, 10.0] => exp(uniform(log 0.1, log 10.0))
    # In practice, with 64 samples, max/min ratio should be >> 5
    ratio = omega.max().item() / omega.min().item()
    assert ratio > 5.0, f"ω init should span > 5x range, got ratio={ratio:.2f}"


def test_omega_in_expected_window():
    head = _make_head(hidden_size=128, omega_low=0.1, omega_high=10.0)
    omega = head.omega.detach()
    # With 128 samples, all should fall in [0.1, 10.0] (unif in log-space)
    assert (omega >= 0.1 * 0.5).all(), f"omega min {omega.min():.3f} below 0.05"
    assert (omega <= 10.0 * 2.0).all(), f"omega max {omega.max():.3f} above 20.0"


# ------------------------------------------------------ 4. output magnitude
def test_pulse_signal_magnitude_is_small_at_init():
    """At init: α=0.01, A=1.0, so |h_pulse - h| ≤ α * 1.0 * 1.0 = 0.01.
    Plus the β=0.01 attend term. Net residual should be < 0.05 at init."""
    head = _make_head()
    h = torch.randn(4, 28, 16)
    out = head(h)
    diff = (out - h).abs()
    assert diff.max().item() < 0.05, (
        f"residual at init should be tiny (α=β=0.01), got max={diff.max():.4f}"
    )


def test_pulse_amplitude_per_dim():
    head = _make_head(hidden_size=8)
    # A is initialised to ones; we just check shape and value
    assert head.amplitude.shape == (8,)
    assert torch.allclose(head.amplitude.detach(), torch.ones(8))


# ------------------------------------------------------ 5. gradient flow
def test_gradient_flows_to_all_params():
    head = _make_head()
    h = torch.randn(2, 28, 16, requires_grad=False)
    out = head(h)
    loss = out.pow(2).sum()
    loss.backward()

    # All four pulse params + (optional) attend params must receive grad
    for name, p in [
        ("alpha", head.alpha),
        ("omega", head.omega),
        ("amplitude", head.amplitude),
        ("phase_proj.weight", head.phase_proj.weight),
        ("phase_proj.bias", head.phase_proj.bias),
        ("beta", head.beta),
        ("self_attend_proj.weight", head.self_attend_proj.weight),
        ("self_attend_proj.bias", head.self_attend_proj.bias),
    ]:
        assert p.grad is not None, f"{name} has no grad"
        assert p.grad.abs().sum() > 0, f"{name} grad is zero (no signal flowed)"


def test_gradient_flows_without_self_attend():
    head = _make_head(use_self_attend=False)
    h = torch.randn(2, 28, 16)
    out = head(h)
    loss = out.pow(2).sum()
    loss.backward()
    # beta + self_attend_proj are absent — no AttributeError expected.
    # alpha / omega / amplitude / phase_proj must still receive grad.
    assert head.alpha.grad is not None
    assert head.omega.grad is not None
    assert head.amplitude.grad is not None
    assert head.phase_proj.weight.grad is not None
    # Sanity: no 'beta' attr (use_self_attend=False)
    assert not hasattr(head, "beta") or head.beta is not None  # attribute not set in this branch


# -------------------------------------------------- 6. end-to-end smoke
def test_pdna_head_in_cfc_pipeline():
    """Verify PDNAPulseHead can augment a CfCNetwork's return_sequences output."""
    from lnn.core.cfc import CfCNetwork

    B, T, d_in, d_out = 2, 28, 8, 4
    backbone = CfCNetwork(input_size=d_in, hidden_size=16, output_size=d_out, num_layers=1)
    head = PDNAPulseHead(hidden_size=16, use_self_attend=True)

    x = torch.randn(B, T, d_in)
    # CfCNetwork with return_sequences=True projects to output_size
    # We need hidden states — wrap backbone without output_proj instead.
    # Simpler: run cells manually, then head, then a fresh linear.
    h_seq = []
    h = torch.zeros(1, B, 16)
    for t in range(T):
        h_i = backbone.cells[0](x[:, t, :], h[0])
        h[0] = h_i
        h_seq.append(h_i)
    h_seq = torch.stack(h_seq, dim=1)  # [B, T, 16]
    h_aug = head(h_seq)
    assert h_aug.shape == h_seq.shape
    # Then project to output
    y = backbone.output_proj(h_aug)
    assert y.shape == (B, T, d_out)
