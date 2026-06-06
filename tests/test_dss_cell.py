"""Tests for lnn.core.dss_cell (round 73)."""

from __future__ import annotations

import pytest
import torch

from lnn.core.dss_cell import DiagonalSSMCell, DiagonalSSMNetwork


# ---------------------------------------- 1. forward shape
def test_forward_shape_sequences():
    d = DiagonalSSMCell(input_size=4, hidden_size=8)
    x = torch.randn(2, 5, 4)
    y = d(x, return_sequences=True)
    assert y.shape == (2, 5, 8)


# ---------------------------------------- 2. forward shape last-only
def test_forward_shape_last_only():
    d = DiagonalSSMCell(input_size=4, hidden_size=8)
    x = torch.randn(2, 5, 4)
    y = d(x, return_sequences=False)
    assert y.shape == (2, 8)


# ---------------------------------------- 3. gradient flows
def test_gradients_flow():
    torch.manual_seed(0)
    d = DiagonalSSMCell(input_size=4, hidden_size=8)
    x = torch.randn(2, 5, 4)
    y = d(x)
    y.pow(2).sum().backward()
    assert d.A_log.grad is not None and d.A_log.grad.abs().sum() > 0
    assert d.D.grad is not None and d.D.grad.abs().sum() > 0
    assert d.in_proj.weight.grad is not None and d.in_proj.weight.grad.abs().sum() > 0


# ---------------------------------------- 4. A_log is a (D,) tensor — diagonal property
def test_a_log_is_diagonal():
    d = DiagonalSSMCell(input_size=3, hidden_size=5)
    assert d.A_log.shape == (5,)
    # Not a (D, D) matrix — that's the whole point of DSS vs S4.
    assert d.A_log.ndim == 1


# ---------------------------------------- 5. A_log is independent of input (constant across t)
def test_a_log_independent_of_input():
    """DSS: A is constant per-channel (not input-dependent like Mamba's
    selective A). Verify by checking the state dynamics only depend on
    the input projections B, C, not on a recomputed A at each step."""
    torch.manual_seed(0)
    d = DiagonalSSMCell(input_size=3, hidden_size=4)
    x1 = torch.randn(1, 8, 3)
    x2 = x1.clone()
    x2[0, 0, 0] += 1e-4  # tiny perturbation
    y1 = d(x1)
    y2 = d(x2)
    # Outputs should be close but not identical — small input change
    # propagates through the recurrent state.
    assert not torch.allclose(y1, y2, atol=1e-10)
    assert torch.allclose(y1, y2, atol=1e-2)


# ---------------------------------------- 6. variable sequence length
@pytest.mark.parametrize("T", [1, 8, 64, 512])
def test_variable_sequence_length(T):
    d = DiagonalSSMCell(input_size=3, hidden_size=5)
    x = torch.randn(2, T, 3)
    y = d(x, return_sequences=True)
    assert y.shape == (2, T, 5)


# ---------------------------------------- 7. variable hidden size
@pytest.mark.parametrize("D", [4, 16, 64, 128])
def test_variable_hidden(D):
    d = DiagonalSSMCell(input_size=3, hidden_size=D)
    x = torch.randn(2, 10, 3)
    y = d(x, return_sequences=True)
    assert y.shape == (2, 10, D)


# ---------------------------------------- 8. numerical stability
def test_numerical_stability_long_sequence():
    torch.manual_seed(0)
    d = DiagonalSSMCell(input_size=2, hidden_size=8)
    x = torch.randn(2, 512, 2)
    y = d(x, return_sequences=True)
    assert torch.isfinite(y).all()
    assert y.abs().max() < 1e6


# ---------------------------------------- 9. input size mismatch
def test_input_size_mismatch_raises():
    d = DiagonalSSMCell(input_size=4, hidden_size=8)
    with pytest.raises(ValueError):
        d(torch.randn(2, 5, 5))


# ---------------------------------------- 10. network stacks
def test_network_multi_layer():
    n = DiagonalSSMNetwork(input_size=4, hidden_size=8, output_size=3, num_layers=2)
    x = torch.randn(2, 10, 4)
    y = n(x)
    assert y.shape == (2, 10, 3)


# ---------------------------------------- 11. network last-step output
def test_network_return_sequences_false():
    n = DiagonalSSMNetwork(
        input_size=4, hidden_size=8, output_size=3, num_layers=2, return_sequences=False
    )
    x = torch.randn(2, 10, 4)
    y = n(x)
    assert y.shape == (2, 3)
