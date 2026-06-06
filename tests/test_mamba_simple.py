"""Tests for lnn.core.mamba_simple (round 73)."""

from __future__ import annotations

import pytest
import torch

from lnn.core.mamba_simple import SelectiveScanMamba, SelectiveScanMambaNetwork


# ---------------------------------------- 1. forward shape (B, T, D)
def test_forward_shape_sequences():
    m = SelectiveScanMamba(input_size=4, hidden_size=8)
    x = torch.randn(2, 5, 4)
    y = m(x, return_sequences=True)
    assert y.shape == (2, 5, 8)


# ---------------------------------------- 2. forward shape return_sequences=False
def test_forward_shape_last_only():
    m = SelectiveScanMamba(input_size=4, hidden_size=8)
    x = torch.randn(2, 5, 4)
    y = m(x, return_sequences=False)
    assert y.shape == (2, 8)


# ---------------------------------------- 3. gradient flows
def test_gradients_flow_to_all_params():
    torch.manual_seed(0)
    m = SelectiveScanMamba(input_size=4, hidden_size=8)
    x = torch.randn(2, 5, 4)
    y = m(x)
    y.pow(2).sum().backward()
    assert m.A_log.grad is not None and m.A_log.grad.abs().sum() > 0
    assert m.D.grad is not None and m.D.grad.abs().sum() > 0
    assert m.in_proj.weight.grad is not None and m.in_proj.weight.grad.abs().sum() > 0
    assert m.out_proj.weight.grad is not None and m.out_proj.weight.grad.abs().sum() > 0


# ---------------------------------------- 4. variable sequence length
@pytest.mark.parametrize("T", [1, 8, 64, 256])
def test_variable_sequence_length(T):
    m = SelectiveScanMamba(input_size=3, hidden_size=5)
    x = torch.randn(2, T, 3)
    y = m(x, return_sequences=True)
    assert y.shape == (2, T, 5)


# ---------------------------------------- 5. variable hidden size
@pytest.mark.parametrize("D", [4, 16, 64, 128])
def test_variable_hidden(D):
    m = SelectiveScanMamba(input_size=3, hidden_size=D)
    x = torch.randn(2, 10, 3)
    y = m(x, return_sequences=True)
    assert y.shape == (2, 10, D)


# ---------------------------------------- 6. numerical stability over T=512
def test_numerical_stability_long_sequence():
    torch.manual_seed(0)
    m = SelectiveScanMamba(input_size=2, hidden_size=8)
    x = torch.randn(2, 512, 2) * 1.0  # not too small, not too large
    y = m(x, return_sequences=True)
    assert torch.isfinite(y).all()
    # Hidden state should not blow up — y should be bounded.
    assert y.abs().max() < 1e6


# ---------------------------------------- 7. bidirectional forward
def test_bidirectional_forward_shape():
    m = SelectiveScanMamba(input_size=4, hidden_size=8, bidirectional=True)
    x = torch.randn(2, 16, 4)
    y = m(x, return_sequences=True)
    assert y.shape == (2, 16, 8)


# ---------------------------------------- 8. bidirectional differs from uni
def test_bidirectional_differs_from_uni():
    torch.manual_seed(0)
    m_uni = SelectiveScanMamba(input_size=4, hidden_size=8, bidirectional=False)
    torch.manual_seed(0)
    m_bi = SelectiveScanMamba(input_size=4, hidden_size=8, bidirectional=True)
    x = torch.randn(1, 16, 4)
    y_uni = m_uni(x)
    y_bi = m_bi(x)
    # Bi-directional average should differ from uni unless the input
    # happens to be perfectly symmetric, which it is not.
    assert not torch.allclose(y_uni, y_bi, atol=1e-6)


# ---------------------------------------- 9. input_size mismatch raises
def test_input_size_mismatch_raises():
    m = SelectiveScanMamba(input_size=4, hidden_size=8)
    with pytest.raises(ValueError):
        m(torch.randn(2, 5, 5))


# ---------------------------------------- 10. wrong rank raises
def test_wrong_rank_raises():
    m = SelectiveScanMamba(input_size=4, hidden_size=8)
    with pytest.raises(ValueError):
        m(torch.randn(2, 4))  # 2D, not 3D


# ---------------------------------------- 11. network stacks multiple layers
def test_network_multi_layer():
    n = SelectiveScanMambaNetwork(
        input_size=4, hidden_size=8, output_size=3, num_layers=2
    )
    x = torch.randn(2, 10, 4)
    y = n(x)
    assert y.shape == (2, 10, 3)


# ---------------------------------------- 12. network last-step output
def test_network_return_sequences_false():
    n = SelectiveScanMambaNetwork(
        input_size=4, hidden_size=8, output_size=3, num_layers=2, return_sequences=False
    )
    x = torch.randn(2, 10, 4)
    y = n(x)
    assert y.shape == (2, 3)
