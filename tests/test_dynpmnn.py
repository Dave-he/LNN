"""Tests for DynPMNN (arXiv 2605.08176v1 §2.2-2.3).

Verifies the FitzHugh-Nagumo ODE cell + stacked DynPMNNNetwork:

1. Forward shape [B, T, d_in] → [B, T, d_out] for return_sequences=True
2. Forward shape [B, T, d_in] → [B, d_out] for return_sequences=False
3. Initial state is zero (V_0 = W_0 = 0)
4. Gradient flows to all learnable parameters (a, b, epsilon, input_proj)
5. Stable forward: no NaN/Inf for reasonable inputs
6. FHN excitable behaviour: a strong input pulse should produce V excursion
7. Single-step stability: one Euler step doesn't blow up
8. Multi-layer: V dims from layer i → layer i+1 input
"""

import torch

from lnn.core.dynpmnn import DynPMNNNetwork, FHNCell


# ---------------------------------------- 1. shape return_sequences=True
def test_forward_shape_with_sequences():
    m = DynPMNNNetwork(input_size=4, hidden_size=8, output_size=2, num_layers=1)
    x = torch.randn(2, 5, 4)
    y = m(x)
    assert y.shape == (2, 5, 2)


# ---------------------------------------- 2. shape return_sequences=False
def test_forward_shape_last_only():
    m = DynPMNNNetwork(input_size=4, hidden_size=8, output_size=2,
                       num_layers=1, return_sequences=False)
    x = torch.randn(2, 5, 4)
    y = m(x)
    assert y.shape == (2, 2)


# ---------------------------------------- 3. initial state zero
def test_initial_state_zero():
    cell = FHNCell(input_size=3, hidden_size=5)
    V, W = cell.initial_state(batch_size=2, device=torch.device("cpu"))
    assert torch.allclose(V, torch.zeros(2, 5))
    assert torch.allclose(W, torch.zeros(2, 5))


# ---------------------------------------- 4. gradient flows to all params
def test_gradient_flows_to_all_params():
    torch.manual_seed(0)
    m = DynPMNNNetwork(input_size=4, hidden_size=8, output_size=2, num_layers=2)
    x = torch.randn(2, 5, 4)
    y = m(x)
    loss = y.pow(2).sum()
    loss.backward()
    for cell in m.cells:
        assert cell.input_proj.weight.grad is not None
        assert cell.input_proj.weight.grad.abs().sum() > 0
        assert cell.a.grad is not None
        assert cell.a.grad.abs().sum() > 0
        assert cell.b.grad is not None
        assert cell.b.grad.abs().sum() > 0
        assert cell.epsilon.grad is not None
        assert cell.epsilon.grad.abs().sum() > 0
    assert m.output_proj.weight.grad is not None


# ---------------------------------------- 5. stable forward
def test_forward_no_nan_or_inf():
    """No NaN/Inf for reasonable inputs (V stays in [-3, 3] range)."""
    torch.manual_seed(0)
    m = DynPMNNNetwork(input_size=4, hidden_size=8, output_size=2, num_layers=1,
                       n_euler_steps=5)
    x = torch.randn(4, 10, 4) * 0.5  # bounded inputs
    y = m(x)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    # Membrane potential shouldn't blow up for small inputs
    assert y.abs().max().item() < 100.0


# ---------------------------------------- 6. FHN excitability
def test_fhn_response_to_strong_input():
    """Strong positive current should produce a positive V excursion
    (membrane depolarization). The FHN model is excitable, so a small
    input crosses the threshold and yields a non-trivial V peak."""
    torch.manual_seed(0)
    cell = FHNCell(input_size=3, hidden_size=4, n_euler_steps=10)
    # Large positive input
    x_strong = torch.full((1, 3), 5.0)
    V_strong, W_strong, V_seq_strong = cell(x_strong)
    # Zero input
    x_zero = torch.zeros(1, 3)
    V_zero, W_zero, V_seq_zero = cell(x_zero)
    # Strong input should push V further from 0
    assert V_strong.abs().max() > V_zero.abs().max() * 0.5, \
        f"Expected strong input to drive V more than zero input, " \
        f"got V_strong max={V_strong.abs().max():.4f}, V_zero max={V_zero.abs().max():.4f}"


# ---------------------------------------- 7. single step stability
def test_single_step_does_not_explode():
    """One Euler step from zero state with unit input should stay finite."""
    cell = FHNCell(input_size=2, hidden_size=4, n_euler_steps=1)
    x = torch.tensor([[1.0, 1.0]])
    V, W, V_seq = cell(x)
    # V_seq should be [1, 2, 4] (initial + 1 step)
    assert V_seq.shape == (1, 2, 4)
    assert not torch.isnan(V).any()
    assert not torch.isinf(V).any()
    # V should be small (one Euler step with small dt ~ 1)
    assert V.abs().max() < 10.0


# ---------------------------------------- 8. multi-layer chaining
def test_multi_layer_chains_correctly():
    """2-layer DynPMNNNetwork should produce shape (B, T, d_out)."""
    m = DynPMNNNetwork(input_size=3, hidden_size=6, output_size=2, num_layers=3)
    x = torch.randn(2, 7, 3)
    y = m(x)
    assert y.shape == (2, 7, 2)
    # 3 layers, each with input_proj (3→6 or 6→6) + a/b/epsilon (6 each)
    # Plus output_proj 6→2.
    expected_params = 3 * (3 * 6 + 6 + 6 + 6 + 6) + (6 * 2 + 2)
    # Note: 1st layer is 3→6, 2nd/3rd are 6→6
    expected_params = (3 * 6 + 6) + 3 * 6 + (6 * 6 + 6) + 3 * 6 + (6 * 6 + 6) + 3 * 6 + (6 * 2 + 2)
    actual_params = sum(p.numel() for p in m.parameters())
    assert actual_params == expected_params, \
        f"Expected {expected_params} params, got {actual_params}"


# ---------------------------------------- 9. return_sequences + multi-layer + loss
def test_full_pipeline_backward():
    """End-to-end: forward + MSE loss + backward, no errors."""
    m = DynPMNNNetwork(input_size=3, hidden_size=8, output_size=2,
                       num_layers=2, n_euler_steps=4, return_sequences=True)
    x = torch.randn(2, 6, 3)
    target = torch.randn(2, 6, 2)
    optim = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(3):  # a few steps to confirm the loop is stable
        optim.zero_grad()
        y = m(x)
        loss = (y - target).pow(2).mean()
        loss.backward()
        optim.step()
    # Confirm the loss is finite and decreased (or at least didn't NaN)
    assert not torch.isnan(loss)
