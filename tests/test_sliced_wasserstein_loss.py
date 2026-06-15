"""Round 190 — tests for Sliced Wasserstein loss (PRD #10-152)."""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.sliced_wasserstein_loss import (
    _wasserstein1d_squared,
    combined_swd_loss,
    sliced_wasserstein2,
)


def test_w1d2_zero_when_identical():
    """1D W2²(x, x) = 0."""
    torch.manual_seed(0)
    x = torch.randn(10, 3)
    w = _wasserstein1d_squared(x, x.clone())
    assert w.item() < 1e-6


def test_w1d2_increases_with_shift():
    """W2² increases as distribution shifts."""
    torch.manual_seed(0)
    x = torch.randn(100, 1)
    y_close = x + 0.1 * torch.randn_like(x)
    y_far = x + 5.0 * torch.randn_like(x)
    w_close = _wasserstein1d_squared(x, y_close).item()
    w_far = _wasserstein1d_squared(x, y_far).item()
    assert w_far > w_close


def test_w1d2_non_negative():
    """W2² ≥ 0."""
    torch.manual_seed(0)
    x = torch.randn(50, 3)
    y = torch.randn(50, 3)
    w = _wasserstein1d_squared(x, y)
    assert w.item() >= 0


def test_swd_zero_when_pred_eq_target():
    """SWD(pred, pred) = 0."""
    torch.manual_seed(0)
    target = torch.randn(4, 8, 1)
    prediction = target.clone()
    swd = sliced_wasserstein2(target, prediction, n_projections=20, seed=0)
    assert swd.item() < 1e-6


def test_swd_increases_with_shift():
    """SWD increases as predictions diverge from target."""
    torch.manual_seed(0)
    target = torch.randn(4, 8, 1)
    pred_close = target + 0.1 * torch.randn_like(target)
    pred_far = target + 5.0 * torch.randn_like(target)
    swd_close = sliced_wasserstein2(target, pred_close, n_projections=20, seed=0).item()
    swd_far = sliced_wasserstein2(target, pred_far, n_projections=20, seed=0).item()
    assert swd_far > swd_close


def test_swd_shape_handling():
    """SWD accepts [B, T, D] and returns scalar."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1)
    swd = sliced_wasserstein2(target, prediction, n_projections=10, seed=0)
    assert swd.dim() == 0
    assert math.isfinite(swd.item())


def test_swd_reproducible_with_seed():
    """Same seed → same SWD value."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1)
    swd1 = sliced_wasserstein2(target, prediction, n_projections=10, seed=42).item()
    swd2 = sliced_wasserstein2(target, prediction, n_projections=10, seed=42).item()
    assert abs(swd1 - swd2) < 1e-6


def test_swd_converges_with_more_projections():
    """More projections → smaller variance in SWD estimate."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1)
    swds_low = [
        sliced_wasserstein2(target, prediction, n_projections=5, seed=i).item()
        for i in range(20)
    ]
    swds_high = [
        sliced_wasserstein2(target, prediction, n_projections=50, seed=i).item()
        for i in range(20)
    ]
    std_low = torch.tensor(swds_low).std().item()
    std_high = torch.tensor(swds_high).std().item()
    assert std_high < std_low, f"expected std_high ({std_high}) < std_low ({std_low})"


def test_combined_loss_gamma_zero_equals_mse():
    """γ=0 → pure MSE."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1)
    mse = torch.nn.functional.mse_loss(prediction, target).item()
    combined = combined_swd_loss(target, prediction, gamma=0.0, seed=0).item()
    assert abs(mse - combined) < 1e-4


def test_combined_loss_gamma_one_equals_swd():
    """γ=1 → pure SWD."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1)
    swd = sliced_wasserstein2(target, prediction, n_projections=10, seed=0).item()
    combined = combined_swd_loss(target, prediction, gamma=1.0, n_projections=10, seed=0).item()
    assert abs(swd - combined) < 1e-4


def test_gradient_flows_to_prediction():
    """Gradient flows back to prediction through SWD loss."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1, requires_grad=True)
    swd = sliced_wasserstein2(target, prediction, n_projections=10, seed=0)
    swd.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum().item() > 0


def test_gradient_flows_combined_loss():
    """Gradient flows back through combined SWD+MSE loss."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1, requires_grad=True)
    loss = combined_swd_loss(target, prediction, gamma=0.5, n_projections=10, seed=0)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_smoke_train_swd_loss():
    """Smoke test: train a linear layer to minimize SWD loss."""
    torch.manual_seed(0)
    target = torch.sin(torch.linspace(0, 4 * math.pi, 8)).reshape(1, 8, 1).expand(4, 8, 1).clone()
    target = target + 0.5 * torch.randn_like(target)
    layer = torch.nn.Linear(1, 1, bias=True)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-1)
    initial_loss = None
    final_loss = 0.0
    for _ in range(50):
        opt.zero_grad()
        pred_seq = layer(target)
        loss = combined_swd_loss(target, pred_seq, gamma=0.1, n_projections=20, seed=0)
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()
        loss.backward()
        opt.step()
    assert initial_loss is not None
    assert math.isfinite(final_loss)


def test_swd_no_matrix_sqrt():
    """SWD uses sort not matrix sqrt — just verify no crash with large D."""
    torch.manual_seed(0)
    target = torch.randn(2, 16, 2)
    prediction = torch.randn(2, 16, 2)
    swd = sliced_wasserstein2(target, prediction, n_projections=10, seed=0)
    assert torch.isfinite(swd)


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
