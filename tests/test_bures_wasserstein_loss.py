"""Round 189 — tests for Bures-Wasserstein loss (PRD #10-151)."""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.bures_wasserstein_loss import (
    _matrix_sqrt,
    bures_wasserstein2,
    combined_distdf_loss,
    joint_bures_wasserstein,
)


def test_matrix_sqrt_identity():
    """Matrix sqrt of identity is identity."""
    I = torch.eye(3)
    sqrt_I = _matrix_sqrt(I)
    assert torch.allclose(sqrt_I, I, atol=1e-4)


def test_matrix_sqrt_diagonal():
    """Matrix sqrt of diag(λ₁, λ₂, λ₃) is diag(√λ₁, √λ₂, √λ₃)."""
    A = torch.diag(torch.tensor([1.0, 4.0, 9.0, 16.0]))
    sqrt_A = _matrix_sqrt(A)
    expected = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(sqrt_A, expected, atol=1e-3)


def test_matrix_sqrt_squared_equals_original():
    """A^(1/2) @ A^(1/2) ≈ A."""
    torch.manual_seed(0)
    A = torch.randn(5, 5)
    A = A @ A.T + 0.1 * torch.eye(5)  # make SPD
    sqrt_A = _matrix_sqrt(A)
    A_reconstructed = sqrt_A @ sqrt_A
    assert torch.allclose(A_reconstructed, A, atol=1e-3)


def test_bw2_zero_when_identical():
    """BW²(μ, Σ; μ, Σ) = 0."""
    torch.manual_seed(0)
    A = torch.randn(5, 5)
    A = A @ A.T + 0.1 * torch.eye(5)
    mu = torch.randn(5)
    bw2 = bures_wasserstein2(mu, A, mu.clone(), A.clone())
    assert bw2.item() < 1e-3


def test_bw2_mean_only():
    """If covariances are zero, BW² reduces to ||μ₁ - μ₂||²."""
    mu1 = torch.tensor([1.0, 2.0, 3.0])
    mu2 = torch.tensor([1.5, 2.5, 3.5])
    zero_cov = torch.zeros(3, 3)
    bw2 = bures_wasserstein2(mu1, zero_cov, mu2, zero_cov)
    expected = ((mu1 - mu2) ** 2).sum().item()
    assert abs(bw2.item() - expected) < 1e-3


def test_bw2_symmetric():
    """BW²(μ₁,Σ₁; μ₂,Σ₂) = BW²(μ₂,Σ₂; μ₁,Σ₁)."""
    torch.manual_seed(42)
    mu1 = torch.randn(3)
    mu2 = torch.randn(3)
    A1 = torch.randn(3, 3)
    A1 = A1 @ A1.T + 0.1 * torch.eye(3)
    A2 = torch.randn(3, 3)
    A2 = A2 @ A2.T + 0.1 * torch.eye(3)
    bw2_12 = bures_wasserstein2(mu1, A1, mu2, A2).item()
    bw2_21 = bures_wasserstein2(mu2, A2, mu1, A1).item()
    assert abs(bw2_12 - bw2_21) < 1e-3


def test_bw2_non_negative():
    """BW² ≥ 0."""
    torch.manual_seed(0)
    mu1 = torch.randn(5)
    mu2 = torch.randn(5)
    A1 = torch.randn(5, 5)
    A1 = A1 @ A1.T + 0.1 * torch.eye(5)
    A2 = torch.randn(5, 5)
    A2 = A2 @ A2.T + 0.1 * torch.eye(5)
    bw2 = bures_wasserstein2(mu1, A1, mu2, A2)
    assert bw2.item() >= 0


def test_joint_bw2_zero_when_pred_eq_target():
    """If prediction == target, joint BW² = 0 (Z = Ẑ)."""
    torch.manual_seed(0)
    target = torch.randn(4, 8, 1)
    prediction = target.clone()
    bw2 = joint_bures_wasserstein(target, prediction)
    assert bw2.item() < 1e-3


def test_joint_bw2_shape_handling():
    """Joint BW² accepts [B, T, D] and returns scalar."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1)
    bw2 = joint_bures_wasserstein(target, prediction)
    assert bw2.dim() == 0
    assert math.isfinite(bw2.item())


def test_joint_bw2_increases_with_shift():
    """BW² increases as prediction diverges from target."""
    torch.manual_seed(0)
    target = torch.randn(4, 8, 1)
    pred_close = target + 0.1 * torch.randn_like(target)
    pred_far = target + 5.0 * torch.randn_like(target)
    bw_close = joint_bures_wasserstein(target, pred_close).item()
    bw_far = joint_bures_wasserstein(target, pred_far).item()
    assert bw_far > bw_close


def test_combined_loss_gamma_zero_equals_mse():
    """γ=0 → pure MSE."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1)
    mse = torch.nn.functional.mse_loss(prediction, target).item()
    combined = combined_distdf_loss(target, prediction, gamma=0.0).item()
    assert abs(mse - combined) < 1e-4


def test_combined_loss_gamma_one_equals_bw():
    """γ=1 → pure Bures-Wasserstein."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1)
    bw = joint_bures_wasserstein(target, prediction).item()
    combined = combined_distdf_loss(target, prediction, gamma=1.0).item()
    assert abs(bw - combined) < 1e-4


def test_combined_loss_middle_gamma():
    """γ=0.5 → 0.5*BW + 0.5*MSE."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1)
    combined = combined_distdf_loss(target, prediction, gamma=0.5).item()
    mse = torch.nn.functional.mse_loss(prediction, target).item()
    bw = joint_bures_wasserstein(target, prediction).item()
    expected = 0.5 * bw + 0.5 * mse
    assert abs(combined - expected) < 1e-4


def test_gradient_flows_to_prediction():
    """Gradient flows back to prediction through BW loss."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1, requires_grad=True)
    bw = joint_bures_wasserstein(target, prediction)
    bw.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum().item() > 0


def test_gradient_flows_combined_loss():
    """Gradient flows back through combined loss."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1, requires_grad=True)
    loss = combined_distdf_loss(target, prediction, gamma=0.5)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_smoke_train_bw_loss():
    """Smoke test: train a linear layer to minimize BW loss."""
    torch.manual_seed(0)
    # Per-sample variation so cov is non-degenerate
    target = torch.sin(torch.linspace(0, 4 * math.pi, 8)).reshape(1, 8, 1).expand(4, 8, 1).clone()
    target = target + 0.5 * torch.randn_like(target)  # add per-sample noise
    layer = torch.nn.Linear(1, 1, bias=True)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-1)
    initial_loss = None
    final_loss = 0.0
    for _ in range(50):
        opt.zero_grad()
        pred_seq = layer(target)  # [4, 8, 1]
        loss = joint_bures_wasserstein(target, pred_seq)
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()
        loss.backward()
        opt.step()
    assert initial_loss is not None
    assert math.isfinite(final_loss)


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
