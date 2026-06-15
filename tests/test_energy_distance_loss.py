"""Round 191 — tests for Energy Distance loss (PRD #10-153)."""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.energy_distance_loss import (
    combined_energy_loss,
    energy_distance2,
)


def test_ed2_zero_when_identical():
    """D²(F, F) = 0 (all three terms equal when distributions identical)."""
    torch.manual_seed(0)
    x = torch.randn(10, 4)
    ed = energy_distance2(x, x.clone())
    assert ed.item() < 1e-4


def test_ed2_non_negative():
    """D² ≥ 0 by definition (energy metric)."""
    torch.manual_seed(0)
    x = torch.randn(20, 4)
    y = torch.randn(20, 4)
    ed = energy_distance2(x, y)
    assert ed.item() >= 0


def test_ed2_increases_with_shift():
    """D² increases as distributions diverge."""
    torch.manual_seed(0)
    target = torch.randn(20, 4)
    pred_close = target + 0.1 * torch.randn_like(target)
    pred_far = target + 5.0 * torch.randn_like(target)
    ed_close = energy_distance2(target, pred_close).item()
    ed_far = energy_distance2(target, pred_far).item()
    assert ed_far > ed_close


def test_ed2_symmetric():
    """D²(F, G) = D²(G, F) (symmetric)."""
    torch.manual_seed(42)
    x = torch.randn(10, 3)
    y = torch.randn(10, 3)
    ed_xy = energy_distance2(x, y).item()
    ed_yx = energy_distance2(y, x).item()
    assert abs(ed_xy - ed_yx) < 1e-4


def test_ed2_shape_handling():
    """ED accepts [B, T, D] and returns scalar."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1)
    ed = energy_distance2(target, prediction)
    assert ed.dim() == 0
    assert math.isfinite(ed.item())


def test_ed2_handles_large_dim():
    """ED handles larger feature dimensions without crashing."""
    torch.manual_seed(0)
    target = torch.randn(8, 16, 4)
    prediction = torch.randn(8, 16, 4)
    ed = energy_distance2(target, prediction)
    assert torch.isfinite(ed)


def test_combined_loss_gamma_zero_equals_mse():
    """γ=0 → pure MSE."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1)
    mse = torch.nn.functional.mse_loss(prediction, target).item()
    combined = combined_energy_loss(target, prediction, gamma=0.0).item()
    assert abs(mse - combined) < 1e-4


def test_combined_loss_gamma_one_equals_ed():
    """γ=1 → pure Energy Distance."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1)
    ed = energy_distance2(target, prediction).item()
    combined = combined_energy_loss(target, prediction, gamma=1.0).item()
    assert abs(ed - combined) < 1e-4


def test_combined_loss_middle_gamma():
    """γ=0.5 → 0.5*ED + 0.5*MSE."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1)
    combined = combined_energy_loss(target, prediction, gamma=0.5).item()
    mse = torch.nn.functional.mse_loss(prediction, target).item()
    ed = energy_distance2(target, prediction).item()
    expected = 0.5 * ed + 0.5 * mse
    assert abs(combined - expected) < 1e-4


def test_gradient_flows_to_prediction():
    """Gradient flows back to prediction through ED loss."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1, requires_grad=True)
    ed = energy_distance2(target, prediction)
    ed.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum().item() > 0


def test_gradient_flows_combined_loss():
    """Gradient flows back through combined ED+MSE loss."""
    torch.manual_seed(0)
    target = torch.randn(2, 8, 1)
    prediction = torch.randn(2, 8, 1, requires_grad=True)
    loss = combined_energy_loss(target, prediction, gamma=0.5)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_smoke_train_ed_loss():
    """Smoke test: train a linear layer to minimize combined loss."""
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
        loss = combined_energy_loss(target, pred_seq, gamma=0.1)
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()
        loss.backward()
        opt.step()
    assert initial_loss is not None
    assert math.isfinite(final_loss)


def test_ed2_3way_decomposition():
    """D² = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||] (manual decomposition)."""
    torch.manual_seed(0)
    B = 30
    target = torch.randn(B, 4)
    prediction = torch.randn(B, 4)
    # Manual decomposition
    cross = torch.cdist(target, prediction, p=2).mean()
    within_t = torch.cdist(target, target, p=2).mean()
    within_p = torch.cdist(prediction, prediction, p=2).mean()
    expected = 2.0 * cross - within_t - within_p
    actual = energy_distance2(target, prediction)
    assert abs(expected.item() - actual.item()) < 1e-4


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
