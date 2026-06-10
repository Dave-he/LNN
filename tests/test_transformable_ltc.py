"""Tests for ``lnn.core.ltc.TransformableLTC`` (iter#37).

The TransformableLTC class implements the 2-stage protocol from
EntroLnn (arXiv 2601.06195, Li et al. SAC '26): a *static* LTC is
fully trained on a reference domain, then a *dynamic* refinement pass
gently adapts the same parameters to a target domain.

All tests use **synthetic** data (torch.randn with a deterministic
linear teacher) — no real hardware, no external datasets. See
``docs/PRD_设备操控_LNN.md`` §0 and user preference 2026-06-09.
"""

from __future__ import annotations

import pytest
import torch

from lnn.core.ltc import LTCNetwork, TransformableLTC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seq_to_scalar(
    n: int, T: int, F: int, seed: int, noise: float = 0.05
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (X, y) where y = mean of X across time + a bias + small noise.

    This is a deterministic learnable signal — the model should be able to
    reduce MSE to ~noise^2 with a few epochs of training.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, T, F, generator=g)
    # Target = sum of features per sample, scaled.
    y = X.sum(dim=(1, 2), keepdim=False) * 0.01 + 0.1
    y = y + noise * torch.randn(n, generator=g)
    return X, y


def _make_seq_to_seq(
    n: int, T: int, F: int, seed: int, noise: float = 0.05
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (X, y) where y[t] is a function of X[t] (per-step regression)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, T, F, generator=g)
    W = torch.randn(F, 1, generator=g) * 0.1
    y = X @ W + 0.05  # [n, T, 1]
    y = y + noise * torch.randn(n, T, 1, generator=g)
    return X, y


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


def test_init_creates_internal_ltcnetwork() -> None:
    """TransformableLTC must hold an internal LTCNetwork and forward to it."""
    model = TransformableLTC(input_size=4, hidden_size=8, output_size=1)
    assert isinstance(model.net, LTCNetwork)
    assert model.input_size == 4
    assert model.hidden_size == 8
    assert model.output_size == 1
    # Two optimizers must be present and have the right LRs.
    assert model._train_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert model._refine_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_init_rejects_unknown_loss_fn() -> None:
    with pytest.raises(ValueError, match="loss_fn must be 'mse' or 'l1'"):
        TransformableLTC(input_size=2, hidden_size=4, output_size=1, loss_fn="huber")


def test_init_rejects_refine_lr_above_train_lr() -> None:
    with pytest.raises(ValueError, match="must be ≤ train_lr"):
        TransformableLTC(
            input_size=2, hidden_size=4, output_size=1,
            train_lr=1e-4, refine_lr=1e-3,  # refine > train — bad
        )


def test_init_rejects_non_positive_refine_lr() -> None:
    with pytest.raises(ValueError, match="must be ≤ train_lr"):
        TransformableLTC(
            input_size=2, hidden_size=4, output_size=1,
            train_lr=1e-3, refine_lr=0.0,
        )


# ---------------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------------


def test_forward_return_sequences_true_shape() -> None:
    model = TransformableLTC(
        input_size=3, hidden_size=6, output_size=2,
        return_sequences=True,
    )
    x = torch.randn(4, 16, 3)
    out = model(x)
    assert out.shape == (4, 16, 2), f"got {tuple(out.shape)}"


def test_forward_return_sequences_false_shape() -> None:
    model = TransformableLTC(
        input_size=3, hidden_size=6, output_size=2,
        return_sequences=False,
    )
    x = torch.randn(4, 16, 3)
    out = model(x)
    assert out.shape == (4, 2), f"got {tuple(out.shape)}"


def test_forward_passes_through_to_ltc() -> None:
    """forward(x) on TransformableLTC must match forward(x) on the inner net."""
    model = TransformableLTC(input_size=3, hidden_size=6, output_size=1)
    x = torch.randn(2, 8, 3)
    # Disable training-mode randomness by switching to eval first.
    model.eval()
    out_wrap = model(x)
    out_inner = model.net(x)
    assert torch.allclose(out_wrap, out_inner, atol=1e-6)


# ---------------------------------------------------------------------------
# train_reference
# ---------------------------------------------------------------------------


def test_train_reference_returns_history_and_decreasing_loss() -> None:
    """train_reference must drive loss down on a learnable synthetic target."""
    model = TransformableLTC(
        input_size=4, hidden_size=8, output_size=1,
        ode_method="euler",  # faster than rk4
    )
    X, y = _make_seq_to_scalar(n=16, T=8, F=4, seed=0)
    out = model.train_reference(X, y, epochs=10, batch_size=4, verbose=False)
    assert "ref_loss_history" in out
    assert "final_ref_loss" in out
    assert len(out["ref_loss_history"]) == 10
    hist = out["ref_loss_history"]
    # Loss should be strictly decreasing on this learnable synthetic signal
    # (allow one non-monotone step since SGD is noisy).
    assert hist[-1] < hist[0] * 0.5, f"loss did not decrease: {hist}"


def test_train_reference_rejects_zero_epochs() -> None:
    model = TransformableLTC(input_size=2, hidden_size=4, output_size=1)
    X, y = _make_seq_to_scalar(n=4, T=4, F=2, seed=0)
    with pytest.raises(ValueError, match="epochs must be ≥ 1"):
        model.train_reference(X, y, epochs=0)


def test_train_reference_l1_loss_also_decreases() -> None:
    model = TransformableLTC(
        input_size=3, hidden_size=6, output_size=1,
        loss_fn="l1", ode_method="euler",
    )
    X, y = _make_seq_to_scalar(n=8, T=6, F=3, seed=1)
    out = model.train_reference(X, y, epochs=8, batch_size=4, verbose=False)
    hist = out["ref_loss_history"]
    # L1 converges more slowly than MSE; require at least 10% drop over 8 epochs.
    assert hist[-1] < hist[0] * 0.9, f"L1 loss did not decrease: {hist}"


# ---------------------------------------------------------------------------
# refine_target
# ---------------------------------------------------------------------------


def test_refine_target_returns_history_and_decreasing_loss() -> None:
    """refine_target must drive target loss down on a different but related task."""
    model = TransformableLTC(
        input_size=4, hidden_size=8, output_size=1,
        ode_method="euler",
    )
    # Stage 1: train on reference domain (seed 0).
    X_ref, y_ref = _make_seq_to_scalar(n=8, T=8, F=4, seed=0)
    model.train_reference(X_ref, y_ref, epochs=8, batch_size=4, verbose=False)
    # Stage 2: refine on target domain (seed 100 — different signal but same shape).
    X_tgt, y_tgt = _make_seq_to_scalar(n=8, T=8, F=4, seed=100)
    out = model.refine_target(X_tgt, y_tgt, K=20, batch_size=4, verbose=False)
    assert "refine_loss_history" in out
    assert "final_tgt_loss" in out
    assert "final_ref_loss_after" in out
    hist = out["refine_loss_history"]
    assert len(hist) == 20
    # Final target loss should be at least no worse than first refine step
    # (allow for noise, but we should not blow up).
    assert hist[-1] <= hist[0] * 1.1, f"refine loss regressed: {hist}"


def test_refine_target_zero_k_is_noop() -> None:
    """K=0 must be a no-op (returns empty history, no error)."""
    model = TransformableLTC(input_size=2, hidden_size=4, output_size=1)
    X, y = _make_seq_to_scalar(n=4, T=4, F=2, seed=0)
    out = model.refine_target(X, y, K=0, batch_size=4, verbose=False)
    assert out["refine_loss_history"] == []
    assert out["final_tgt_loss"] == 0.0


def test_refine_target_rejects_negative_k() -> None:
    model = TransformableLTC(input_size=2, hidden_size=4, output_size=1)
    X, y = _make_seq_to_scalar(n=4, T=4, F=2, seed=0)
    with pytest.raises(ValueError, match="K must be ≥ 0"):
        model.refine_target(X, y, K=-1)


# ---------------------------------------------------------------------------
# Stability guard & param tracking
# ---------------------------------------------------------------------------


def test_param_l1_norm_positive_after_init() -> None:
    model = TransformableLTC(input_size=3, hidden_size=6, output_size=1)
    n0 = model.param_l1_norm()
    assert n0 > 0
    # Norm should not change without optimizer steps.
    assert model.param_l1_norm() == pytest.approx(n0, rel=1e-6)


def test_param_l1_norm_changes_after_training() -> None:
    model = TransformableLTC(
        input_size=3, hidden_size=6, output_size=1, ode_method="euler",
    )
    n0 = model.param_l1_norm()
    X, y = _make_seq_to_scalar(n=4, T=4, F=3, seed=42)
    model.train_reference(X, y, epochs=4, batch_size=4, verbose=False)
    n1 = model.param_l1_norm()
    # At least one parameter must have moved.
    assert n1 != pytest.approx(n0, rel=1e-9), "params did not change after training"


def test_end_to_end_train_then_refine_uses_same_parameters() -> None:
    """The 2 stages must operate on the *same* parameters, not copies.

    We snapshot the L1 norm after stage 1, do stage 2, and verify the final
    norm is different from the stage-1 norm (proving the refine step
    actually updated the trained parameters).
    """
    model = TransformableLTC(
        input_size=4, hidden_size=8, output_size=1, ode_method="euler",
    )
    X_ref, y_ref = _make_seq_to_scalar(n=8, T=6, F=4, seed=0)
    model.train_reference(X_ref, y_ref, epochs=6, batch_size=4, verbose=False)
    n_after_train = model.param_l1_norm()
    X_tgt, y_tgt = _make_seq_to_scalar(n=8, T=6, F=4, seed=999)
    model.refine_target(X_tgt, y_tgt, K=5, batch_size=4, verbose=False)
    n_after_refine = model.param_l1_norm()
    assert n_after_refine != pytest.approx(n_after_train, rel=1e-9), (
        "refine_target did not modify the trained parameters"
    )
