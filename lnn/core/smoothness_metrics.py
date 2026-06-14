"""Round 91 (PRD #10-53): temporal smoothness metrics.

Used to test the claim from arXiv:2606.07670 (Li, Pal, Tan, June 2026)
that CfC cells produce smoother outputs in t than equivalent MLPs
because of the closed-form time-constant inductive bias.

Three metrics, all based on finite differences of a 1D signal:
- :func:`total_variation` — mean |y[i+1] - y[i]|. Lower = smoother.
- :func:`l2_derivative` — RMS finite-diff derivative. Lower = smoother.
- :func:`max_gradient` — max |f'(t)|. Lower = smoother.

All metrics are scale-invariant to the absolute magnitude of y — they
only care about how rapidly y changes. For unnormalised data, divide
the y values by their range before computing.
"""
from __future__ import annotations

import torch


def total_variation(y: torch.Tensor) -> float:
    """Mean absolute first difference of a 1D signal.

    Args:
        y: 1D tensor (n,).

    Returns:
        TV = mean(|y[i+1] - y[i]|). Lower = smoother.
    """
    if y.numel() < 2:
        return 0.0
    return float((y[1:] - y[:-1]).abs().mean().item())


def l2_derivative(y: torch.Tensor, dt: float = 1.0) -> float:
    """RMS finite-difference derivative.

    Args:
        y: 1D tensor (n,).
        dt: spacing between consecutive samples. Default 1.0.

    Returns:
        sqrt(mean((y[i+1] - y[i])^2)) / dt. Lower = smoother.
    """
    if y.numel() < 2:
        return 0.0
    d = (y[1:] - y[:-1]) / dt
    return float((d ** 2).mean().sqrt().item())


def max_gradient(y: torch.Tensor, dt: float = 1.0) -> float:
    """Max absolute finite-difference derivative.

    Args:
        y: 1D tensor (n,).
        dt: spacing between consecutive samples. Default 1.0.

    Returns:
        max(|y[i+1] - y[i]|) / dt. Lower = smoother.
    """
    if y.numel() < 2:
        return 0.0
    d = (y[1:] - y[:-1]) / dt
    return float(d.abs().max().item())


def smoothness_summary(y: torch.Tensor, dt: float = 1.0) -> dict:
    """All three smoothness metrics in one call.

    Returns:
        dict with keys: tv, l2_deriv, max_grad, n.
    """
    return {
        "tv": total_variation(y),
        "l2_deriv": l2_derivative(y, dt=dt),
        "max_grad": max_gradient(y, dt=dt),
        "n": int(y.numel()),
    }


def backward_coherence_loss(
    states: torch.Tensor,
    lambda_coeff: float = 0.001,
) -> torch.Tensor:
    """Backward-coherence penalty over a (T, d) hidden-state trajectory (round 98).

    Implements the backward-coherence regularization from
    arXiv:2606.08934 (Chang, June 2026) — *Backward Coherence and
    Hidden-State Stability in Recurrent Neural Networks: A
    Quasi-Reverse-Martingale Theory*.

    The penalty is::

        L = λ * mean_{t=0..T-2} ||h_{t+1} - h_t||^2

    i.e. the mean squared backward difference of the hidden-state
    sequence.  This is distinct from :func:`total_variation` (which
    is on a 1D output) and from :func:`max_gradient` (which is a max
    rather than a mean).  Backward coherence specifically penalizes
    the **hidden state**'s tendency to jump between consecutive
    steps — encouraging a quasi-reverse-martingale where ``h_t ≈
    E[h_{t+1}]``.

    Args:
        states: 2D tensor of shape ``(T, d)`` — a hidden-state
            trajectory collected from a recurrent cell over T steps.
        lambda_coeff: scaling factor λ applied to the penalty.
            A value of 0.0 disables the penalty entirely.

    Returns:
        aux_loss: scalar tensor ``λ * mean(||h_{t+1} - h_t||^2)``.
        Returns a 0-d tensor so it composes with the main task loss
        via simple addition.
    """
    if lambda_coeff == 0.0:
        return torch.zeros((), device=states.device if isinstance(states, torch.Tensor) else "cpu")
    if states.dim() == 1:
        states = states.unsqueeze(0)
    if states.dim() != 2:
        raise ValueError(f"expected 2D (T, d) tensor, got {states.dim()}D")
    if states.shape[0] < 2:
        return torch.zeros((), device=states.device)
    diffs = states[1:] - states[:-1]  # (T-1, d)
    penalty = (diffs ** 2).mean()
    return lambda_coeff * penalty
