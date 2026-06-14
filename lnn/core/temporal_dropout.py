"""Round 92 (PRD #10-54): temporal dropout helper.

Used to test the claim from arXiv:2605.27467 (Thu/Oo/Supnithi, May 2026)
that CfC degrades more gracefully than LSTM under temporal dropout
(missing observations in the input sequence).

This module provides a single helper: :func:`temporal_dropout`, which
randomly masks a fraction p of (t, y) pairs in a 1D sequence.

For the round 92 bench, we mask by zeroing out y values (preserves
the t grid so the model still sees the same input dimensionality).
"""
from __future__ import annotations

import torch


def temporal_dropout(
    t: torch.Tensor,
    y: torch.Tensor,
    p: float,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly mask p fraction of y values by setting them to 0.

    Args:
        t: 1D tensor of time values, shape (N,).
        y: 1D tensor of target values, shape (N,).
        p: dropout probability in [0, 1]. 0 = no dropout, 1 = all masked.
        seed: optional RNG seed for reproducibility.

    Returns:
        (t, y_masked) where y_masked is the input y with a fraction
        p of its values replaced by 0.
    """
    if p < 0 or p > 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if p == 0:
        return t, y
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
        mask = torch.rand(y.shape, generator=gen) > p
    else:
        mask = torch.rand_like(y) > p
    y_masked = y * mask.to(y.dtype)
    return t, y_masked


def dropout_mask(
    n: int,
    p: float,
    seed: int | None = None,
) -> torch.Tensor:
    """Return a boolean keep-mask of shape (n,). True = keep, False = drop.

    Useful for callers who want to apply the mask themselves (e.g., to
    multiple tensors aligned on the same time grid).
    """
    if p < 0 or p > 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if p == 0:
        return torch.ones(n, dtype=torch.bool)
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
        return torch.rand(n, generator=gen) > p
    return torch.rand(n) > p


def input_dropout(
    t: torch.Tensor,
    y: torch.Tensor,
    p: float,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round 93 (PRD #10-55): input-side temporal dropout.

    Conceptually the caller passes the returned ``y_masked`` as INPUT
    to the model (not as the loss target). Functionally equivalent to
    :func:`temporal_dropout` — the distinction is semantic, indicating
    that the caller should treat the masked values as missing
    observations rather than missing labels.

    For stateless models (MLP, CfC stateless), this is identical to
    :func:`temporal_dropout`. For stateful models (LSTM, GRU), the
    masked values corrupt the running state, so the two helpers are
    NOT equivalent in their effect on the model.

    Args:
        t: 1D tensor of time values, shape (N,).
        y: 1D tensor of input values, shape (N,).
        p: dropout probability in [0, 1]. 0 = no dropout, 1 = all masked.
        seed: optional RNG seed for reproducibility.

    Returns:
        (t, y_masked) where y_masked is the input y with a fraction
        p of its values replaced by 0.
    """
    return temporal_dropout(t, y, p, seed=seed)


def apply_input_dropout_to_input(
    t: torch.Tensor,
    y: torch.Tensor,
    p: float,
    seed: int | None = None,
) -> torch.Tensor:
    """Helper: return ONLY the masked y (caller already has t).

    Convenience wrapper for callers that only need the masked y tensor.
    """
    _, y_masked = input_dropout(t, y, p, seed=seed)
    return y_masked
