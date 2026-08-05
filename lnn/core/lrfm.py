"""Liquid Random Feature Methods (L-RFM) — frozen LTC feature primitives.

Implements the core mechanism from arXiv 2606.15571 (Linghu & Wang 2026):
use Liquid Time-Constant (LTC) closed-form responses as *frozen* feature
primitives, then learn only a linear readout.

The closed-form LTC solution is:

    dh/dt = -[1/tau + g(x)] h + g(x) A          (LTC ODE)
    h(0) = h_0(x)
    =>  h(x, t) = h_0(x) * exp(-[1/tau + g(x)] t) + g(x) A * [1 - exp(-...)] / [...]

For each random feature i, we sample (tau_i, A_i, w_i, b_i, w0_i, b0_i) and compute:
    g_i(x)    = tanh(w_i . x + b_i)
    h0_i(x)   = tanh(w0_i . x + b0_i)
    alpha_i(x)= 1/tau_i + g_i(x)
    phi_i(x, t) = h0_i(x) * exp(-alpha_i(x) t)
              + g_i(x) * A_i * (1 - exp(-alpha_i(x) t)) / alpha_i(x)

The phi_i are *frozen* (no learning); the user adds a linear readout on top.
This implements the "frozen LTC feature primitive" pattern from L-RFM.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LiquidRandomFeatureBasis(nn.Module):
    """Frozen LTC random feature basis (L-RFM primitive).

    Samples ``n_features`` LTC features with random (tau, A, w, b, w0, b0).
    For each input x and time t, returns the closed-form LTC response
    phi(x, t) of shape ``(batch, n_features)``.

    Args:
        input_size: dimension of input x.
        n_features: number of random LTC features to sample.
        tau_min, tau_max: range for sampling tau (initial time constant).
        A_std: std for sampling A (equilibrium target).
        seed: RNG seed for reproducible feature sampling.
    """

    def __init__(
        self,
        input_size: int,
        n_features: int = 64,
        tau_min: float = 0.1,
        tau_max: float = 5.0,
        A_std: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_features = n_features
        g = torch.Generator().manual_seed(seed)
        # Sample tau uniformly in log space
        log_tau = torch.empty(n_features).uniform_(
            math.log(tau_min), math.log(tau_max), generator=g
        )
        self.tau = nn.Parameter(log_tau.exp(), requires_grad=False)
        # Sample A, w, b, w0, b0
        self.A = nn.Parameter(torch.randn(n_features, generator=g) * A_std, requires_grad=False)
        self.w = nn.Parameter(torch.randn(n_features, input_size, generator=g) * 0.5, requires_grad=False)
        self.b = nn.Parameter(torch.randn(n_features, generator=g) * 0.5, requires_grad=False)
        self.w0 = nn.Parameter(torch.randn(n_features, input_size, generator=g) * 0.5, requires_grad=False)
        self.b0 = nn.Parameter(torch.randn(n_features, generator=g) * 0.5, requires_grad=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute phi(x, t) for all frozen features.

        Args:
            x: ``(batch, input_size)`` or ``(batch, seq_len, input_size)``
            t: scalar time, ``(batch,)`` per-sample, or ``(batch, seq_len)`` per-step

        Returns:
            phi: ``(..., n_features)`` (matching x's leading dims)
        """
        squeeze_back = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_back = True
        # x: (B, T, input_size)
        # g: tanh(w . x + b) -> (B, T, n_features)
        g = torch.tanh(torch.einsum("bti,fi->btf", x, self.w) + self.b)
        h0 = torch.tanh(torch.einsum("bti,fi->btf", x, self.w0) + self.b0)
        # alpha(x): (B, T, n_features)
        inv_tau = 1.0 / self.tau  # (n_features,)
        alpha = inv_tau + g  # broadcasts
        # t: dispatch based on shape
        if t.dim() == 0:
            # scalar time: same for all (B, T)
            t_b = t.expand(x.shape[0], x.shape[1], 1)
        elif t.dim() == 1:
            t_len = t.shape[0]
            if t_len == x.shape[0]:
                # (B,) per-sample time
                t_b = t.unsqueeze(-1).unsqueeze(-1).expand(x.shape[0], x.shape[1], 1)
            elif t_len == x.shape[1]:
                # (T,) per-step time index
                t_b = t.unsqueeze(0).unsqueeze(-1).expand(x.shape[0], x.shape[1], 1)
            else:
                raise ValueError(
                    f"t.shape[0]={t_len} doesn't match x.shape[0]={x.shape[0]} "
                    f"or x.shape[1]={x.shape[1]}"
                )
        elif t.dim() == 2:
            # (B, T) per-step per-sample
            t_b = t.unsqueeze(-1)
        else:
            t_b = t
        # phi = h0 * exp(-alpha t) + g * A * (1 - exp(-alpha t)) / alpha
        exp_term = torch.exp(-alpha * t_b)
        # Stable division when alpha ≈ 0
        # Original formula: (1 - exp(-alpha t)) / alpha → t  as  alpha → 0
        fallback = t_b if t_b.shape == alpha.shape else torch.full_like(alpha, t_b.item() if t_b.numel() == 1 else 1.0)
        one_over_alpha = torch.where(
            alpha.abs() < 1e-8,
            fallback,
            (1.0 - exp_term) / alpha,
        )
        phi = h0 * exp_term + g * self.A * one_over_alpha
        if squeeze_back:
            phi = phi.squeeze(1)
        return phi  # (B, T, n_features) or (B, n_features)


class LRFMSequenceRegressor(nn.Module):
    """Sequence-to-sequence regressor using frozen LTC features + linear readout.

    Pattern: feature extraction is frozen L-RFM, only the linear readout learns.
    For sequence modeling, time t is taken as a per-step index scaled by dt.

    Args:
        input_size: dimension of input sequence at each step.
        output_size: dimension of regression target at each step.
        n_features: number of frozen LTC features (controls capacity).
        hidden_size: optional linear bottleneck between L-RFM and output.
        seed: RNG seed for L-RFM feature sampling.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_features: int = 64,
        hidden_size: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.basis = LiquidRandomFeatureBasis(
            input_size=input_size, n_features=n_features, seed=seed
        )
        if hidden_size > 0:
            self.readout = nn.Sequential(
                nn.Linear(n_features, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, output_size),
            )
        else:
            self.readout = nn.Linear(n_features, output_size)

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Run frozen L-RFM + linear readout on sequence.

        Args:
            x: ``(batch, seq_len, input_size)``
            dt: scalar time step (effective "t" passed to LTC features).
        """
        B, T, _ = x.shape
        device = x.device
        # Build t as arange(T) * dt
        t = torch.arange(T, device=device, dtype=x.dtype) * dt  # (T,)
        phi = self.basis(x, t)  # (B, T, n_features)
        return self.readout(phi)  # (B, T, output_size)


__all__ = [
    "LiquidRandomFeatureBasis",
    "LRFMSequenceRegressor",
]
