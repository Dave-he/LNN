"""Input-to-State Stable (ISS) CfC (arXiv:2606.14136 response, round 242).

Reference: arXiv:2606.14136 "Environment-Aware Stable Neural Koopman Dynamics
Learning for Input-Driven Systems under Environmental Constraints" (June
2026). The paper integrates Koopman operator lifting with stability and
**Input-to-State Stability (ISS)**. The discrete-time ISS condition is::

    V(h_{t+1}) - V(h_t) <= -alpha * V(h_t) + beta * ||x_t||^2

which rearranges to::

    V(h_{t+1}) <= (1 - alpha) * V(h_t) + beta * ||x_t||^2

* When ``x_t = 0`` this reduces to V(h_{t+1}) <= (1 - alpha) * V(h_t)
  (round-240 Lyapunov contraction).
* When ``||x_t||`` is large, V can grow but is bounded by ``beta * ||x_t||^2``
  (round-241 input sensitivity guarantee).

This module ships ``ISSStableCfCCell`` — a CfC wrapper with an
**ISS Lyapunov function V(h) = h^T P h** and the corresponding ISS loss.

Key difference vs round 240 (``LyapunovStableCfCCell``) and round 241
(``ControllabilityCfCCell``): ISS unifies both into a single quadratic bound
that explicitly trades contraction against input drive. The hope (H1 of the
digest) is that ISS is more robust on noisy data than either mechanism
alone, because the ``beta * ||x||^2`` term allows the model to attend to
inputs without exploding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell
from lnn.core.lyapunov_stable_cfc import (
    lyapunov_value,
    make_lyapunov_matrix,
    positive_definite_loss,
)


def iss_decay_loss(
    h: torch.Tensor,
    h_next: torch.Tensor,
    x_t: torch.Tensor,
    P: torch.Tensor,
    alpha: float = 0.05,
    beta: float = 0.01,
    margin: float = 0.0,
) -> torch.Tensor:
    """ISS Lyapunov loss.

    Returns ``mean( relu( V(h_next) - (1 - alpha) * V(h) - beta * ||x_t||^2 + margin ) )``,
    which is 0 whenever the ISS contraction

        V(h_next) <= (1 - alpha) * V(h) + beta * ||x_t||^2 - margin

    holds for every sample in the batch. ``beta * ||x||^2`` is the
    *upper-bound allowance* — a large input is permitted to grow V by up
    to ``beta * ||x||^2`` without the loss firing.
    """
    V_t = lyapunov_value(h, P)
    V_next = lyapunov_value(h_next, P)
    x_norm_sq = (x_t * x_t).sum(dim=-1)
    return torch.clamp(V_next - (1.0 - alpha) * V_t - beta * x_norm_sq + margin,
                        min=0.0).mean()


def input_bound_ratio(
    h: torch.Tensor,
    h_next: torch.Tensor,
    x_t: torch.Tensor,
    P: torch.Tensor,
    alpha: float = 0.05,
    beta: float = 0.01,
) -> torch.Tensor:
    """Diagnostic: per-sample ratio of V growth to bound.

    Returns ``V(h_next) / ((1 - alpha) * V(h) + beta * ||x||^2 + eps)``
    averaged over the batch. A ratio < 1 means the ISS bound is satisfied;
    a ratio >> 1 means V is growing faster than the input can explain
    (catastrophic instability).
    """
    V_t = lyapunov_value(h, P)
    V_next = lyapunov_value(h_next, P)
    x_norm_sq = (x_t * x_t).sum(dim=-1)
    bound = (1.0 - alpha) * V_t + beta * x_norm_sq + 1e-6
    return (V_next / bound).mean()


class ISSStableCfCCell(nn.Module):
    """CfC cell with an ISS (Input-to-State Stability) Lyapunov certificate.

    The wrapped ``CfCCell`` performs the regular closed-form update. The
    wrapper exposes two extra losses:

    * ``compute_iss_loss(h, h_next, x_t)`` — ISS contraction loss
    * ``compute_pd_loss()`` — positive-definite penalty on P

    Args:
        input_size, hidden_size, n_tau, tau_scales: forwarded to CfCCell.
        alpha: contraction rate when x=0 (default 0.05).
        beta:  ISS gain — V is allowed to grow by ``beta * ||x||^2`` per step
               (default 0.01). Higher beta = model can attend more to inputs.
        pd_eps: minimum eigenvalue of P (default 1e-3).
        lyap_scale: initial scale of P (default 1.0).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_tau: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        alpha: float = 0.05,
        beta: float = 0.01,
        pd_eps: float = 1e-3,
        lyap_scale: float = 1.0,
    ):
        super().__init__()
        self.cell = CfCCell(input_size, hidden_size, n_tau=n_tau, tau_scales=tau_scales)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.pd_eps = float(pd_eps)
        self.lyapunov_P = make_lyapunov_matrix(hidden_size, scale=lyap_scale)

    @property
    def hidden_size(self) -> int:
        return self.cell.hidden_size

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        return self.cell(x_t, h, dt=dt)

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
        iss_lambda: float = 0.0,
        pd_lambda: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """CfC step + auxiliary ISS/PD losses.

        ``aux_dict`` contains:

        * ``"h"``, ``"h_next"``, ``"x_t"`` — inputs and outputs
        * ``"V_h"``, ``"V_next"`` — Lyapunov values
        * ``"x_norm_sq"`` — per-sample input norm squared
        * ``"iss_loss"`` — always present (raw loss)
        * ``"pd_loss"`` — always present (raw PD penalty)
        * ``"iss_loss_total"`` — only when ``iss_lambda > 0``
        * ``"pd_loss_total"`` — only when ``pd_lambda > 0``
        * ``"bound_ratio"`` — diagnostic (always present)
        """
        h_next = self.cell(x_t, h, dt=dt)
        V_h = lyapunov_value(h, self.lyapunov_P)
        V_next = lyapunov_value(h_next, self.lyapunov_P)
        x_norm_sq = (x_t * x_t).sum(dim=-1)
        loss = iss_decay_loss(h, h_next, x_t, self.lyapunov_P,
                              alpha=self.alpha, beta=self.beta)
        pd = positive_definite_loss(self.lyapunov_P, eps=self.pd_eps)
        ratio = input_bound_ratio(h, h_next, x_t, self.lyapunov_P,
                                  alpha=self.alpha, beta=self.beta)
        aux: dict[str, torch.Tensor] = {
            "h": h,
            "h_next": h_next,
            "x_t": x_t,
            "V_h": V_h,
            "V_next": V_next,
            "x_norm_sq": x_norm_sq,
            "iss_loss": loss,
            "pd_loss": pd,
            "bound_ratio": ratio,
        }
        if iss_lambda > 0:
            aux["iss_loss_total"] = iss_lambda * loss
        if pd_lambda > 0:
            aux["pd_loss_total"] = pd_lambda * pd
        return h_next, aux

    def compute_iss_loss(self, h: torch.Tensor, h_next: torch.Tensor,
                          x_t: torch.Tensor) -> torch.Tensor:
        return iss_decay_loss(h, h_next, x_t, self.lyapunov_P,
                              alpha=self.alpha, beta=self.beta)

    def compute_pd_loss(self) -> torch.Tensor:
        return positive_definite_loss(self.lyapunov_P, eps=self.pd_eps)


__all__ = [
    "iss_decay_loss",
    "input_bound_ratio",
    "ISSStableCfCCell",
]