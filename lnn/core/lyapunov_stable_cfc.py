"""Lyapunov-Stable CfC (arXiv:2606.19109 response).

Round 240 introduces a jointly-learned Lyapunov function V(h) = h^T P h
that certifies exponential stability of CfC's discrete update through a
**discrete-time Lyapunov condition**:

    V(h_{t+1}) - V(h_t) <= -alpha * V(h_t)

This module ships three primitives:

* ``lyapunov_value``       — V(h) = h^T P h  given a PSD matrix P
* ``lyapunov_decay_loss``  — relu( V(h_next) - (1 - alpha) * V(h) + margin )
* ``positive_definite_loss`` — relu( -lambda_min(P) + eps )

and one drop-in wrapper:

* ``LyapunovStableCfCCell`` — wraps a ``CfCCell`` so that ``forward_with_aux``
  returns ``(h_next, aux_dict)`` with the auxiliary losses ready to be added
  to the task loss.

Reference: arXiv:2606.19109 "Locally Stable Neural ODEs with Characterized
Region of Attraction" (June 2026). The paper proves universal approximation
of locally exponentially stable dynamics under the gradient-field constraint;
this implementation ships the **discrete analogue** suitable for the
closed-form CfC step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell


def lyapunov_value(h: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Compute V(h) = h^T P h for a PSD matrix P.

    Args:
        h: Tensor of shape ``(..., d)``.
        P: PSD matrix of shape ``(d, d)`` (will be symmetrized internally).

    Returns:
        Tensor of shape ``(...)`` (one scalar per batch element).
    """
    d = h.shape[-1]
    # Symmetrize so that round-off does not break PSD assumptions.
    Psym = 0.5 * (P + P.transpose(-1, -2))
    # V = sum_{i,j} h_i P_{ij} h_j  -> use einsum for clarity
    return torch.einsum("...i,ij,...j->...", h, Psym, h)


def lyapunov_decay_loss(
    h: torch.Tensor,
    h_next: torch.Tensor,
    P: torch.Tensor,
    alpha: float = 0.05,
    margin: float = 0.0,
) -> torch.Tensor:
    """Discrete Lyapunov contraction loss.

    Returns ``mean( relu( V(h_next) - (1 - alpha) * V(h) + margin ) )``,
    which is 0 whenever the contraction condition
    ``V(h_next) <= (1 - alpha) * V(h) - margin`` is satisfied for every
    sample in the batch.

    Args:
        h: hidden state at step ``t``, shape ``(B, d)``.
        h_next: hidden state at step ``t+1``, shape ``(B, d)``.
        P: PSD matrix of shape ``(d, d)``.
        alpha: contraction rate (default 0.05, i.e. 5% per step).
        margin: extra safety margin.
    """
    V_t = lyapunov_value(h, P)
    V_next = lyapunov_value(h_next, P)
    return torch.clamp(V_next - (1.0 - alpha) * V_t + margin, min=0.0).mean()


def positive_definite_loss(P: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Penalty that keeps ``P`` positive definite (lambda_min >= eps).

    Uses ``relu(-lambda_min(P) + eps)`` where ``lambda_min`` is the smallest
    eigenvalue estimated via ``torch.linalg.eigvalsh`` (symmetric Hermitian).
    """
    Psym = 0.5 * (P + P.transpose(-1, -2))
    eigvals = torch.linalg.eigvalsh(Psym)
    lambda_min = eigvals.min()
    return torch.clamp(-lambda_min + eps, min=0.0)


def make_lyapunov_matrix(d: int, scale: float = 1.0) -> nn.Parameter:
    """Initialise a learnable Lyapunov matrix as ``scale * I``.

    Returned as an unconstrained ``nn.Parameter`` so it can drift off the
    diagonal during training; the ``positive_definite_loss`` keeps it PSD.
    """
    return nn.Parameter(scale * torch.eye(d))


class LyapunovStableCfCCell(nn.Module):
    """CfC cell with a jointly-learned Lyapunov stability certificate.

    The wrapped ``CfCCell`` performs the regular closed-form update
    ``h_{t+1} = decay * g + (1 - decay) * h_branch``. The wrapper exposes
    two extra losses so the user can add them to the task loss:

    * ``compute_decay_loss(h, h_next)`` — discrete Lyapunov contraction
    * ``compute_pd_loss()``             — positive-definite penalty on P

    Args:
        input_size, hidden_size, n_tau, tau_scales: forwarded to CfCCell.
        alpha: contraction rate (default 0.05).
        pd_eps: minimum eigenvalue of P (default 1e-3).
        lyap_scale: initial scale of the Lyapunov matrix (default 1.0).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_tau: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        alpha: float = 0.05,
        pd_eps: float = 1e-3,
        lyap_scale: float = 1.0,
    ):
        super().__init__()
        self.cell = CfCCell(input_size, hidden_size, n_tau=n_tau, tau_scales=tau_scales)
        self.alpha = float(alpha)
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
        """Plain CfC step (no auxiliary losses)."""
        return self.cell(x_t, h, dt=dt)

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
        lyap_lambda: float = 0.0,
        pd_lambda: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """CfC step plus auxiliary Lyapunov/PD losses.

        Returns ``(h_next, aux_dict)``. ``aux_dict`` contains:

        * ``"h"``, ``"h_next"`` — the states used for the decay loss.
        * ``"V_h"``, ``"V_next"`` — Lyapunov values.
        * ``"lyap_decay_loss"`` — contraction loss (always present).
        * ``"pd_loss"`` — positive-definite penalty (always present).
        * ``"lyap_loss_total"`` — ``lyap_lambda * lyap_decay_loss``
          (only present if ``lyap_lambda > 0``).
        * ``"pd_loss_total"`` — ``pd_lambda * pd_loss``
          (only present if ``pd_lambda > 0``).
        """
        h_next = self.cell(x_t, h, dt=dt)
        V_h = lyapunov_value(h, self.lyapunov_P)
        V_next = lyapunov_value(h_next, self.lyapunov_P)
        decay_loss = lyapunov_decay_loss(h, h_next, self.lyapunov_P, alpha=self.alpha)
        pd_loss = positive_definite_loss(self.lyapunov_P, eps=self.pd_eps)
        aux: dict[str, torch.Tensor] = {
            "h": h,
            "h_next": h_next,
            "V_h": V_h,
            "V_next": V_next,
            "lyap_decay_loss": decay_loss,
            "pd_loss": pd_loss,
        }
        if lyap_lambda > 0:
            aux["lyap_loss_total"] = lyap_lambda * decay_loss
        if pd_lambda > 0:
            aux["pd_loss_total"] = pd_lambda * pd_loss
        return h_next, aux

    def compute_decay_loss(self, h: torch.Tensor, h_next: torch.Tensor) -> torch.Tensor:
        """Public helper: discrete Lyapunov contraction loss."""
        return lyapunov_decay_loss(h, h_next, self.lyapunov_P, alpha=self.alpha)

    def compute_pd_loss(self) -> torch.Tensor:
        """Public helper: positive-definite penalty on P."""
        return positive_definite_loss(self.lyapunov_P, eps=self.pd_eps)


__all__ = [
    "lyapunov_value",
    "lyapunov_decay_loss",
    "positive_definite_loss",
    "make_lyapunov_matrix",
    "LyapunovStableCfCCell",
]