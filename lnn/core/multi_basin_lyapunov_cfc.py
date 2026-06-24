"""Multi-Basin Lyapunov-Stable CfC (arXiv:2606.18315 response, round 244).

Reference: arXiv:2606.18315 "Ghost Attractor Networks: Basin-Structured
Dynamical Decoders for Closed-Loop Sequential Generation" (Wang et al.
June 2026). The paper proposes a hidden state that lives in a learned
potential with **multiple basin-attractors** rather than a single fixed
point. Mode transitions happen through saddle-node bifurcations and
"ghost attractor escape".

Round 240 introduced CfC with a **single-basin** Lyapunov function
``V(h) = h^T P h`` whose unique basin is the origin. Round 244 extends
this to **K learned basins** ``{c_1, ..., c_K}`` with a soft-min
distance Lyapunov::

    V(h) = -(1/beta_v) * logsumexp_k exp(-beta_v * ||h - c_k||^2)

This is a soft version of ``min_k ||h - c_k||^2`` — the model is
attracted to the *closest* basin. As ``beta_v → ∞`` the function
collapses to the hard min; as ``beta_v → 0`` it becomes the average
squared distance. With ``beta_v = 2.0`` (default) it is differentiable
and provides a smooth Lyapunov certificate.

Multi-basin contraction loss (round 240 generalization)::

    relu( V(h_next) - (1 - alpha) * V(h) + margin )

with optional ISS extension ``+ beta * ||x||^2``.

API::

    MultiBasinLyapunovStableCfCCell(input_size, hidden_size, n_basin=3,
                                   alpha=0.05, beta_v=2.0)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell


def multi_basin_distance(
    h: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    """Squared distance from ``h`` to each basin center.

    Args:
        h: Tensor of shape ``(B, H)`` or ``(..., H)``.
        centers: Tensor of shape ``(K, H)``.

    Returns:
        Tensor of shape ``(B, K)`` (or ``(..., K)``) where each entry
        ``[b, k]`` is ``||h[b] - centers[k]||^2``.
    """
    # h.unsqueeze(1): (B, 1, H); centers.unsqueeze(0): (1, K, H)
    diff = h.unsqueeze(-2) - centers.unsqueeze(-3)  # (B, K, H)
    return (diff * diff).sum(dim=-1)


def multi_basin_lyapunov_value(
    h: torch.Tensor,
    centers: torch.Tensor,
    beta_v: float = 2.0,
) -> torch.Tensor:
    """Soft-min Lyapunov value V(h) = -(1/β_v) · log Σ exp(-β_v · d_k).

    Returns one scalar per batch element. As ``β_v → ∞`` V(h) → min_k d_k,
    and as ``β_v → 0`` V(h) → mean_k d_k.
    """
    d_sq = multi_basin_distance(h, centers)  # (B, K)
    return -(1.0 / beta_v) * torch.logsumexp(-beta_v * d_sq, dim=-1)


def multi_basin_lyap_decay_loss(
    h: torch.Tensor,
    h_next: torch.Tensor,
    centers: torch.Tensor,
    alpha: float = 0.05,
    beta_v: float = 2.0,
    margin: float = 0.0,
) -> torch.Tensor:
    """Multi-basin contraction loss: relu(V_next - (1-α)V + margin).

    This generalizes round 240's ``lyapunov_decay_loss`` to K basins.
    """
    V_t = multi_basin_lyapunov_value(h, centers, beta_v)
    V_next = multi_basin_lyapunov_value(h_next, centers, beta_v)
    return torch.clamp(V_next - (1.0 - alpha) * V_t + margin,
                       min=0.0).mean()


def multi_basin_iss_decay_loss(
    h: torch.Tensor,
    h_next: torch.Tensor,
    x_t: torch.Tensor,
    centers: torch.Tensor,
    alpha: float = 0.05,
    beta_v: float = 2.0,
    beta_x: float = 0.01,
    margin: float = 0.0,
) -> torch.Tensor:
    """Multi-basin ISS: V_next ≤ (1-α)V + β_x · ||x||² + margin.

    Combines round 242's ISS (input-tolerated contraction) with round
    244's multi-basin structure.
    """
    V_t = multi_basin_lyapunov_value(h, centers, beta_v)
    V_next = multi_basin_lyapunov_value(h_next, centers, beta_v)
    x_norm_sq = (x_t * x_t).sum(dim=-1)
    return torch.clamp(V_next - (1.0 - alpha) * V_t - beta_x * x_norm_sq + margin,
                       min=0.0).mean()


def basin_assignment_entropy(
    h: torch.Tensor,
    centers: torch.Tensor,
    beta_v: float = 2.0,
) -> torch.Tensor:
    """Average per-sample Shannon entropy of the soft basin assignment.

    Returns ``mean( - sum_k p_k · log p_k )`` where ``p_k ∝ exp(-β_v·d_k)``.
    Max entropy is ``log K``. A high value means different samples are
    using multiple basins; a low value means the network always picks the
    same basin (collapse).
    """
    d_sq = multi_basin_distance(h, centers)  # (B, K)
    log_p = -beta_v * d_sq - torch.logsumexp(-beta_v * d_sq, dim=-1,
                                             keepdim=True)
    p = log_p.exp()
    eps = 1e-8
    return (-p * (p + eps).log()).sum(dim=-1).mean()


class MultiBasinLyapunovStableCfCCell(nn.Module):
    """CfC with **multi-basin** Lyapunov stability certificate.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        n_basin: Number of learned basin centers ``K``.
        alpha: Contraction rate (per-step factor ``1 - α``).
        beta_v: Soft-min temperature (higher → closer to hard min).
        pd_eps: Minimum basin-center separation (anti-collapse).
        tau_init: Initial basin-center scale.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_basin: int = 3,
        alpha: float = 0.05,
        beta_v: float = 2.0,
        pd_eps: float = 1e-2,
        tau_init: float = 0.3,
    ):
        super().__init__()
        assert n_basin >= 2, f"n_basin must be >= 2, got {n_basin}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_basin = int(n_basin)
        self.alpha = float(alpha)
        self.beta_v = float(beta_v)
        self.pd_eps = float(pd_eps)

        self.cell = CfCCell(input_size, hidden_size)

        # K basin centers in H-space, initialised as unit-norm
        # directions scaled by tau_init so they are spread out.
        centres = torch.randn(self.n_basin, hidden_size) * tau_init
        self.basin_centers = nn.Parameter(centres)

    @property
    def centers(self) -> torch.Tensor:
        return self.basin_centers

    def basin_separation_loss(self) -> torch.Tensor:
        """Encourage basins to stay spread out (penalise collapse)."""
        if self.n_basin < 2:
            return torch.tensor(0.0)
        # Pairwise distances squared.
        c = self.basin_centers  # (K, H)
        diff = c.unsqueeze(0) - c.unsqueeze(1)  # (K, K, H)
        d_sq = (diff * diff).sum(dim=-1)  # (K, K)
        # Off-diagonal min.
        K = self.n_basin
        mask = ~torch.eye(K, dtype=torch.bool, device=d_sq.device)
        off = d_sq.masked_select(mask)
        return torch.clamp(self.pd_eps - off.min(), min=0.0)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.cell(x_t, h)

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        lyap_lambda: float = 0.0,
        sep_lambda: float = 0.0,
        iss_lambda: float = 0.0,
        beta_x: float = 0.01,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """CfC step + multi-basin auxiliary diagnostics.

        ``aux_dict`` contains:

        * ``"h_next"`` — output of the step
        * ``"V_h"`` — multi-basin Lyapunov value at h
        * ``"V_next"`` — multi-basin Lyapunov value at h_next
        * ``"basin_assign"`` — (B, K) softmax over basins for h_next
        * ``"basin_entropy"`` — mean entropy (always present)
        * ``"lyap_loss"`` — multi-basin contraction loss (always present)
        * ``"lyap_loss_total"`` — only when ``lyap_lambda > 0``
        * ``"iss_loss_total"`` — only when ``iss_lambda > 0``
        * ``"sep_loss_total"`` — only when ``sep_lambda > 0``
        """
        h_next = self.cell(x_t, h)
        V_t = multi_basin_lyapunov_value(h, self.basin_centers, self.beta_v)
        V_next = multi_basin_lyapunov_value(h_next, self.basin_centers,
                                             self.beta_v)
        d_sq = multi_basin_distance(h_next, self.basin_centers)  # (B, K)
        basin_assign = torch.softmax(-self.beta_v * d_sq, dim=-1)
        basin_ent = basin_assignment_entropy(h_next, self.basin_centers,
                                             self.beta_v)

        lyap = multi_basin_lyap_decay_loss(
            h, h_next, self.basin_centers,
            alpha=self.alpha, beta_v=self.beta_v,
        )
        sep = self.basin_separation_loss()

        aux: dict[str, torch.Tensor] = {
            "h_next": h_next,
            "V_h": V_t,
            "V_next": V_next,
            "basin_assign": basin_assign,
            "basin_entropy": basin_ent,
            "lyap_loss": lyap,
            "sep_loss": sep,
        }
        if lyap_lambda > 0:
            aux["lyap_loss_total"] = lyap_lambda * lyap
        if iss_lambda > 0:
            iss = multi_basin_iss_decay_loss(
                h, h_next, x_t, self.basin_centers,
                alpha=self.alpha, beta_v=self.beta_v, beta_x=beta_x,
            )
            aux["iss_loss_total"] = iss_lambda * iss
        if sep_lambda > 0:
            aux["sep_loss_total"] = sep_lambda * sep
        return h_next, aux


__all__ = [
    "MultiBasinLyapunovStableCfCCell",
    "multi_basin_distance",
    "multi_basin_lyapunov_value",
    "multi_basin_lyap_decay_loss",
    "multi_basin_iss_decay_loss",
    "basin_assignment_entropy",
]