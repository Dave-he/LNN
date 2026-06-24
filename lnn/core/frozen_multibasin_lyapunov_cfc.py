"""Frozen Multi-Basin Lyapunov CfC (round 246 × 244 composition, round 247).

This module composes two winning mechanisms from the 240-246 audit:

  - **Round 246 (frozen random τ)** — K branches with frozen
    log-uniform time scales, learned softmax mix weights
    (BIGGEST win in audit: -65.7/-37.2/-54.7%)
  - **Round 244 (multi-basin Lyapunov)** — K' learned basin centers
    in h-space, soft-min Lyapunov contraction loss
    (strict win on toy_sin: -63.8%)

The composition hypothesis: **multi-scale temporal + multi-basin
spatial** compose constructively rather than interfere. The multi-τ
structure provides temporal coverage; the multi-basin structure
provides geometric regularization on the hidden-state manifold.

API::

    FrozenMultiBasinLyapunovCfCCell(input_size, hidden_size,
                                    n_branches=4, n_basin=3,
                                    tau_min=0.05, tau_max=20.0)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell
from lnn.core.frozen_sampled_multitau_cfc import sample_log_uniform
from lnn.core.multi_basin_lyapunov_cfc import (
    basin_assignment_entropy,
    multi_basin_lyap_decay_loss,
    multi_basin_lyapunov_value,
)


class FrozenMultiBasinLyapunovCfCCell(nn.Module):
    """CfC with **frozen multi-τ branches + multi-basin Lyapunov**.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension per branch.
        n_branches: Number of frozen-τ branches (round 246).
        n_basin: Number of learned basin centers (round 244).
        tau_min: Lower bound of log-uniform τ sampling.
        tau_max: Upper bound of log-uniform τ sampling.
        seed: RNG seed for τ sampling.
        alpha: Lyapunov contraction rate (per-step factor 1-α).
        beta_v: Soft-min temperature for multi-basin V.
        pd_eps: Minimum basin-center separation.
        learn_mix: If True, branch mix weights are learned.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_branches: int = 4,
        n_basin: int = 3,
        tau_min: float = 0.05,
        tau_max: float = 20.0,
        seed: int = 42,
        alpha: float = 0.05,
        beta_v: float = 2.0,
        pd_eps: float = 1e-2,
        learn_mix: bool = True,
    ):
        super().__init__()
        assert n_branches >= 2
        assert n_basin >= 2

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_branches = int(n_branches)
        self.n_basin = int(n_basin)
        self.alpha = float(alpha)
        self.beta_v = float(beta_v)
        self.pd_eps = float(pd_eps)

        # Round 246: K frozen random τ branches.
        taus = sample_log_uniform(self.n_branches, tau_min, tau_max, seed=seed)
        self.register_buffer("tau_frozen", taus)
        self.cells = nn.ModuleList([
            CfCCell(input_size, hidden_size) for _ in range(self.n_branches)
        ])
        for cell, tau in zip(self.cells, taus):
            with torch.no_grad():
                cell.time_scale.fill_(float(tau))

        self.learn_mix = bool(learn_mix)
        if learn_mix:
            self.mix_param = nn.Parameter(torch.zeros(self.n_branches))
        else:
            self.register_buffer(
                "_mix_const", torch.ones(self.n_branches) / self.n_branches,
            )

        # Round 244: K' learned basin centers in h-space.
        centres = torch.randn(self.n_basin, hidden_size) * 0.3
        self.basin_centers = nn.Parameter(centres)

    @property
    def tau_values(self) -> torch.Tensor:
        return self.tau_frozen

    @property
    def alpha_mix(self) -> torch.Tensor:
        if self.learn_mix:
            return torch.softmax(self.mix_param, dim=0)
        return self._mix_const

    def basin_separation_loss(self) -> torch.Tensor:
        if self.n_basin < 2:
            return torch.tensor(0.0)
        c = self.basin_centers
        diff = c.unsqueeze(0) - c.unsqueeze(1)
        d_sq = (diff * diff).sum(dim=-1)
        K = self.n_basin
        mask = ~torch.eye(K, dtype=torch.bool, device=d_sq.device)
        off = d_sq.masked_select(mask)
        return torch.clamp(self.pd_eps - off.min(), min=0.0)

    def init_state(self, batch_size: int, device: torch.device | None = None
                    ) -> list[torch.Tensor]:
        d = device or torch.device("cpu")
        return [torch.zeros(batch_size, self.hidden_size, device=d)
                for _ in range(self.n_branches)]

    def forward(self, x_t: torch.Tensor, h_list: list[torch.Tensor]
                 ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        outs = [cell(x_t, h_k) for cell, h_k in zip(self.cells, h_list)]
        h_next = sum(a * o for a, o in zip(self.alpha_mix, outs))
        return h_next, outs

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h_list: list[torch.Tensor],
        lyap_lambda: float = 0.0,
        sep_lambda: float = 0.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        """Step + diagnostics.

        ``aux_dict`` contains:

        * ``"h_next"`` — fused state
        * ``"alpha_mix"`` — branch mix weights
        * ``"V_h"`` — multi-basin Lyapunov value at h (using h_0 as proxy)
        * ``"V_next"`` — multi-basin Lyapunov value at h_next
        * ``"basin_assign"`` — softmax over basins for h_next
        * ``"basin_entropy"`` — mean entropy (always present)
        * ``"lyap_loss"`` — multi-basin contraction loss (always present)
        * ``"lyap_loss_total"`` — only when ``lyap_lambda > 0``
        * ``"sep_loss_total"`` — only when ``sep_lambda > 0``
        """
        h_next, outs = self.forward(x_t, h_list)
        # Use the first branch's input state as the "previous h" for Lyapunov.
        h_prev = h_list[0]
        V_t = multi_basin_lyapunov_value(
            h_prev, self.basin_centers, self.beta_v,
        )
        V_next = multi_basin_lyapunov_value(
            h_next, self.basin_centers, self.beta_v,
        )
        basin_ent = basin_assignment_entropy(
            h_next, self.basin_centers, self.beta_v,
        )
        # basin_assign: (B, n_basin)
        d_sq = (h_next.unsqueeze(-2) - self.basin_centers.unsqueeze(-3)
                ).pow(2).sum(-1)
        basin_assign = torch.softmax(-self.beta_v * d_sq, dim=-1)

        lyap = multi_basin_lyap_decay_loss(
            h_prev, h_next, self.basin_centers,
            alpha=self.alpha, beta_v=self.beta_v,
        )
        sep = self.basin_separation_loss()

        aux: dict[str, torch.Tensor] = {
            "h_next": h_next,
            "alpha_mix": self.alpha_mix.detach(),
            "V_h": V_t,
            "V_next": V_next,
            "basin_assign": basin_assign,
            "basin_entropy": basin_ent,
            "lyap_loss": lyap,
            "sep_loss": sep,
        }
        if lyap_lambda > 0:
            aux["lyap_loss_total"] = lyap_lambda * lyap
        if sep_lambda > 0:
            aux["sep_loss_total"] = sep_lambda * sep
        return h_next, outs, aux


__all__ = ["FrozenMultiBasinLyapunovCfCCell"]