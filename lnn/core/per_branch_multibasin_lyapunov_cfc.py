"""Per-Branch Multi-Basin Lyapunov CfC (round 248).

Composition of round 246 (frozen random τ branches) + round 244
(multi-basin Lyapunov) lifted to **per-branch** geometry: each
τ-branch has its own set of learned basin centers in h-space, with
its own multi-basin Lyapunov value and contraction loss.

Key idea: in round 247 the multi-basin structure was global (single
set of basin centers shared across branches). That is a *summative*
composition (K + K' structure). Per-branch basins are *multiplicative*
(K × K' effective centers), allowing each timescale to specialize in
its own geometric manifold.

API::

    PerBranchMultiBasinLyapunovCfCCell(input_size, hidden_size,
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


class PerBranchMultiBasinLyapunovCfCCell(nn.Module):
    """CfC with frozen multi-τ branches **and per-branch multi-basin Lyapunov**.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension per branch.
        n_branches: Number of frozen-τ branches (round 246).
        n_basin: Number of learned basin centers **per branch**.
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

        # Per-branch basin centers: (n_branches, n_basin, hidden_size).
        centres = torch.randn(self.n_branches, self.n_basin, hidden_size) * 0.3
        self.basin_centers = nn.Parameter(centres)

    @property
    def tau_values(self) -> torch.Tensor:
        return self.tau_frozen

    @property
    def alpha_mix(self) -> torch.Tensor:
        if self.learn_mix:
            return torch.softmax(self.mix_param, dim=0)
        return self._mix_const

    def per_branch_separation_loss(self) -> torch.Tensor:
        """Sum of separation losses across branches."""
        if self.n_basin < 2:
            return torch.tensor(0.0)
        total = torch.tensor(0.0, device=self.basin_centers.device)
        for k in range(self.n_branches):
            c = self.basin_centers[k]  # (n_basin, hidden_size)
            diff = c.unsqueeze(0) - c.unsqueeze(1)
            d_sq = (diff * diff).sum(dim=-1)
            K = self.n_basin
            mask = ~torch.eye(K, dtype=torch.bool, device=d_sq.device)
            off = d_sq.masked_select(mask)
            total = total + torch.clamp(self.pd_eps - off.min(), min=0.0)
        return total

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

    def per_branch_lyapunov_value(self, h_k: torch.Tensor, k: int
                                   ) -> torch.Tensor:
        """Multi-basin V for branch k."""
        return multi_basin_lyapunov_value(
            h_k, self.basin_centers[k], self.beta_v,
        )

    def per_branch_basin_entropy(self, h_k: torch.Tensor, k: int
                                  ) -> torch.Tensor:
        return basin_assignment_entropy(
            h_k, self.basin_centers[k], self.beta_v,
        )

    def per_branch_lyap_decay(self, h_prev_k: torch.Tensor, h_next_k: torch.Tensor,
                              k: int) -> torch.Tensor:
        return multi_basin_lyap_decay_loss(
            h_prev_k, h_next_k, self.basin_centers[k],
            alpha=self.alpha, beta_v=self.beta_v,
        )

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
        * ``"per_branch_V_next"`` — (n_branches,) V at h_next per branch
        * ``"per_branch_basin_H"`` — (n_branches,) entropy per branch
        * ``"mean_basin_H"`` — scalar mean over branches
        * ``"lyap_loss"`` — sum of per-branch contraction losses
        * ``"lyap_loss_total"`` — only when ``lyap_lambda > 0``
        * ``"sep_loss_total"`` — only when ``sep_lambda > 0``
        """
        h_next, outs = self.forward(x_t, h_list)

        V_per_branch = []
        H_per_branch = []
        lyap_total = torch.tensor(0.0, device=x_t.device)
        for k, (h_prev_k, h_next_k) in enumerate(zip(h_list, outs)):
            # Reduce across batch to (1,) per branch for stacking.
            V_per_branch.append(self.per_branch_lyapunov_value(h_next_k, k).mean())
            H_per_branch.append(self.per_branch_basin_entropy(h_next_k, k).mean())
            lyap_total = lyap_total + self.per_branch_lyap_decay(
                h_prev_k, h_next_k, k,
            )
        V_per_branch_t = torch.stack(V_per_branch)
        H_per_branch_t = torch.stack(H_per_branch)

        sep = self.per_branch_separation_loss()

        aux: dict[str, torch.Tensor] = {
            "h_next": h_next,
            "alpha_mix": self.alpha_mix.detach(),
            "per_branch_V_next": V_per_branch_t,
            "per_branch_basin_H": H_per_branch_t,
            "mean_basin_H": H_per_branch_t.mean(),
            "lyap_loss": lyap_total,
            "sep_loss": sep,
        }
        if lyap_lambda > 0:
            aux["lyap_loss_total"] = lyap_lambda * lyap_total
        if sep_lambda > 0:
            aux["sep_loss_total"] = sep_lambda * sep
        return h_next, outs, aux


__all__ = ["PerBranchMultiBasinLyapunovCfCCell"]