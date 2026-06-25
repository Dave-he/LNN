"""Frozen Random Basin CfC (round 250).

Extends the L-RFM "frozen random basis" pattern from round 246 (frozen
random τ) to the geometric level: basin centers are **frozen at init**
(random), not learned. The only learnable structural parameter is the
per-branch mix. This tests the hypothesis:

  Frozen random τ + frozen random basins + learned mix ≈ round 248?

If true, the geometric structure is just a basis that doesn't need to
be learned — the linear output layer and the per-branch mix carry all
the adaptive capacity. This would be the "deepest" form of the L-RFM
insight: **all randomness in the structural basis, all learning in
the output projection**.

API::

    FrozenRandomBasinCfCCell(input_size, hidden_size,
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


class FrozenRandomBasinCfCCell(nn.Module):
    """CfC with **frozen random τ branches AND frozen random basins**.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension per branch.
        n_branches: Number of frozen-τ branches (round 246).
        n_basin: Number of FROZEN random basin centers per branch.
        tau_min: Lower bound of log-uniform τ sampling.
        tau_max: Upper bound of log-uniform τ sampling.
        basin_seed: RNG seed for basin center sampling.
        tau_seed: RNG seed for τ sampling.
        alpha: Lyapunov contraction rate (per-step factor 1-α).
        beta_v: Soft-min temperature for multi-basin V.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_branches: int = 4,
        n_basin: int = 3,
        tau_min: float = 0.05,
        tau_max: float = 20.0,
        basin_seed: int = 137,
        tau_seed: int = 42,
        alpha: float = 0.05,
        beta_v: float = 2.0,
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

        # Round 246: K frozen random τ branches.
        taus = sample_log_uniform(self.n_branches, tau_min, tau_max,
                                   seed=tau_seed)
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

        # Frozen random basin centers (per-branch).
        g = torch.Generator().manual_seed(basin_seed)
        centres = torch.randn(
            self.n_branches, self.n_basin, hidden_size, generator=g,
        ) * 0.3
        self.register_buffer("basin_centers", centres)

    @property
    def tau_values(self) -> torch.Tensor:
        return self.tau_frozen

    @property
    def alpha_mix(self) -> torch.Tensor:
        if self.learn_mix:
            return torch.softmax(self.mix_param, dim=0)
        return self._mix_const

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
        h_next, outs = self.forward(x_t, h_list)

        V_per_branch = []
        H_per_branch = []
        lyap_total = torch.tensor(0.0, device=x_t.device)
        for k, (h_prev_k, h_next_k) in enumerate(zip(h_list, outs)):
            V_per_branch.append(multi_basin_lyapunov_value(
                h_next_k, self.basin_centers[k], self.beta_v,
            ).mean())
            H_per_branch.append(basin_assignment_entropy(
                h_next_k, self.basin_centers[k], self.beta_v,
            ).mean())
            lyap_total = lyap_total + multi_basin_lyap_decay_loss(
                h_prev_k, h_next_k, self.basin_centers[k],
                alpha=self.alpha, beta_v=self.beta_v,
            )
        V_per_branch_t = torch.stack(V_per_branch)
        H_per_branch_t = torch.stack(H_per_branch)

        aux: dict[str, torch.Tensor] = {
            "h_next": h_next,
            "alpha_mix": self.alpha_mix.detach(),
            "per_branch_V_next": V_per_branch_t.detach(),
            "per_branch_basin_H": H_per_branch_t.detach(),
            "mean_basin_H": H_per_branch_t.mean().detach(),
            "lyap_loss": lyap_total,
        }
        if lyap_lambda > 0:
            aux["lyap_loss_total"] = lyap_lambda * lyap_total
        return h_next, outs, aux


__all__ = ["FrozenRandomBasinCfCCell"]