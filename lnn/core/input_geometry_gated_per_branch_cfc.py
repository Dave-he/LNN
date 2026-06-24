"""Input + Geometry-Conditioned Per-Branch Multi-Basin Lyapunov CfC (round 249).

Lifts the per-branch multi-basin architecture (round 248) with a
**gating signal** that conditions on both the current input AND the
per-branch Lyapunov value. This couples routing with geometric
contraction evidence — branches whose basins are most "in-contract"
for the current state get higher weight.

Mechanism::

    V_k    = multi_basin_lyapunov_value(h_k, basin_centers[k], beta_v)
    logits = W_gate · [x_t; V_k for k=1..K]   # (K,)
    alpha  = softmax(logits)
    h_next = sum_k alpha_k * h_k_out

Compare to:
  - round 248: alpha = softmax(mix_param) — input-blind constant mix
  - round 243: alpha = softmax(W_g · x_t) — input-only mix (negative)
  - round 249: alpha = softmax(W_g · [x_t, V_1, ..., V_K]) — input + geometry

API::

    InputGeometryGatedPerBranchCfCCell(input_size, hidden_size,
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


class InputGeometryGatedPerBranchCfCCell(nn.Module):
    """CfC with frozen multi-τ branches, per-branch multi-basin Lyapunov,
    AND input + geometry-conditioned gating.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension per branch.
        n_branches: Number of frozen-τ branches.
        n_basin: Number of learned basin centers **per branch**.
        tau_min: Lower bound of log-uniform τ sampling.
        tau_max: Upper bound of log-uniform τ sampling.
        seed: RNG seed for τ sampling.
        alpha: Lyapunov contraction rate (per-step factor 1-α).
        beta_v: Soft-min temperature for multi-basin V.
        pd_eps: Minimum basin-center separation.
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

        # Per-branch basin centers: (n_branches, n_basin, hidden_size).
        centres = torch.randn(self.n_branches, self.n_basin, hidden_size) * 0.3
        self.basin_centers = nn.Parameter(centres)

        # Gate network: input + per-branch V → logits over branches.
        # Init small so initial mix ≈ uniform (matches r248 starting point).
        self.gate = nn.Linear(input_size + n_branches, n_branches)
        nn.init.normal_(self.gate.weight, std=0.05)
        nn.init.zeros_(self.gate.bias)

        # Constant baseline mix (used for diagnostics, not in forward).
        self.register_buffer(
            "_mix_const", torch.ones(n_branches) / n_branches,
        )

    @property
    def tau_values(self) -> torch.Tensor:
        return self.tau_frozen

    def per_branch_separation_loss(self) -> torch.Tensor:
        if self.n_basin < 2:
            return torch.tensor(0.0)
        total = torch.tensor(0.0, device=self.basin_centers.device)
        for k in range(self.n_branches):
            c = self.basin_centers[k]
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

    def _compute_V_per_branch(self, h_list: list[torch.Tensor]
                                ) -> torch.Tensor:
        """Per-branch multi-basin V, batch-mean reduced → (n_branches,)."""
        Vs = []
        for k, h_k in enumerate(h_list):
            V_k = multi_basin_lyapunov_value(
                h_k, self.basin_centers[k], self.beta_v,
            )
            Vs.append(V_k.mean(dim=0))  # (1,)
        return torch.stack(Vs)

    def forward(self, x_t: torch.Tensor, h_list: list[torch.Tensor]
                 ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        outs = [cell(x_t, h_k) for cell, h_k in zip(self.cells, h_list)]
        V_per_branch = self._compute_V_per_branch(h_list)
        # Gate input: concat raw input + per-branch V.
        gate_in = torch.cat([x_t, V_per_branch.unsqueeze(0).expand(
            x_t.shape[0], -1)], dim=-1)
        alpha = torch.softmax(self.gate(gate_in), dim=-1)  # (B, K)
        # Stack outs to (B, K, d_h) and contract with alpha (B, K).
        outs_stack = torch.stack(outs, dim=1)  # (B, K, d_h)
        h_next = (alpha.unsqueeze(-1) * outs_stack).sum(dim=1)  # (B, d_h)
        # alpha here is per-batch; aggregate to (K,) for aux.
        alpha_mean = alpha.mean(dim=0)
        return h_next, outs, alpha_mean

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h_list: list[torch.Tensor],
        lyap_lambda: float = 0.0,
        sep_lambda: float = 0.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        h_next, outs, alpha_mean = self.forward(x_t, h_list)

        V_per_branch_next = self._compute_V_per_branch(outs)
        H_per_branch = []
        lyap_total = torch.tensor(0.0, device=x_t.device)
        for k, (h_prev_k, h_next_k) in enumerate(zip(h_list, outs)):
            H_per_branch.append(basin_assignment_entropy(
                h_next_k, self.basin_centers[k], self.beta_v,
            ).mean())
            lyap_total = lyap_total + multi_basin_lyap_decay_loss(
                h_prev_k, h_next_k, self.basin_centers[k],
                alpha=self.alpha, beta_v=self.beta_v,
            )
        H_per_branch_t = torch.stack(H_per_branch)

        sep = self.per_branch_separation_loss()

        aux: dict[str, torch.Tensor] = {
            "h_next": h_next,
            "alpha_mix": alpha_mean.detach(),
            "per_branch_V_next": V_per_branch_next.detach(),
            "per_branch_basin_H": H_per_branch_t.detach(),
            "mean_basin_H": H_per_branch_t.mean().detach(),
            "lyap_loss": lyap_total,
            "sep_loss": sep,
        }
        if lyap_lambda > 0:
            aux["lyap_loss_total"] = lyap_lambda * lyap_total
        if sep_lambda > 0:
            aux["sep_loss_total"] = sep_lambda * sep
        return h_next, outs, aux


__all__ = ["InputGeometryGatedPerBranchCfCCell"]