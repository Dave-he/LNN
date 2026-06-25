"""AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell (round 253).

Closes the 7-round arc (r246-r252) with **adaptive per-branch aux weight**:

  * r252 (LyapAuxPerBranch): aux weight is **constant** across branches
    → strict-win on toy_sin/random but **+34.5% regression on structured**
    (contraction kills natural periodic dynamics on smooth/periodic data).

  * r253 (this): aux weight is **proportional to per-branch basin
    assignment entropy H_k**. When a branch is CONFIDENT (low H_k —
    has identified a single dominant basin, periodic dynamics on a
    single attractor), the aux weight collapses to 0 and the branch
    flows naturally. When a branch is UNCERTAIN (high H_k — basin
    boundary region, mixed dynamics), the aux weight is high to
    enforce contraction toward the closest basin.

Mechanism:
  ``lambda_k = lambda_max * (H_k / log(n_basin))``
  ``lyap_loss_adaptive = sum_k lambda_k * lyap_loss_k``

This is a **content-aware** contraction prior: contraction only
fires on the branches that need it, leaving confident branches free
to model the natural periodic dynamics of structured data.

Hypothesis (PRD #10-90, round 253):
  H1: Adaptive aux matches r252 toy_sin/random (within ±5%)
  H2: Adaptive aux IMPROVES r252 on structured (closing the +34.5%
      gap with r249 input_geom_gated)
  H3: mean aux weight on structured < mean aux weight on toy_sin
      (structured branches are confident → low λ)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from lnn.core.per_branch_multibasin_lyapunov_cfc import (
    PerBranchMultiBasinLyapunovCfCCell,
)


class AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell(
    PerBranchMultiBasinLyapunovCfCCell
):
    """PerBranch + adaptive per-branch aux supervision (round 253).

    Args:
        lyap_lambda_max: Upper bound on aux weight per branch.
        adaptive_aux: If True, use H-scaled per-branch λ. If False, fall
            back to constant λ = ``lyap_lambda_max`` (matches r252).
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
        lyap_lambda_max: float = 0.1,
        adaptive_aux: bool = True,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            n_branches=n_branches,
            n_basin=n_basin,
            tau_min=tau_min,
            tau_max=tau_max,
            seed=seed,
            alpha=alpha,
            beta_v=beta_v,
            pd_eps=pd_eps,
            learn_mix=True,
        )
        self.default_lyap_lambda_max = float(lyap_lambda_max)
        self.adaptive_aux = bool(adaptive_aux)

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h_list: list[torch.Tensor],
        lyap_lambda: float | None = None,
        sep_lambda: float = 0.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        if lyap_lambda is None:
            lyap_lambda = self.default_lyap_lambda_max

        h_next, outs = self.forward(x_t, h_list)

        V_per_branch = []
        H_per_branch = []
        lyap_per_branch = []
        for k, (h_prev_k, h_next_k) in enumerate(zip(h_list, outs)):
            V_per_branch.append(self.per_branch_lyapunov_value(h_next_k, k).mean())
            H_per_branch.append(self.per_branch_basin_entropy(h_next_k, k).mean())
            lyap_per_branch.append(self.per_branch_lyap_decay(
                h_prev_k, h_next_k, k,
            ))
        V_per_branch_t = torch.stack(V_per_branch)
        H_per_branch_t = torch.stack(H_per_branch)
        lyap_per_branch_t = torch.stack(lyap_per_branch)

        sep = self.per_branch_separation_loss()

        # Adaptive per-branch aux weight: λ_k = λ_max · H_k / log(n_basin).
        # Confident branch (H_k → 0): λ_k → 0 (let natural dynamics flow).
        # Uncertain branch (H_k → log n_basin): λ_k → λ_max (contract).
        log_nb = math.log(float(self.n_basin))
        if self.adaptive_aux and log_nb > 0:
            lambda_per_branch = (
                float(lyap_lambda) * H_per_branch_t.detach() / log_nb
            )
        else:
            lambda_per_branch = torch.full_like(
                H_per_branch_t, float(lyap_lambda),
            )

        # Per-branch aux-weighted sum.
        lyap_adaptive = (lambda_per_branch * lyap_per_branch_t).sum()
        # Also expose a constant-λ version for ablation comparison.
        lyap_constant = float(lyap_lambda) * lyap_per_branch_t.sum()

        aux: dict[str, torch.Tensor] = {
            "h_next": h_next,
            "alpha_mix": self.alpha_mix.detach(),
            "per_branch_V_next": V_per_branch_t,
            "per_branch_basin_H": H_per_branch_t,
            "mean_basin_H": H_per_branch_t.mean(),
            "lyap_loss": lyap_adaptive,
            "lyap_loss_constant": lyap_constant,
            "lyap_per_branch": lyap_per_branch_t,
            "lambda_per_branch": lambda_per_branch,
            "mean_lambda": lambda_per_branch.mean(),
            "sep_loss": sep,
        }
        if lyap_lambda > 0:
            aux["lyap_loss_total"] = lyap_adaptive
        if sep_lambda > 0:
            aux["sep_loss_total"] = sep_lambda * sep
        return h_next, outs, aux


__all__ = ["AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell"]
