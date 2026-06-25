"""PerStepAdaptiveAuxMultiBasinLyapunovCfCCell (round 254).

Closes the temporal axis of the per-branch basin mechanism:

  * r253 (AdaptiveAuxPerBranch): per-BRANCH aux weight
    ``lambda_k = lambda_max * H_k / log(n_basin)`` — gates aux on
    WHICH branch is uncertain.

  * r254 (this): per-STEP aux weight
    ``lambda_t = lambda_max * (1/K) * sum_k H_k_t / log(n_basin)``
    — gates aux on WHEN the network is in a basin transition.

When the network is in a STABLE state (low mean H across branches),
lambda_t -> 0 and the network flows naturally. When in a TRANSITION
(high mean H, basin assignment is uncertain), lambda_t -> lambda_max
and contraction fires.

Combines with r253: a step aux is only "expensive" when the mean
H is high (basin-boundary regions), regardless of which branch is
uncertain. The two are complementary axes (which branch vs when).

Hypothesis (PRD #10-91, round 254):
  H1: per_step matches r253 on toy_sin/random (within ±5%) — extra
      gating does not hurt.
  H2: per_step IMPROVES r253 on structured (closes the gap with r249).
  H3: mean lambda_t is low on toy_sin (stable) and high on random
      (transitions dominate).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from lnn.core.per_branch_multibasin_lyapunov_cfc import (
    PerBranchMultiBasinLyapunovCfCCell,
)


class PerStepAdaptiveAuxMultiBasinLyapunovCfCCell(
    PerBranchMultiBasinLyapunovCfCCell
):
    """PerBranch + per-step adaptive aux supervision (round 254).

    Args:
        lyap_lambda_max: Upper bound on aux weight per step.
        per_step_aux: If True, use mean H across branches per step.
            If False, fall back to per-branch (r253) weighting.
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
        per_step_aux: bool = True,
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
        self.per_step_aux = bool(per_step_aux)

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

        # Per-step aux weight: scalar λ_t = λ_max · mean_k H_k / log(n_basin).
        # STABLE step (low mean H): λ_t → 0.
        # TRANSITION step (high mean H): λ_t → λ_max.
        log_nb = math.log(float(self.n_basin))
        if self.per_step_aux and log_nb > 0:
            mean_H = H_per_branch_t.mean()
            # Detach: gating signal, not learnable.
            lambda_step = (float(lyap_lambda) * mean_H.detach() / log_nb)
        else:
            lambda_step = torch.tensor(float(lyap_lambda), device=x_t.device)

        # Aux loss: λ_t × sum_k lyap_loss_k.
        lyap_step = lambda_step * lyap_per_branch_t.sum()
        # Per-branch version for ablation.
        if log_nb > 0:
            lambda_per_branch = (
                float(lyap_lambda) * H_per_branch_t.detach() / log_nb
            )
        else:
            lambda_per_branch = torch.full_like(H_per_branch_t, float(lyap_lambda))
        lyap_per_branch = (lambda_per_branch * lyap_per_branch_t).sum()

        aux: dict[str, torch.Tensor] = {
            "h_next": h_next,
            "alpha_mix": self.alpha_mix.detach(),
            "per_branch_V_next": V_per_branch_t,
            "per_branch_basin_H": H_per_branch_t,
            "mean_basin_H": H_per_branch_t.mean(),
            "lyap_loss": lyap_step,
            "lyap_loss_per_branch": lyap_per_branch,
            "lyap_per_branch": lyap_per_branch_t,
            "lambda_step": lambda_step,
            "lambda_per_branch": lambda_per_branch,
            "sep_loss": sep,
        }
        if lyap_lambda > 0:
            aux["lyap_loss_total"] = lyap_step
        if sep_lambda > 0:
            aux["sep_loss_total"] = sep_lambda * sep
        return h_next, outs, aux


__all__ = ["PerStepAdaptiveAuxMultiBasinLyapunovCfCCell"]
