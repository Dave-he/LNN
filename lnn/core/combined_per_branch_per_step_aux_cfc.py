"""CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell (round 255).

Closes the 9-round arc (r246-254) by combining the two complementary
aux-gating axes:

  * r253 (per-branch):  λ_k = λ_max · H_k / log(n_basin)
  * r254 (per-step):    λ_t = λ_max · mean_k(H_k_t) / log(n_basin)
  * r255 (this, combined): λ_k,t = λ_max · (H_k / log n_basin) · (mean_k H_k / log n_basin)

The combined mechanism is a **product** of the two axes — aux fires
only when BOTH:
  1. The branch is uncertain (high H_k), AND
  2. The whole network is in a transition (high mean H across branches)

This is a 2D gating mechanism that is more conservative than either
axis alone: contraction fires only at the intersection of branch
uncertainty AND temporal transition. Confident branches in stable
states get zero aux; uncertain branches in stable states get small
aux (per-step low); confident branches in transitions get small aux
(per-branch low); only uncertain branches in transitions get full aux.

Hypothesis (PRD #10-92, round 255):
  H1: combined aux matches r253/r254 in toy regime (safe superset).
  H2: combined aux lambda is more SPARSE than r253/r254 (product
      of two [0, 1] factors is more conservative).
  H3: combined aux is the most robust production default.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from lnn.core.per_branch_multibasin_lyapunov_cfc import (
    PerBranchMultiBasinLyapunovCfCCell,
)


class CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell(
    PerBranchMultiBasinLyapunovCfCCell
):
    """PerBranch + combined per-branch × per-step adaptive aux (round 255).

    Args:
        lyap_lambda_max: Upper bound on aux weight.
        combination: "product" (default) or "max" or "mean" of the two axes.
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
        combination: str = "product",
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
        assert combination in ("product", "max", "mean")
        self.combination = combination

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

        # Per-branch and per-step aux weights (both in [0, 1] after scaling).
        log_nb = math.log(float(self.n_basin))
        if log_nb > 0:
            H_norm_per_branch = H_per_branch_t.detach() / log_nb  # (K,)
            H_norm_step = H_per_branch_t.detach().mean() / log_nb  # scalar
            H_norm_per_branch = H_norm_per_branch.clamp(0.0, 1.0)
            H_norm_step = H_norm_step.clamp(0.0, 1.0)
        else:
            H_norm_per_branch = torch.zeros_like(H_per_branch_t)
            H_norm_step = torch.zeros(())

        if self.combination == "product":
            # Per-branch × per-step (broadcast scalar to all branches).
            lambda_per_branch_combined = (
                float(lyap_lambda) * H_norm_per_branch * H_norm_step
            )
        elif self.combination == "max":
            lambda_per_branch_combined = (
                float(lyap_lambda) * torch.max(H_norm_per_branch, H_norm_step)
            )
        else:  # mean
            lambda_per_branch_combined = (
                float(lyap_lambda)
                * 0.5 * (H_norm_per_branch + H_norm_step)
            )

        # Aux loss: weighted sum.
        lyap_combined = (lambda_per_branch_combined * lyap_per_branch_t).sum()

        # Per-branch only (r253) and per-step only (r254) for ablation.
        lambda_per_branch_only = (
            float(lyap_lambda) * H_norm_per_branch
        )
        lyap_per_branch_only = (lambda_per_branch_only * lyap_per_branch_t).sum()
        lambda_step_only = float(lyap_lambda) * H_norm_step
        lyap_step_only = lambda_step_only * lyap_per_branch_t.sum()

        aux: dict[str, torch.Tensor] = {
            "h_next": h_next,
            "alpha_mix": self.alpha_mix.detach(),
            "per_branch_V_next": V_per_branch_t,
            "per_branch_basin_H": H_per_branch_t,
            "mean_basin_H": H_per_branch_t.mean(),
            "lyap_loss": lyap_combined,
            "lyap_loss_per_branch": lyap_per_branch_only,
            "lyap_loss_per_step": lyap_step_only,
            "lyap_per_branch": lyap_per_branch_t,
            "lambda_combined": lambda_per_branch_combined,
            "lambda_per_branch": lambda_per_branch_only,
            "lambda_step": lambda_step_only,
            "sep_loss": sep,
        }
        if lyap_lambda > 0:
            aux["lyap_loss_total"] = lyap_combined
        if sep_lambda > 0:
            aux["sep_loss_total"] = sep_lambda * sep
        return h_next, outs, aux


__all__ = ["CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell"]
