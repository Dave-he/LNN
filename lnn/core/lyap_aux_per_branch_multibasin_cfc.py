"""LyapAuxPerBranchMultiBasinLyapunovCfCCell (round 252).

Lifts round 248's per-branch multi-basin Lyapunov architecture with
**built-in Lyap auxiliary supervision**. Round 251 tested this on
*frozen* basin centers (no grad path through basin geometry) and found
that aux supervision hurts task loss on smooth/structured data.

This round tests the hypothesis:

  **Learned basins + aux supervision might WORK** because the basin
  centers can adapt to satisfy BOTH the task prior and the contraction
  prior simultaneously. Frozen basins cannot adapt, hence the r251
  regression.

Hypothesis (round 252 PRD):

  H1: LyapAuxPerBranch recovers task-loss parity with r248 within ±10%
      per dataset (test if learned basins can co-exist with aux).
  H2: aux loss decreases over training.
  H3: V contracts (V_next <= V_prev × (1 - alpha)).

API::

    LyapAuxPerBranchMultiBasinLyapunovCfCCell(input_size, hidden_size,
                                               n_branches=4, n_basin=3,
                                               lyap_lambda=0.1)
"""

from __future__ import annotations

import torch

from lnn.core.per_branch_multibasin_lyapunov_cfc import (
    PerBranchMultiBasinLyapunovCfCCell,
)


class LyapAuxPerBranchMultiBasinLyapunovCfCCell(
    PerBranchMultiBasinLyapunovCfCCell
):
    """PerBranchMultiBasinLyapunovCfCCell with built-in Lyap aux supervision.

    Args:
        lyap_lambda: Default aux loss weight (can be overridden in
            ``forward_with_aux``).
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
        lyap_lambda: float = 0.1,
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
        self.default_lyap_lambda = float(lyap_lambda)

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h_list: list[torch.Tensor],
        lyap_lambda: float | None = None,
        sep_lambda: float = 0.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        if lyap_lambda is None:
            lyap_lambda = self.default_lyap_lambda
        return super().forward_with_aux(
            x_t, h_list, lyap_lambda=lyap_lambda, sep_lambda=sep_lambda,
        )


__all__ = ["LyapAuxPerBranchMultiBasinLyapunovCfCCell"]