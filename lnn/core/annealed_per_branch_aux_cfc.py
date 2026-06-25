"""AnnealedPerBranchMultiBasinLyapunovCfCCell (round 256).

Pivots the 10-round arc (r246-255) to a NEW axis: **training-epoch
annealing** of the aux weight.

  * r252: constant λ = 0.1 across all training — hurts on toy_sin/random
  * r253/r254/r255: H-gated λ — collapses to 0 in toy regime
  * r256 (this): epoch-annealed λ = λ_max · max(0, 1 - ep / T_anneal)

The mechanism applies a strong contraction prior EARLY in training
(when the model needs regularization) and lets the task dominate LATE
in training (when the model has converged to task-specific solutions).

This tests the hypothesis:

  **Contraction prior is most useful as INITIAL regularizer, not as
  persistent training signal.**

API::

    AnnealedPerBranchMultiBasinLyapunovCfCCell(input_size, hidden_size,
                                               n_branches=4, n_basin=3,
                                               lyap_lambda_max=0.1,
                                               anneal_epochs=50)

Hypothesis (PRD #10-93, round 256):
  H1: annealed aux improves early loss reduction (faster convergence).
  H2: annealed aux matches r248 final performance (no regression at ep=100).
  H3: annealed aux beats constant aux (r252) on toy_sin/random where
      constant aux hurts.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from lnn.core.per_branch_multibasin_lyapunov_cfc import (
    PerBranchMultiBasinLyapunovCfCCell,
)


class AnnealedPerBranchMultiBasinLyapunovCfCCell(
    PerBranchMultiBasinLyapunovCfCCell
):
    """PerBranch + epoch-annealed constant aux (round 256).

    Args:
        lyap_lambda_max: Upper bound on aux weight (epoch 0).
        anneal_epochs: Number of epochs to anneal from λ_max to 0.
        anneal_schedule: "linear" (default) or "cosine" or "exp".
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
        anneal_epochs: int = 50,
        anneal_schedule: str = "linear",
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
        self.anneal_epochs = int(anneal_epochs)
        assert anneal_schedule in ("linear", "cosine", "exp")
        self.anneal_schedule = anneal_schedule
        self._current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch for annealing. Call this before each epoch."""
        self._current_epoch = int(epoch)

    def get_lambda(self) -> float:
        """Compute the current λ based on epoch and schedule."""
        if self.anneal_epochs <= 0:
            return self.default_lyap_lambda_max
        ratio = min(1.0, max(0.0, self._current_epoch / self.anneal_epochs))
        if self.anneal_schedule == "linear":
            scale = max(0.0, 1.0 - ratio)
        elif self.anneal_schedule == "cosine":
            scale = 0.5 * (1.0 + math.cos(math.pi * ratio))
            scale = max(0.0, scale)
        else:  # exp
            scale = math.exp(-3.0 * ratio)
        return self.default_lyap_lambda_max * scale

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h_list: list[torch.Tensor],
        lyap_lambda: float | None = None,
        sep_lambda: float = 0.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        if lyap_lambda is None:
            lyap_lambda = self.get_lambda()

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

        # Constant aux (per epoch) — same as r252 but with annealed λ.
        lyap_const = lyap_lambda * lyap_per_branch_t.sum()

        aux: dict[str, torch.Tensor] = {
            "h_next": h_next,
            "alpha_mix": self.alpha_mix.detach(),
            "per_branch_V_next": V_per_branch_t,
            "per_branch_basin_H": H_per_branch_t,
            "mean_basin_H": H_per_branch_t.mean(),
            "lyap_loss": lyap_const,
            "lyap_per_branch": lyap_per_branch_t,
            "sep_loss": sep,
            "current_lambda": torch.tensor(lyap_lambda),
        }
        if lyap_lambda > 0:
            aux["lyap_loss_total"] = lyap_const
        if sep_lambda > 0:
            aux["sep_loss_total"] = sep_lambda * sep
        return h_next, outs, aux


__all__ = ["AnnealedPerBranchMultiBasinLyapunovCfCCell"]
