"""Aux-Supervised Frozen Random Basin CfC (round 251).

Lifts round 250's frozen random basin centers with **auxiliary
supervision** — the Lyap contraction loss is enabled during training
(``lyap_lambda > 0``) so that the frozen geometry enters the gradient
through the aux path.

Round 250 found that frozen basins reduce to round 246 when
``lyap_lambda=0`` because basin geometry never enters the loss.
This round closes that gap: with aux supervision enabled, the
frozen basins can influence training through the contraction loss.

Hypothesis (round 251 PRD):

  H1: aux-supervised frozen basins recover task loss parity with
      learned basins (round 248) within ±10% per dataset.
  H2: aux loss decreases monotonically over training.
  H3: V_next converges toward V_prev × (1 - alpha).

API::

    AuxSupervisedFrozenRandomBasinCfCCell(input_size, hidden_size,
                                           n_branches=4, n_basin=3,
                                           tau_min=0.05, tau_max=20.0,
                                           lyap_lambda=0.1)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.frozen_random_basin_cfc import FrozenRandomBasinCfCCell


class AuxSupervisedFrozenRandomBasinCfCCell(FrozenRandomBasinCfCCell):
    """FrozenRandomBasinCfCCell with **built-in Lyap aux supervision**.

    The ``forward_with_aux`` method automatically applies ``lyap_lambda``
    when called during training. During eval, pass ``lyap_lambda=0``
    (the default) to skip aux.

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
        basin_seed: int = 137,
        tau_seed: int = 42,
        alpha: float = 0.05,
        beta_v: float = 2.0,
        lyap_lambda: float = 0.1,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            n_branches=n_branches,
            n_basin=n_basin,
            tau_min=tau_min,
            tau_max=tau_max,
            basin_seed=basin_seed,
            tau_seed=tau_seed,
            alpha=alpha,
            beta_v=beta_v,
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
        """Forward + aux. Uses ``self.default_lyap_lambda`` if not provided."""
        if lyap_lambda is None:
            lyap_lambda = self.default_lyap_lambda
        return super().forward_with_aux(
            x_t, h_list, lyap_lambda=lyap_lambda, sep_lambda=sep_lambda,
        )


__all__ = ["AuxSupervisedFrozenRandomBasinCfCCell"]