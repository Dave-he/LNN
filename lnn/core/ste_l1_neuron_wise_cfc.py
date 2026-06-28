"""STEWithL1 — STENeuronWiseCfCCell + L1 sparsity reg (round 266).

A direct refinement of r265's STENeuronWiseCfCCell that adds
an L1 penalty on ``neighbor_logits`` to encourage concentrated
structure.

The hypothesis: STE already provides implicit sparsity (hard
top-k in forward), but the soft sigmoid gradient may produce
**ambiguous** structure (many logits of similar magnitude).
An L1 penalty pushes the model toward a few dominant edges.

Mechanism::

    L_total = L_task + λ × mean(|neighbor_logits|)

The forward pass is unchanged from r265. The L1 penalty is
added to the training loss via ``extra_loss()``.

Why this might help:
1. Concentrated structure: a few clear edges vs. many
   ambiguous ones.
2. Reduced overfitting: spurious edges in the top-k are
   penalized.
3. Better generalization: simpler structure may transfer
   better.

Why this might hurt:
1. L1 is sub-optimal for sparsity vs. L0 or hard top-k.
2. STE already provides sparsity in the forward pass.
3. Strong L1 may collapse neighbor_logits to zero, making
   top-k selection purely random.

Hypotheses (PRD #10-103):

  H1: STE with L1 (any λ) beats r265's no-L1 baseline on ≥ 1
      dataset.
  H2: Sweet spot for λ — too small (no effect) or too large
      (collapses structure) is worse.
  H3: L1-penalized model has std > 1.5 × no-L1 std (more
      concentrated).
  H4: STE + L1 is a strict superset of r265 no-L1 (λ → 0
      recovers r265).

API::

    STEWithL1(input_size, hidden_size, density=0.3,
               l1_lambda=0.0, ste_temperature=1.0,
               ...)
"""

from __future__ import annotations

import torch

from lnn.core.ste_neuron_wise_cfc import STENeuronWiseCfCCell


class STEWithL1(STENeuronWiseCfCCell):
    """STENeuronWiseCfCCell with L1 sparsity reg on neighbor_logits.

    Args:
        input_size: Input feature dimension (d_in).
        hidden_size: Hidden state dimension (d_h).
        density: Fraction of edges to keep per row (forward).
        base_tau: Initial τ for all neurons.
        tau_min: Lower bound of the learned τ.
        tau_max: Upper bound of the learned τ.
        alpha_max: Absolute value clamp for per-neuron α.
        ste_temperature: Temperature for the soft sigmoid in
            the backward pass.
        l1_lambda: L1 penalty on neighbor_logits. 0.0 disables.
        init_rec_scale: Scale of the recurrent weight init.
        input_strength_init: Initial per-neuron input
            projection strength.
        seed: Random seed for init.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        density: float = 0.3,
        base_tau: float = 0.5,
        tau_min: float = 0.05,
        tau_max: float = 0.95,
        alpha_max: float = 0.5,
        ste_temperature: float = 1.0,
        l1_lambda: float = 0.0,
        init_rec_scale: float | None = None,
        input_strength_init: float = 0.1,
        seed: int = 42,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            density=density,
            base_tau=base_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            alpha_max=alpha_max,
            ste_temperature=ste_temperature,
            init_rec_scale=init_rec_scale,
            input_strength_init=input_strength_init,
            seed=seed,
        )
        if l1_lambda < 0:
            raise ValueError("l1_lambda must be >= 0")
        self.l1_lambda = float(l1_lambda)

    def extra_loss(self) -> torch.Tensor:
        """L1 penalty on neighbor_logits.

        Returns:
            Scalar tensor = l1_lambda × mean(|neighbor_logits|).
        """
        if self.l1_lambda <= 0:
            return torch.tensor(0.0, device=self.neighbor_logits.device)
        return self.l1_lambda * self.neighbor_logits.abs().mean()

    def l1_loss_value(self) -> float:
        """Current L1 penalty value (for diagnostics)."""
        return float(self.extra_loss().item())


__all__ = ["STEWithL1"]
