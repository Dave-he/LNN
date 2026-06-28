"""STE-NeuronWiseCfCCell (round 265).

A CfC variant that combines the **best of both worlds** from
r263 (hard top-k sparsification) and r264 (soft attention):

  - **Forward pass**: hard top-k (binary mask, true sparsity)
  - **Backward pass**: soft mask (sigmoid, gradients flow)

This is the **straight-through estimator (STE)** pattern
(Jang et al. 2016 Gumbel-Softmax, Courbariaux BinaryNet,
Bengio 2013 Estimating or Propagating Gradients Through
Stochastic Neurons).

The r264 report (SoftNeuronAttentionCfCCell) was an HONEST
NEGATIVE on H1 (soft beats hard). Soft attention underperformed
hard top-k by 6.6× on structured. The fix identified by the
r264 report was: use STE to keep the hard mask in forward
(preserving r263's true sparsity) but enable gradient flow
in backward (preserving r264's differentiability).

Why this works:
  1. Forward: mask = hard (binary, top-k per row). Same as r263.
  2. Backward: ∂L/∂neighbor_logits = ∂L/∂soft · ∂soft/∂logits.
     soft is a sigmoid, so gradients flow smoothly.
  3. The "soft" and "hard" masks are typically very close
     (top-k of logits ≈ top-k of sigmoid(neighbor_logits)),
     so the approximation error is small.

The cell extends r263's NeuronWiseCfCCell with STE. All other
behavior (per-neuron τ, per-neuron α, per-neuron input
strength) is inherited unchanged.

Hypotheses (PRD #10-102):

  H1: STE beats r263 (hard top-k, non-learnable) on at least
      one dataset.
  H2: STE beats r264 (soft attention) on at least one dataset.
  H3: The learned neighbor_logits become structured (std > 0.05
      after training).
  H4: STE is a strict superset of r263 (at τ_ste → 0) and r264
      (at τ_ste → ∞).

API::

    STENeuronWiseCfCCell(input_size, hidden_size, density=0.3,
                          base_tau=0.5, tau_min=0.05, tau_max=0.95,
                          alpha_max=0.5, ste_temperature=1.0,
                          seed=42)
"""

from __future__ import annotations

import torch

from lnn.core.neuron_wise_cfc import (
    NeuronWiseCfCCell,
    sparse_topk_mask,
)


class STENeuronWiseCfCCell(NeuronWiseCfCCell):
    """NeuronWiseCfCCell with **straight-through estimator** for
    the neighborhood mask.

    Inherits all of r263's per-neuron dynamics. The only change
    is in `get_neighborhood_mask()`:

      hard = sparse_topk_mask(neighbor_logits, density)   # binary
      soft = sigmoid(neighbor_logits / τ_ste)             # differentiable
      mask_STE = (hard - soft).detach() + soft            # STE

    The forward pass uses `hard`. The backward pass computes
    gradients via `soft`.

    Args:
        input_size: Input feature dimension (d_in).
        hidden_size: Hidden state dimension (d_h).
        density: Fraction of edges to keep per row in the
            hard mask (forward). 1.0 = fully connected;
            0.3 = sparse.
        base_tau: Initial τ for all neurons.
        tau_min: Lower bound of the learned τ.
        tau_max: Upper bound of the learned τ.
        alpha_max: Absolute value clamp for per-neuron α.
        ste_temperature: Temperature for the soft sigmoid in
            the backward pass. Smaller = sharper gradient.
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
            init_rec_scale=init_rec_scale,
            input_strength_init=input_strength_init,
            seed=seed,
        )
        if ste_temperature <= 0:
            raise ValueError("ste_temperature must be > 0")
        self.ste_temperature = float(ste_temperature)

    def get_neighborhood_mask(self) -> torch.Tensor:
        """Straight-through estimator mask.

        Forward: hard top-k (binary, true sparsity).
        Backward: soft sigmoid (gradients flow to neighbor_logits).

        Returns:
            (d_h, d_h) tensor. Forward values are 0 or 1.
        """
        # Hard top-k (binary mask, true sparsity) — like r263.
        hard = sparse_topk_mask(self.neighbor_logits, self.density)
        # Soft sigmoid mask (differentiable).
        soft = torch.sigmoid(self.neighbor_logits / self.ste_temperature)
        # STE: (hard - soft).detach() + soft
        # Forward: hard (because soft is detached and re-added → cancels).
        # Backward: ∂L/∂soft flows (because soft is not detached).
        return (hard - soft).detach() + soft

    def get_ste_soft_mask(self) -> torch.Tensor:
        """The soft (differentiable) mask used in the STE backward.

        Useful for diagnostics and tests.
        """
        return torch.sigmoid(self.neighbor_logits / self.ste_temperature)

    def get_ste_hard_mask(self) -> torch.Tensor:
        """The hard (binary) mask used in the STE forward.

        Useful for diagnostics and tests.
        """
        return sparse_topk_mask(self.neighbor_logits, self.density)


__all__ = ["STENeuronWiseCfCCell"]
