"""STEWithEntropy — STENeuronWiseCfCCell + soft-mask entropy reg (round 267).

A direct refinement of r265's STENeuronWiseCfCCell that adds a
**soft-mask entropy penalty** to encourage concentrated structure.

The hypothesis (PRD #10-104): the right regularizer targets the
**concentration** of the soft sigmoid mask, not the **magnitude**
of neighbor_logits (which is what r266 L1 targeted — and that
collapsed the logits).

Mechanism::

    L_total = L_task + λ × H(soft_mask_row)

Where H is the per-row Shannon entropy of the soft sigmoid
mask (after row-softmax):

    p_i = softmax(soft_mask_row_i)
    H_i = -sum_j p_i[j] · log(p_i[j] + eps)

For uniform soft mask: H_i = log(d_h) (max entropy).
For peaked soft mask: H_i ≈ 0 (min entropy).

Why entropy and not L1:
  - L1 targets logit magnitudes → collapses logits → destroys
    the ranking that hard top-k uses (r266 NEGATIVE).
  - Entropy targets soft mask distribution → preserves logit
    magnitudes → only regularizes the **concentration** of the
    soft mask.
  - Entropy is bounded (∈ [0, log(d_h)]) so λ doesn't need to
    scale with logit magnitude.
  - Backward: entropy gradient is smooth and bounded (unlike
    L1's non-smooth at zero).

Hypotheses (PRD #10-104):

  H1: STE + entropy reg (any λ) beats r265 no-reg on ≥ 1 dataset.
  H2: Sweet spot for λ — too small (no effect) or too large
      (over-regularizes) is worse.
  H3: Entropy reg REDUCES soft-mask entropy without collapsing
      logits (unlike r266 L1).
  H4: Entropy reg preserves logit std (≥ 0.5 × no-reg std).
  H5: Entropy reg is a strict superset of no-reg (λ → 0
      recovers r265).

API::

    STEWithEntropy(input_size, hidden_size, density=0.3,
                   entropy_lambda=0.0, ste_temperature=1.0,
                   ...)
"""

from __future__ import annotations

import torch

from lnn.core.ste_neuron_wise_cfc import STENeuronWiseCfCCell


class STEWithEntropy(STENeuronWiseCfCCell):
    """STENeuronWiseCfCCell with soft-mask entropy reg.

    The entropy is computed on the **soft sigmoid mask**
    (the differentiable backward component), then row-softmaxed
    to form a probability distribution, then Shannon entropy
    per row.

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
        entropy_lambda: Entropy penalty on soft mask. 0.0
            disables.
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
        entropy_lambda: float = 0.0,
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
        if entropy_lambda < 0:
            raise ValueError("entropy_lambda must be >= 0")
        self.entropy_lambda = float(entropy_lambda)

    def _soft_mask_entropy(self) -> torch.Tensor:
        """Per-row Shannon entropy of the soft sigmoid mask.

        Returns:
            Scalar tensor = mean over rows of entropy.
            Range: [0, log(d_h)].
        """
        soft = torch.sigmoid(self.neighbor_logits / self.ste_temperature)
        # Row-softmax to convert soft mask values to probabilities.
        # This treats each row as a distribution over source neurons.
        p = torch.softmax(soft, dim=-1)
        # Shannon entropy per row: -sum p log p (with eps for stability).
        eps = 1e-8
        H = -(p * torch.log(p + eps)).sum(dim=-1)  # (d_h,)
        return H.mean()

    def extra_loss(self) -> torch.Tensor:
        """Entropy penalty on the soft sigmoid mask.

        Returns:
            Scalar tensor = entropy_lambda × mean_row_entropy.
        """
        if self.entropy_lambda <= 0:
            return torch.tensor(0.0, device=self.neighbor_logits.device)
        return self.entropy_lambda * self._soft_mask_entropy()

    def entropy_value(self) -> float:
        """Current soft-mask entropy (for diagnostics)."""
        return float(self._soft_mask_entropy().item())

    def max_entropy(self) -> float:
        """Maximum possible entropy = log(d_h)."""
        import math
        return math.log(self.hidden_size)


__all__ = ["STEWithEntropy"]