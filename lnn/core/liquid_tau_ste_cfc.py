"""LiquidTauSTECfCCell — input-dependent (liquid) τ on top of STE (round 277).

Motivation (2026 literature audit, /loop 2026-07-03):
    Every 2026 LTC/CfC paper found (Efficient Semantic Segmentation
    via LTC with Adaptive Dynamics; LTC for urban-drainage water-level
    forecasting; Liquid NN for natural-gas spot price arXiv:2604.24788)
    emphasises the **input-dependent time constant** as the core source
    of the "liquid" advantage on nonstationary sequences.

    Our STE sparsity line (r263-r276) is built on ``NeuronWiseCfCCell``,
    which uses a **static, learned per-neuron τ** (``tau_per_neuron`` is a
    plain ``nn.Parameter`` fixed after training):

        raw = sigmoid(tau_per_neuron)                # (d_h,) — no dep on x_t
        τ   = tau_min + (tau_max - tau_min) * raw

    This is a "half-liquid" base: it never restores the defining
    input-dependent τ of Hasani 2021 LTC / CfC. Round 277 tests whether
    making τ **flow with the input** helps.

Mechanism::

    τ_i(t) = τ_min + (τ_max - τ_min) *
             sigmoid( tau_per_neuron_i + s · (W_τ · [x_t, h_{t-1}])_i )

    - ``tau_per_neuron`` is the inherited static per-neuron bias.
    - ``W_τ`` is a NEW gate (input_size + hidden_size → hidden_size).
    - ``s`` = ``liquid_tau_strength`` scales the gate contribution.
    - ``W_τ`` is **zero-initialised** ⇒ at init, τ_i(t) == the static
      per-neuron τ exactly (strict superset of r265/r267). Any liquid
      behaviour is learned, not imposed.

The STE hard/soft neighborhood mask and the soft-mask entropy penalty
are inherited unchanged from ``STEWithEntropy``. Only the τ computation
in the forward loop changes (now per-timestep, per-batch).

Hypotheses (PRD #10-114):

    H1: liquid τ beats static τ on ≥1 dataset (esp. structured —
        the analogue of the papers' nonstationary claim).
    H2: liquid τ does NOT hurt toy_sin (smooth single-freq data
        doesn't need adaptation) — ties or mild regression.
    H3: zero-init gate ⇒ training-start equivalence to static τ
        (no added instability).
    H4: the learned τ actually flows: temporal std of τ across
        timesteps > 0 after training.

API::

    LiquidTauSTECfCCell(input_size, hidden_size, density=0.3,
                        entropy_lambda=0.0, ste_temperature=1.0,
                        liquid_tau_strength=1.0, ...)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.ste_entropy_neuron_wise_cfc import STEWithEntropy


class LiquidTauSTECfCCell(STEWithEntropy):
    """STEWithEntropy with input-dependent (liquid) per-neuron τ.

    Args:
        input_size: Input feature dimension (d_in).
        hidden_size: Hidden state dimension (d_h).
        density: Fraction of edges to keep per row (forward).
        base_tau: Initial τ for all neurons (via static bias).
        tau_min: Lower bound of τ.
        tau_max: Upper bound of τ.
        alpha_max: Absolute value clamp for per-neuron α.
        ste_temperature: Temperature for the soft sigmoid backward.
        entropy_lambda: Soft-mask entropy penalty (inherited). 0 disables.
        liquid_tau_strength: Scale ``s`` on the τ gate. 0.0 ⇒ exactly
            static τ (degenerates to r267). Default 1.0.
        init_rec_scale: Scale of recurrent weight init.
        input_strength_init: Initial per-neuron input strength.
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
        liquid_tau_strength: float = 1.0,
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
            entropy_lambda=entropy_lambda,
            init_rec_scale=init_rec_scale,
            input_strength_init=input_strength_init,
            seed=seed,
        )
        if liquid_tau_strength < 0:
            raise ValueError("liquid_tau_strength must be >= 0")
        self.liquid_tau_strength = float(liquid_tau_strength)

        # --- NEW: τ gate from [x_t, h_{t-1}] → per-neuron τ modulation ---
        # Zero-init so at start τ_i(t) == static per-neuron τ exactly.
        self.W_tau = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.W_tau.weight)

    # ------------------------------------------------------------------
    # Liquid τ
    # ------------------------------------------------------------------
    def get_tau_dynamic(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Input-dependent per-neuron τ, bounded to [tau_min, tau_max].

        Args:
            x_t: (B, d_in) current input.
            h: (B, d_h) previous hidden state.

        Returns:
            (B, d_h) τ values that vary per-sample and per-timestep.
        """
        gate = self.W_tau(torch.cat([x_t, h], dim=-1))  # (B, d_h)
        logit = self.tau_per_neuron.unsqueeze(0) + self.liquid_tau_strength * gate
        raw = torch.sigmoid(logit)  # (B, d_h) in (0, 1)
        return self.tau_min + (self.tau_max - self.tau_min) * raw

    # ------------------------------------------------------------------
    # Forward (τ recomputed per-timestep from [x_t, h])
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """Run the cell; τ flows with the input at every timestep.

        Args:
            x: (B, T, d_in) input sequence.
            h0: (B, d_h) initial hidden state. Defaults to zeros.
            return_aux: If True, also return a dict of diagnostics
                (including the liquid-τ temporal-variation summary).

        Returns:
            outputs: (B, T, d_h) hidden states at each step.
            h_final: (B, d_h) final hidden state.
            aux (optional): dict of diagnostics.
        """
        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        if h0 is None:
            h = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)
        else:
            h = h0

        # Fixed-per-forward quantities (mask + static parts).
        mask = self.get_neighborhood_mask()
        W_eff = mask * self.W_rec  # (d_h, d_h)
        alpha = self.get_alpha()  # (d_h,)

        outputs = []
        tau_steps = [] if return_aux else None  # for temporal-flow diagnostic
        for t in range(T):
            x_t = x[:, t, :]  # (B, d_in)
            tau_t = self.get_tau_dynamic(x_t, h)  # (B, d_h) — liquid τ
            rec = h @ W_eff.T  # (B, d_h)
            in_proj = self.W_in(x_t)  # (B, d_h)
            in_proj = self.input_strength_per_neuron.unsqueeze(0) * in_proj
            s = rec + in_proj + self.bias_per_neuron + alpha.unsqueeze(0) * h
            # CfC-style per-neuron leaky update with input-dependent τ.
            h = (1.0 - tau_t) * h + tau_t * torch.tanh(s)
            outputs.append(h)
            if return_aux:
                tau_steps.append(tau_t.detach())

        out = torch.stack(outputs, dim=1)
        if not return_aux:
            return out, h

        # Temporal-flow diagnostic: how much does τ move across time?
        tau_stack = torch.stack(tau_steps, dim=1)  # (B, T, d_h)
        # std across the time axis, averaged over batch × neuron.
        tau_temporal_std = float(tau_stack.std(dim=1).mean().item())
        aux = {
            "mask": mask.detach(),
            "neighborhood_density": self.neighborhood_density(mask),
            "neighborhood_asymmetry": self.neighborhood_asymmetry(mask),
            "tau_summary": self.per_neuron_tau_summary(),  # static bias only
            "tau_temporal_std": tau_temporal_std,  # liquid flow magnitude
            "tau_dynamic_mean": float(tau_stack.mean().item()),
            "tau_dynamic_min": float(tau_stack.min().item()),
            "tau_dynamic_max": float(tau_stack.max().item()),
            "liquid_tau_strength": self.liquid_tau_strength,
            "alpha_mean": float(alpha.mean().item()),
            "alpha_std": float(alpha.std().item()),
        }
        return out, h, aux


__all__ = ["LiquidTauSTECfCCell"]
