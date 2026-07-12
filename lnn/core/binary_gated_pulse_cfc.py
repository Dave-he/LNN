"""BinaryGatedPulseCfCCell — round 287.

Paper grounding (/loop 2026-07-12, continuation):
    Three rounds of multiplicative-gate variants (r284 / r285 / r286) all
    produced target-dependent results with different trade-offs but no
    strict-positive default. The failure mode is amplitude scaling:
    the optimizer grows `A` to compensate for the multiplicative
    attenuation at high-`g_t` steps, and the larger A then leaks through
    on noise steps.

    This round abandons the multiplicative-gate family entirely. The
    new mechanism is an **additive / threshold gate**:
        `pulse = (g_t > τ) · A · sin(...)`
    The pulse is *full strength* when `g_t > τ` and *exactly zero*
    otherwise. No attenuation on the active steps → no compensation by
    A → no noise leakage.

Mechanism::

    g_eff_t = 1.0 if g_t > τ else 0.0
    pulse_i = g_eff_t · strength · A_i · sin(...)

    - ``threshold=0.0`` ⇒ g_eff is always 1.0 ⇒ unconditional pulse
      (≡ r284 PulseGatedLiquidTauCfCCell bit-for-bit).
    - ``threshold=10.0`` ⇒ g_eff is always 0.0 ⇒ zero pulse
      (≡ r280 BlendGatedLiquidTauCfCCell bit-for-bit).
    - ``threshold=0.5`` (default) ⇒ pulse fires only when g_t > 0.5,
      i.e. on structured/gappy data (gate ≈ 0.8) but not on noise
      (gate ≈ 0.1).

Hypotheses (PRD #10-128):

    H1 (structured gap_ratio ≤ r284 = 61) at τ ∈ {0.3, 0.5}.
    H2 (random Δ% ≤ +5% vs blend) at τ ∈ {0.3, 0.5}.
    H3 (random pulse_amp ≤ 0.20) — A-chase killed because the optimizer
       gets *zero* gradient on A when the pulse is suppressed (no
       pulse → no A contribution → no A gradient).
    H4 (H1 ∧ H2 ∧ H3) → strict-positive default — **first in the line**.
    H5 (threshold=0 ≡ r284) — unit test.
    H6 (threshold=10 ≡ r280) — unit test.

API::

    BinaryGatedPulseCfCCell(input_size, hidden_size, density=0.3,
        pred_gate_beta=4.0, ema_gamma=0.5, gate_mode='blend',
        pulse_strength=1.0, pulse_amp_init=0.1, pulse_mode='sin',
        state_phase=True, threshold=0.5, ...)
"""

from __future__ import annotations

import torch

from lnn.core.predictability_gated_pulse_cfc import (
    PredictabilityGatedPulseCfCCell,
)


class BinaryGatedPulseCfCCell(PredictabilityGatedPulseCfCCell):
    """PulseGatedLiquidTauCfCCell with an additive *threshold gate* on
    the pulse term.

    The gate is binary: ``pulse = (gate > threshold).float() * raw_pulse``.
    When the input is predictable (g_t > τ) the pulse fires at *full*
    amplitude. When the input is erratic (g_t ≤ τ) the pulse is exactly
    zero — no leak, no A-chase.

    Args:
        (all r284 args, plus:)
        threshold: scalar in [0, 1+]. The gate must exceed this value
            for the pulse to fire. Default 0.5. ``threshold=0`` ⇒ pulse
            is always on (≡ r284); ``threshold > 1`` ⇒ pulse is always
            off (≡ r280 blend).
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
        pred_gate_beta: float = 4.0,
        ema_gamma: float = 0.5,
        gate_mode: str = "blend",
        pulse_strength: float = 1.0,
        pulse_amp_init: float = 0.1,
        pulse_mode: str = "sin",
        state_phase: bool = True,
        pulse_seed: int = 7,
        init_rec_scale: float | None = None,
        input_strength_init: float = 0.1,
        seed: int = 42,
        threshold: float = 0.5,
        gate_pulse: bool = True,
    ):
        if not (0.0 <= threshold):
            raise ValueError(
                f"threshold must be ≥ 0, got {threshold}")
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
            liquid_tau_strength=liquid_tau_strength,
            pred_gate_beta=pred_gate_beta,
            ema_gamma=ema_gamma,
            gate_mode=gate_mode,
            pulse_strength=pulse_strength,
            pulse_amp_init=pulse_amp_init,
            pulse_mode=pulse_mode,
            state_phase=state_phase,
            pulse_seed=pulse_seed,
            init_rec_scale=init_rec_scale,
            input_strength_init=input_strength_init,
            seed=seed,
            gate_pulse=gate_pulse,
        )
        self.threshold = float(threshold)

    # ------------------------------------------------------------------
    def _pulse_term(self, t: int, T: int, h: torch.Tensor,
                    noise_drive: torch.Tensor | None,
                    gate: torch.Tensor | None = None) -> torch.Tensor:
        out = super()._pulse_term(t, T, h, noise_drive)
        if (self.gate_pulse
                and gate is not None
                and self.pulse_strength != 0.0):
            # Binary gate: 1 if gate > threshold else 0.
            mask = (gate > self.threshold).float()
            out = mask * out
        return out


__all__ = ["BinaryGatedPulseCfCCell"]