"""SqrtGatedPulseCfCCell — round 286.

Paper grounding (/loop 2026-07-12, continuation):
    Round 285 (linear `g_t`-gated pulse) was a HONEST NEGATIVE-WITH-NUANCE.
    The linear gate suppressed noise somewhat (Δ% +44.6 → +9.4) but
    destroyed the structured gap-robustness (gap_ratio 61 → 394) and
    amplified the parameter chase on `A` (0.40 → 0.71).

    The failure mode is **shape**: a linear multiplicative gate is too
    aggressive on high-`g_t` steps (structured) where the pulse needs to
    carry state through gaps. A shape-preserving gate, `sqrt(g_t)`,
    keeps more amplitude on structured (sqrt(0.8)=0.89 vs 0.80) while
    still attenuating noise (sqrt(0.1)=0.32 vs 0.10). The gradient on
    `A` is also scaled by `1/sqrt(g_t)` instead of `1/g_t`, which may
    slow the A-chase on noise.

Mechanism (strict superset of r284 AND r285)::

    g_eff = { g_t           if gate_pulse_shape='linear' (≡ r285)
            { sqrt(g_t)     if gate_pulse_shape='sqrt'   (r286, default)
            { 1.0           if gate_pulse_shape='none'   (≡ r284)
    pulse_i = g_eff · strength · A_i · sin(...)

    - ``gate_pulse_shape='none'`` ⇒ r284 (ungated pulse) bit-for-bit.
    - ``gate_pulse_shape='linear'`` ⇒ r285 bit-for-bit (for back-compat).
    - ``gate_pulse_shape='sqrt', pulse_strength=0`` ⇒ r280 blend cell.

Hypotheses (PRD #10-127):

    H1 (structured gap_ratio ≤ r284 = 61).
    H2 (random Δ% ≤ +5% vs blend_gated; r285 was +9.4%, r284 was +44.6%).
    H3 (random pulse_amp ≤ 0.20; r285 was 0.71, r284 was 0.40).
    H4 (H1 AND H2 AND H3 all pass) → strict-positive default — the
       first in the r284/r285/r286 pulse line.
    H5 (gate_pulse_shape='none' ≡ r284) — unit test.
    H6 (gate_pulse_shape='linear' ≡ r285) — unit test.

API::

    SqrtGatedPulseCfCCell(input_size, hidden_size, density=0.3,
        pred_gate_beta=4.0, ema_gamma=0.5, gate_mode='blend',
        pulse_strength=1.0, pulse_amp_init=0.1, pulse_mode='sin',
        state_phase=True, gate_pulse_shape='sqrt', ...)
"""

from __future__ import annotations

import torch

from lnn.core.predictability_gated_pulse_cfc import (
    PredictabilityGatedPulseCfCCell,
)


_VALID_SHAPES = ("sqrt", "linear", "none")


class SqrtGatedPulseCfCCell(PredictabilityGatedPulseCfCCell):
    """PulseGatedLiquidTauCfCCell with a *shape-preserving* gate on the
    pulse amplitude.

    Args:
        (all r284 args, plus:)
        gate_pulse_shape: ``'sqrt'`` (default r286 — `pulse = sqrt(g_t)·A·sin`),
            ``'linear'`` (≡ r285 — `pulse = g_t·A·sin`), or ``'none'``
            (≡ r284 — unconditional pulse).
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
        gate_pulse_shape: str = "sqrt",
    ):
        if gate_pulse_shape not in _VALID_SHAPES:
            raise ValueError(
                f"gate_pulse_shape must be one of {_VALID_SHAPES}, "
                f"got {gate_pulse_shape!r}")
        # Map gate_pulse_shape to the parent's boolean + we'll apply the
        # shape transform in `_pulse_term`. Use a non-trivial marker: we
        # always pass gate to the parent (gate_pulse=True) and then apply
        # the shape via a private method.
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
            gate_pulse=(gate_pulse_shape != "none"),
        )
        self.gate_pulse_shape = gate_pulse_shape

    # ------------------------------------------------------------------
    def _shape_gate(self, gate: torch.Tensor) -> torch.Tensor:
        """Apply the configured shape transform to ``gate``.

        ``gate`` is the per-step r280 blend score, shape (B, 1), values
        in (0, 1]. We clamp to ≥ 0 for numerical safety before the
        square root.
        """
        if self.gate_pulse_shape == "sqrt":
            return gate.clamp(min=0.0).sqrt()
        if self.gate_pulse_shape == "linear":
            return gate
        # 'none' branch: parent never calls this because gate_pulse=False.
        return gate

    # ------------------------------------------------------------------
    def _pulse_term(self, t: int, T: int, h: torch.Tensor,
                    noise_drive: torch.Tensor | None,
                    gate: torch.Tensor | None = None) -> torch.Tensor:
        out = super()._pulse_term(t, T, h, noise_drive)
        if (self.gate_pulse_shape != "none"
                and gate is not None
                and self.pulse_strength != 0.0):
            out = self._shape_gate(gate) * out
        return out


__all__ = ["SqrtGatedPulseCfCCell"]