"""EmaGatedPulseCfCCell — round 288.

Paper grounding (/loop 2026-07-12, continuation):
    Four rounds of pulse variants (r284 / r285 / r286 / r287) revealed an
    anti-correlated trade-off: stronger gating → better noise safety but
    worse gap-robustness. The root cause: ANY per-step gate interrupts
    the pulse when `g_t` momentarily dips during a gap, breaking the
    "continuous endogenous rhythm" claim of arXiv:2603.00153.

    This round attacks the root cause by **EMA-smoothing the gate**:
        `g_ema_t = α · g_t + (1-α) · g_ema_{t-1}`
    and thresholding `g_ema_t` (not `g_t`). On structured + gap: when
    input drops out, `g_t` momentarily dips but `g_ema` stays high (α·1
    decays slowly from initial 1.0) → pulse fires continuously through
    the gap. On noise: `g_t` is *consistently* low → `g_ema` collapses
    → mask stays off → no noise chasing.

Mechanism::

    g_ema_t = α · g_t + (1-α) · g_ema_{t-1}    # EMA-smoothed gate
    mask_t  = 1.0 if g_ema_t > τ else 0.0      # binary threshold
    pulse_i = mask_t · strength · A_i · sin(...)

    - ``ema_alpha=1.0`` ⇒ no smoothing ⇒ ``g_ema_t = g_t`` ⇒
      ≡ r287 (BinaryGatedPulseCfCCell) bit-for-bit (superset).
    - ``threshold=0`` ⇒ pulse always on ⇒ ≡ r284.
    - ``threshold=10`` ⇒ pulse always off ⇒ ≡ r280.

Hypotheses (PRD #10-129):

    H1 (structured gap_ratio ≤ r284 = 61) — the headline test.
    H2 (random Δ% ≤ +5% vs blend) at τ ∈ {0.3, 0.5}.
    H3 (random pulse_amp ≤ 0.20).
    H4 (H1 ∧ H2 ∧ H3) → strict-positive default — **first in the line**.
    H5 (ema_alpha=1.0 ≡ r287) — unit test.
    H6 (threshold=0 ≡ r284) — unit test.

API::

    EmaGatedPulseCfCCell(input_size, hidden_size, density=0.3,
        pred_gate_beta=4.0, ema_gamma=0.5, gate_mode='blend',
        pulse_strength=1.0, pulse_amp_init=0.1, pulse_mode='sin',
        state_phase=True, threshold=0.5, ema_alpha=0.3,
        g_ema_init=1.0, ...)
"""

from __future__ import annotations

import torch

from lnn.core.binary_gated_pulse_cfc import BinaryGatedPulseCfCCell


class EmaGatedPulseCfCCell(BinaryGatedPulseCfCCell):
    """BinaryGatedPulseCfCCell with an EMA-smoothed gate.

    The per-step ``g_t`` (r280 blend score) is exponentially smoothed
    before thresholding. This decouples the per-step gate fluctuation
    from the firing decision, so the pulse can fire continuously through
    structured input gaps while still being suppressed on consistently
    noisy inputs.

    Args:
        (all r287 args, plus:)
        ema_alpha: EMA smoothing factor in (0, 1]. Smaller ⇒ more
            smoothing (g_ema_t dominated by history). Larger ⇒ less
            smoothing (closer to raw g_t). Default 0.3. ``ema_alpha=1.0``
            ≡ no smoothing ≡ r287 bit-for-bit.
        g_ema_init: initial value of g_ema (before step 1). Default 1.0
            so the pulse fires on the first step regardless of input.
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
        ema_alpha: float = 0.3,
        g_ema_init: float = 1.0,
    ):
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(
                f"ema_alpha must be in (0, 1], got {ema_alpha}")
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
            threshold=threshold,
            gate_pulse=gate_pulse,
        )
        self.ema_alpha = float(ema_alpha)
        self.g_ema_init = float(g_ema_init)
        # EMA state is held in the cell, not on the aux dict. Reset to
        # init at the start of every forward() call.
        self._g_ema_state: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def _reset_state(self, B: int, device: torch.device,
                     dtype: torch.dtype) -> None:
        self._g_ema_state = torch.full(
            (B, 1), self.g_ema_init, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    def _pulse_term(self, t: int, T: int, h: torch.Tensor,
                    noise_drive: torch.Tensor | None,
                    gate: torch.Tensor | None = None) -> torch.Tensor:
        out = super()._pulse_term(t, T, h, noise_drive)
        if (self.gate_pulse
                and gate is not None
                and self.pulse_strength != 0.0):
            # Update EMA state. ema_alpha=1.0 ⇒ no smoothing (superset).
            if self._g_ema_state is None:
                self._reset_state(h.shape[0], h.device, h.dtype)
            assert self._g_ema_state is not None
            self._g_ema_state = (
                self.ema_alpha * gate
                + (1.0 - self.ema_alpha) * self._g_ema_state
            )
            mask = (self._g_ema_state > self.threshold).float()
            out = mask * out
        return out

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        # Reset EMA state at the start of every forward call so the
        # test cells see a clean slate.
        self._g_ema_state = None
        return super().forward(x, h0=h0, return_aux=return_aux)


__all__ = ["EmaGatedPulseCfCCell"]