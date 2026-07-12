"""PredictabilityGatedPulseCfCCell — round 285.

Paper grounding (/loop 2026-07-12):
    Round 284 (arXiv:2603.00153, Pulse-Driven Neural Architecture) added a
    learnable oscillatory pulse to the gated liquid-τ cell. The pulse
    bought real gap-robustness on structured data (6× vs blend gate) but
    broke the gate line's "parameter-free ⇒ noise-safe" invariant — the
    learned amplitude grew 4× on random noise (+44.6% MSE).

    The r284 report itself recommended gating the pulse amplitude by the
    r280 predictability score `g_t ∈ (0,1]`:
        `pulse = g_t · A · sin(...)`
    so the endogenous drive is suppressed exactly when input is erratic
    (restoring noise safety) but active on predictable / gappy data
    (keeping the robustness). This round does exactly that — **zero new
    parameters, zero new loss, zero new schedule** — and asks whether
    that combination (a) keeps H3 (structured gap-robustness) and (b)
    restores H2 (noise safety).

Mechanism (strict superset of r284)::

    gate_t  = max(g_vel, g_acc)                  # r280 (per-step scalar)
    pulse_i = g_t · strength · A_i · sin(...)    # ← THE FIX
    s_i     = rec_i + in_i + bias_i + α_i·h_i + pulse_i
    h       = (1-τ)·h + τ·tanh(s)

    - ``gate_pulse=False`` ⇒ pulse is unconditional ⇒ reproduces r284
      ``PulseGatedLiquidTauCfCCell`` bit-for-bit (superset).
    - ``gate_pulse=True, pulse_strength=0`` ⇒ pulse is exactly 0 ⇒
      reproduces r280 ``BlendGatedLiquidTauCfCCell`` bit-for-bit.

Hypotheses (PRD #10-126):

    H1 (robustness preserved): structured gap_ratio (gated_pulse) ≤
       blend (368) AND ≤ r284 (61).
    H2 (safety restored, THE FIX): random Δ% (gated_pulse vs blend) ≤
       +5% (r284 was +44.6%).
    H3 (amplitude no longer chases noise): on random, final
       pulse_amp.abs().mean() ≤ 0.20 (r284 grew to 0.40).
    H4 (superset, gate_pulse=False): ≡ PulseGatedLiquidTauCfCCell
       forward output bit-equal within float tolerance.
    H5 (gating not just clamping): on structured+gap, post-training
       gate.mean() ≥ 0.5 so the pulse is still active when input is
       predictable.

API::

    PredictabilityGatedPulseCfCCell(input_size, hidden_size, density=0.3,
        pred_gate_beta=4.0, ema_gamma=0.5, gate_mode='blend',
        pulse_strength=1.0, pulse_amp_init=0.1, pulse_mode='sin',
        state_phase=True, gate_pulse=True, ...)
"""

from __future__ import annotations

import torch

from lnn.core.pulse_gated_liquid_tau_cfc import PulseGatedLiquidTauCfCCell


class PredictabilityGatedPulseCfCCell(PulseGatedLiquidTauCfCCell):
    """PulseGatedLiquidTauCfCCell with pulse amplitude gated by the
    per-step r280 predictability score ``g_t``.

    Args:
        (all r284 args, plus:)
        gate_pulse: if True (default, r285) the per-step gate
            ``g_t = max(g_vel, g_acc)`` multiplies the pulse term so the
            endogenous drive is suppressed exactly when input is erratic.
            If False the cell is bit-equivalent to
            ``PulseGatedLiquidTauCfCCell`` (superset; unit-tested).
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
        gate_pulse: bool = True,
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
        )
        self.gate_pulse = bool(gate_pulse)

    # ------------------------------------------------------------------
    def _pulse_term(self, t: int, T: int, h: torch.Tensor,
                    noise_drive: torch.Tensor | None,
                    gate: torch.Tensor | None = None) -> torch.Tensor:
        """Return the (B, d_h) pulse contribution for timestep ``t``.

        If ``gate`` (B,1) is provided AND ``self.gate_pulse`` is True,
        the pulse amplitude is multiplied by ``g_t`` so the endogenous
        drive is suppressed exactly when the input is erratic (the r280
        predictability score).
        """
        out = super()._pulse_term(t, T, h, noise_drive)
        if self.gate_pulse and gate is not None and self.pulse_strength != 0.0:
            out = gate * out
        return out

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """Run the predictability-gated pulse-augmented liquid-τ cell.

        Identical to the r284 forward except the pulse term is multiplied
        by the per-step r280 blend gate (when ``gate_pulse=True``).
        Gradients flow through ``g_t`` into ``pulse_amp``, ``pulse_omega``,
        ``pulse_phase0``, and ``pulse_phase_proj``.
        """
        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        if h0 is None:
            h = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)
        else:
            h = h0

        mask = self.get_neighborhood_mask()
        W_eff = mask * self.W_rec
        alpha = self.get_alpha()

        noise_drive = None
        if self.pulse_mode == "noise" and self.pulse_strength != 0.0:
            gen = torch.Generator(device="cpu").manual_seed(self.pulse_seed)
            noise_drive = torch.randn(T, self.hidden_size, generator=gen).to(
                device=device, dtype=dtype)

        vol1 = torch.zeros(B, device=device, dtype=dtype)
        vol2 = torch.zeros(B, device=device, dtype=dtype)
        prev_x = None
        prev_prev_x = None

        outputs = []
        if return_aux:
            tau_steps: list = []
            gate_steps: list = []
            pulse_steps: list = []
        else:
            tau_steps = gate_steps = pulse_steps = None
        for t in range(T):
            x_t = x[:, t, :]
            if prev_x is not None:
                d1 = (x_t - prev_x).abs().mean(dim=-1)
                vol1 = self.ema_gamma * vol1 + (1.0 - self.ema_gamma) * d1
            if prev_x is not None and prev_prev_x is not None:
                d2 = (x_t - 2.0 * prev_x + prev_prev_x).abs().mean(dim=-1)
                vol2 = self.ema_gamma * vol2 + (1.0 - self.ema_gamma) * d2
            g_vel = torch.exp(-self.pred_gate_beta * vol1)
            g_acc = torch.exp(-self.pred_gate_beta * vol2)
            if self.gate_mode == "velocity":
                gate = g_vel
            elif self.gate_mode == "acceleration":
                gate = g_acc
            else:
                gate = torch.max(g_vel, g_acc)
            gate = gate.unsqueeze(-1)
            prev_prev_x = prev_x
            prev_x = x_t

            tau_t = self.get_tau_gated(x_t, h, gate)
            rec = h @ W_eff.T
            in_proj = self.W_in(x_t)
            in_proj = self.input_strength_per_neuron.unsqueeze(0) * in_proj
            pulse = self._pulse_term(t, T, h, noise_drive, gate=gate)
            s = rec + in_proj + self.bias_per_neuron + alpha.unsqueeze(0) * h + pulse
            h = (1.0 - tau_t) * h + tau_t * torch.tanh(s)
            outputs.append(h)
            if return_aux:
                tau_steps.append(tau_t.detach())
                gate_steps.append(gate.detach())
                pulse_steps.append(pulse.detach())

        out = torch.stack(outputs, dim=1)
        if not return_aux:
            return out, h

        tau_stack = torch.stack(tau_steps, dim=1)
        gate_stack = torch.stack(gate_steps, dim=1)
        pulse_stack = torch.stack(pulse_steps, dim=1)
        aux = {
            "mask": mask.detach(),
            "tau_summary": self.per_neuron_tau_summary(),
            "tau_temporal_std": float(tau_stack.std(dim=1).mean().item()),
            "tau_dynamic_mean": float(tau_stack.mean().item()),
            "gate_mean": float(gate_stack.mean().item()),
            "gate_min": float(gate_stack.min().item()),
            "gate_max": float(gate_stack.max().item()),
            "gate_mode": self.gate_mode,
            "pulse_mode": self.pulse_mode,
            "pulse_strength": self.pulse_strength,
            "pulse_amp_mean": float(self.pulse_amp.abs().mean().item()),
            "pulse_amp_max": float(self.pulse_amp.abs().max().item()),
            "pulse_rms": float(pulse_stack.pow(2).mean().sqrt().item()),
            "pulse_omega_mean": float(self.pulse_omega.mean().item()),
            "state_phase": self.state_phase,
            "gate_pulse": self.gate_pulse,
            "alpha_mean": float(alpha.mean().item()),
        }
        return out, h, aux


__all__ = ["PredictabilityGatedPulseCfCCell"]