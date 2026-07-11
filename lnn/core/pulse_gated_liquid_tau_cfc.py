"""PulseGatedLiquidTauCfCCell — pulse-augmented gated liquid τ (round 284).

Paper grounding (/loop 2026-07-11):
    arXiv:2603.00153 "Pulse-Driven Neural Architecture: Learnable
    Oscillatory Dynamics for Robust Continuous-Time Sequence Processing"
    (Paras Sharma, 2026-03). The paper augments a CfC cell with a
    *learnable oscillatory pulse* ``A·sin(ω·t + φ(h))`` so the hidden
    state keeps evolving with an endogenous rhythm even when the input is
    erratic or absent (gaps). Its headline control: a *non-oscillatory*
    perturbation of equal magnitude gives NO benefit — the temporal
    STRUCTURE of the pulse is what matters, not added capacity/noise.

This round grafts that pulse onto the r280 blend-gated liquid-τ cell
(the current production gate). The gate decides *how liquid* τ is at each
step (r278 velocity / r279 acceleration / r280 blend); the pulse adds an
endogenous oscillatory drive to the pre-activation so a periodic or
gap-interrupted signal can be carried by the cell's own rhythm.

Mechanism (superset of r280)::

    gate_t  = { g_vel, g_acc, max(g_vel,g_acc) }         # r278/r279/r280
    τ_i(t)  = tau_min + (tau_max-tau_min)·σ(bias_i + gate_t·s·(W_τ[x_t,h])_i)
    phase_i = φ0_i + (W_φ·h)_i            (state-dependent phase, optional)
    pulse_i = strength · A_i · sin(2π·ω_i·(t/T) + phase_i)   # 'sin' mode
    s_i     = rec_i + in_i + bias_i + α_i·h_i + pulse_i
    h       = (1-τ)·h + τ·tanh(s)

    - ``pulse_strength=0`` ⇒ pulse term is exactly 0 ⇒ reproduces the
      r280 BlendGatedLiquidTauCfCCell bit-for-bit (strict superset).
    - ``pulse_mode='noise'`` replaces sin(·) with an RMS-matched
      non-oscillatory random drive — the paper's mechanism control
      (equal magnitude, no temporal structure).
    - ``state_phase=False`` uses a pure learnable oscillator (phase = φ0
      only); ``True`` makes the phase depend on the hidden state (φ(h)),
      closer to the paper's self-referential pulse.

Hypotheses (PRD #10-125):

    H1 (headline): the sin pulse helps the periodic toy_sin task
       (structure matches) — pulse_sin ≤ gated_blend on toy_sin.
    H2 (safety): the pulse is safe on random noise — its amplitude A
       shrinks toward 0 and pulse_sin Δ% ≤ +5% vs gated_blend.
    H3 (robustness, paper claim): under eval-time temporal dropout (input
       gaps), pulse_sin degrades LESS than gated_blend — the endogenous
       rhythm carries the state through gaps.
    H4 (superset): pulse_strength=0 reproduces r280 exactly.
    H5 (mechanism, paper control): pulse_mode='noise' (equal-magnitude,
       non-oscillatory) does NOT reproduce the sin benefit — structure,
       not added capacity, is responsible.

API::

    PulseGatedLiquidTauCfCCell(input_size, hidden_size, density=0.3,
        pred_gate_beta=4.0, ema_gamma=0.5, gate_mode='blend',
        pulse_strength=1.0, pulse_amp_init=0.1, pulse_mode='sin',
        state_phase=True, n_pulse_freqs=1, ...)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from lnn.core.blend_gated_liquid_tau_cfc import BlendGatedLiquidTauCfCCell


class PulseGatedLiquidTauCfCCell(BlendGatedLiquidTauCfCCell):
    """BlendGatedLiquidTauCfCCell augmented with a learnable oscillatory
    pulse injected into the per-neuron pre-activation.

    Args:
        (all r280 args, plus:)
        pulse_strength: global multiplier on the pulse term. ``0.0``
            disables the pulse and reproduces r280 exactly (superset).
        pulse_amp_init: init scale of the per-neuron amplitude A_i (kept
            small so training starts near the r280 baseline).
        pulse_mode: ``'sin'`` (learnable oscillator, default),
            ``'noise'`` (RMS-matched non-oscillatory control, the paper's
            mechanism ablation), or ``'off'`` (alias for strength 0).
        state_phase: if True the pulse phase is φ0 + W_φ·h (state
            dependent, per the paper); if False it is a pure learnable
            oscillator (phase = φ0).
        pulse_seed: seed for the fixed 'noise'-mode drive (reproducible).
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
            init_rec_scale=init_rec_scale,
            input_strength_init=input_strength_init,
            seed=seed,
        )
        if pulse_mode not in ("sin", "noise", "off"):
            raise ValueError("pulse_mode must be 'sin', 'noise', or 'off'")
        self.pulse_mode = pulse_mode
        self.pulse_strength = float(pulse_strength if pulse_mode != "off" else 0.0)
        self.state_phase = bool(state_phase)
        self.pulse_seed = int(pulse_seed)

        d_h = hidden_size
        gen = torch.Generator().manual_seed(seed + 1000)
        # Per-neuron amplitude A_i (small init so we start near baseline).
        self.pulse_amp = nn.Parameter(
            torch.abs(torch.randn(d_h, generator=gen)) * pulse_amp_init)
        # Per-neuron angular frequency ω_i (cycles over the sequence);
        # init spread across 0.5..4 cycles so neurons cover time scales.
        self.pulse_omega = nn.Parameter(
            0.5 + 3.5 * torch.rand(d_h, generator=gen))
        # Per-neuron base phase φ0_i.
        self.pulse_phase0 = nn.Parameter(
            2.0 * math.pi * torch.rand(d_h, generator=gen))
        # State→phase coupling W_φ (only used when state_phase=True).
        self.pulse_phase_proj = nn.Linear(d_h, d_h, bias=False)
        with torch.no_grad():
            self.pulse_phase_proj.weight.mul_(0.1)

    # ------------------------------------------------------------------
    def _pulse_term(self, t: int, T: int, h: torch.Tensor,
                    noise_drive: torch.Tensor | None) -> torch.Tensor:
        """Return the (B, d_h) pulse contribution for timestep ``t``."""
        if self.pulse_strength == 0.0:
            return torch.zeros_like(h)
        amp = self.pulse_amp.unsqueeze(0)  # (1, d_h)
        if self.pulse_mode == "noise":
            # RMS-matched non-oscillatory control: sin has RMS = A/√2.
            return self.pulse_strength * amp * noise_drive[t].unsqueeze(0) / math.sqrt(2.0)
        t_norm = t / max(T, 1)
        phase = self.pulse_phase0.unsqueeze(0)  # (1, d_h)
        if self.state_phase:
            phase = phase + self.pulse_phase_proj(h)  # (B, d_h)
        angle = 2.0 * math.pi * self.pulse_omega.unsqueeze(0) * t_norm + phase
        return self.pulse_strength * amp * torch.sin(angle)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """Run the pulse-augmented gated liquid-τ cell.

        Identical to the r280 blend cell except an endogenous oscillatory
        pulse is added to the pre-activation at every timestep.
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

        # Fixed non-oscillatory drive for the 'noise' control (reproducible,
        # same shape as the sin pulse would occupy).
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
        tau_steps = [] if return_aux else None
        gate_steps = [] if return_aux else None
        pulse_steps = [] if return_aux else None
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
            pulse = self._pulse_term(t, T, h, noise_drive)
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
        pulse_stack = torch.stack(pulse_steps, dim=1)  # (B, T, d_h)
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
            "alpha_mean": float(alpha.mean().item()),
        }
        return out, h, aux

    # ------------------------------------------------------------------
    def pulse_summary(self) -> dict:
        """Learned-pulse diagnostics (no forward pass needed)."""
        return {
            "amp_mean": float(self.pulse_amp.abs().mean().item()),
            "amp_max": float(self.pulse_amp.abs().max().item()),
            "omega_mean": float(self.pulse_omega.mean().item()),
            "omega_std": float(self.pulse_omega.std().item()),
            "pulse_mode": self.pulse_mode,
            "state_phase": self.state_phase,
        }


__all__ = ["PulseGatedLiquidTauCfCCell"]
