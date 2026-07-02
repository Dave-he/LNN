"""AccelGatedLiquidTauCfCCell — acceleration-gated liquid τ (round 279).

Motivation (r278 follow-up, /loop 2026-07-03):
    Round 278's predictability gate ``g_t = exp(-beta * vol_t)`` with
    ``vol_t = EMA|x_t - x_{t-1}|`` (first difference) fixed r277's
    catastrophic noise regression (random +106% → +0.3%) but LOST
    r277's toy_sin win (+41% vs static). The report diagnosed the
    cause: a clean sine has large *velocity* |Δx| even though it is
    perfectly predictable, so the first-difference gate throttles the
    liquid τ (gate ≈ 0.49-0.79 on sine) below its optimum.

    The fix, found by pre-bench signal analysis (r279): gate on the
    **second difference** (acceleration) |x_t - 2·x_{t-1} + x_{t-2}|
    instead of the first. A smooth signal has small acceleration even
    when it has large velocity — a sine's Δ²x is itself a scaled sine,
    an order of magnitude below its Δ¹x. Pure i.i.d. noise, by contrast,
    has large acceleration everywhere (each sample is independent).

    Measured (EMA vol, 64-step, 16 seqs, β=4):

        signal          sine   structured  noise   noise/sine
        |Δ¹x| (r278)     0.186    0.057      1.139     6.1×
        |Δ²x| (r279)     0.057    0.115      1.986    35.1×

        gate (β=4)      sine    structured  noise
        Δ¹x (r278)      0.494     0.896      0.053
        Δ²x (r279)      0.800     0.839      0.018   ← sine recovered

    The second difference is the *constant-velocity forecast error*:
    it is ~0 for any locally-linear (hence predictable) trajectory and
    large only for genuinely erratic input. This lets the predictable
    sine through (gate 0.80 ⇒ near-full liquid ⇒ recovers toy_sin)
    while still collapsing on noise (gate 0.018 ⇒ keeps r278's fix).

Mechanism (parameter-free acceleration gate)::

    accel_t = mean_c |x_t - 2·x_{t-1} + x_{t-2}|        # 2nd difference
    vol_t   = EMA_γ(accel_t)                            # causal smoothed
    g_t     = exp( -beta * vol_t )   ∈ (0, 1]           # smooth→1, erratic→0
    τ_i(t)  = tau_min + (tau_max - tau_min) *
              sigmoid( tau_bias_i + g_t · s · (W_τ·[x_t, h])_i )

    - Parameter-free ⇒ cannot chase noise. First two timesteps have no
      2nd difference ⇒ accel=0 ⇒ g=1 (full liquid), consistent with
      r278's t=0 convention.
    - ``diff_order=1`` ⇒ exactly r278 (first-difference gate, superset).

Hypotheses (PRD #10-117 revised):

    H1 (headline): accel gate recovers toy_sin toward liquid's -59%
       (better than r278's +41%).
    H2: accel gate keeps the random fix (random Δ% ≤ +5% vs static).
    H3: accel gate preserves the structured win (≤ static).
    H4: gate_mean(sine) >> gate_mean(noise) after the fix (mechanism).
    H5: diff_order=1 reproduces r278 exactly (superset).

API::

    AccelGatedLiquidTauCfCCell(input_size, hidden_size, density=0.3,
        entropy_lambda=0.0, ste_temperature=1.0, liquid_tau_strength=1.0,
        pred_gate_beta=4.0, ema_gamma=0.5, diff_order=2, ...)
"""

from __future__ import annotations

import torch

from lnn.core.pred_gated_liquid_tau_cfc import PredictabilityGatedLiquidTauCfCCell


class AccelGatedLiquidTauCfCCell(PredictabilityGatedLiquidTauCfCCell):
    """PredictabilityGatedLiquidTauCfCCell whose volatility estimate uses
    the *second* difference (acceleration) instead of the first.

    Args:
        (all r278 args, plus:)
        diff_order: ``2`` (default) gates on |Δ²x| (acceleration), which
            lets smooth-but-fast signals through. ``1`` reproduces r278
            (gates on |Δ¹x|, velocity).
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
        diff_order: int = 2,
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
            init_rec_scale=init_rec_scale,
            input_strength_init=input_strength_init,
            seed=seed,
        )
        if diff_order not in (1, 2):
            raise ValueError("diff_order must be 1 (r278) or 2 (accel)")
        self.diff_order = int(diff_order)

    # ------------------------------------------------------------------
    # Forward (τ = acceleration-gated liquid τ, per-timestep)
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """Run the cell; liquid τ contribution is gated by input
        *acceleration* (second difference) at every timestep.

        Args:
            x: (B, T, d_in) input sequence.
            h0: (B, d_h) initial hidden state. Defaults to zeros.
            return_aux: If True, also return a dict of diagnostics.

        Returns:
            outputs: (B, T, d_h) hidden states at each step.
            h_final: (B, d_h) final hidden state.
            aux (optional): dict of diagnostics.
        """
        # diff_order=1 → identical to r278; defer to the parent forward.
        if self.diff_order == 1:
            return super().forward(x, h0=h0, return_aux=return_aux)

        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        if h0 is None:
            h = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)
        else:
            h = h0

        mask = self.get_neighborhood_mask()
        W_eff = mask * self.W_rec  # (d_h, d_h)
        alpha = self.get_alpha()  # (d_h,)

        vol = torch.zeros(B, device=device, dtype=dtype)  # EMA of |Δ²x|
        prev_x = None  # x_{t-1}
        prev_prev_x = None  # x_{t-2}

        outputs = []
        tau_steps = [] if return_aux else None
        gate_steps = [] if return_aux else None
        for t in range(T):
            x_t = x[:, t, :]  # (B, d_in)
            # Causal acceleration = |x_t - 2 x_{t-1} + x_{t-2}| (no peek).
            if prev_x is not None and prev_prev_x is not None:
                accel = (x_t - 2.0 * prev_x + prev_prev_x).abs().mean(dim=-1)
                vol = self.ema_gamma * vol + (1.0 - self.ema_gamma) * accel
            gate = self.predictability_gate(vol)  # (B, 1)
            prev_prev_x = prev_x
            prev_x = x_t

            tau_t = self.get_tau_gated(x_t, h, gate)  # (B, d_h)
            rec = h @ W_eff.T
            in_proj = self.W_in(x_t)
            in_proj = self.input_strength_per_neuron.unsqueeze(0) * in_proj
            s = rec + in_proj + self.bias_per_neuron + alpha.unsqueeze(0) * h
            h = (1.0 - tau_t) * h + tau_t * torch.tanh(s)
            outputs.append(h)
            if return_aux:
                tau_steps.append(tau_t.detach())
                gate_steps.append(gate.detach())

        out = torch.stack(outputs, dim=1)
        if not return_aux:
            return out, h

        tau_stack = torch.stack(tau_steps, dim=1)  # (B, T, d_h)
        gate_stack = torch.stack(gate_steps, dim=1)  # (B, T, 1)
        aux = {
            "mask": mask.detach(),
            "neighborhood_density": self.neighborhood_density(mask),
            "neighborhood_asymmetry": self.neighborhood_asymmetry(mask),
            "tau_summary": self.per_neuron_tau_summary(),
            "tau_temporal_std": float(tau_stack.std(dim=1).mean().item()),
            "tau_dynamic_mean": float(tau_stack.mean().item()),
            "gate_mean": float(gate_stack.mean().item()),
            "gate_min": float(gate_stack.min().item()),
            "gate_max": float(gate_stack.max().item()),
            "diff_order": self.diff_order,
            "pred_gate_beta": self.pred_gate_beta,
            "ema_gamma": self.ema_gamma,
            "liquid_tau_strength": self.liquid_tau_strength,
            "alpha_mean": float(alpha.mean().item()),
            "alpha_std": float(alpha.std().item()),
        }
        return out, h, aux


__all__ = ["AccelGatedLiquidTauCfCCell"]
