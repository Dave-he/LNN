"""PredictabilityGatedLiquidTauCfCCell — predictability-gated liquid τ (round 278).

Motivation (2026 literature audit, /loop 2026-07-03 #2):
    Round 277 found liquid (input-dependent) τ is a *target-dependent
    positive*: it WINS on predictable data (toy_sin -59%, structured
    -12%) but HURTS badly on pure i.i.d. noise (random +106%). The
    failure mode is clear — on structureless input the τ gate chases
    noise and over-adapts, destabilising the recurrence.

    The 2026 natural-gas-price LNN paper (arXiv:2604.24788) frames the
    exact tension: dynamics must "limit responsiveness when market
    regimes shift rapidly" — i.e. be responsive on structured regime
    change but NOT chase unpredictable jitter. The urban-flood CfC
    (2026) and SCTP-Net (2026) echo: discrete networks struggle with
    nonstationarity, but naive adaptivity is unstable on noise.

    Round 277's own report proposed the fix: gate the liquid strength
    on a signal-predictability estimate (analogous to the r99
    reliability gate).

Mechanism (parameter-free predictability gate)::

    vol_t   = EMA_γ( mean_c |x_t - x_{t-1}| )        # causal input volatility
    g_t     = exp( -beta * vol_t )   ∈ (0, 1]         # predictable→1, noisy→0
    τ_i(t)  = tau_min + (tau_max - tau_min) *
              sigmoid( tau_bias_i + g_t · s · (W_τ·[x_t, h])_i )

    - ``g_t`` scales the *liquid* contribution only. When the input is
      smooth/structured (low volatility) g_t≈1 ⇒ full liquid τ (recovers
      r277). When the input is noisy (high volatility) g_t≈0 ⇒ τ collapses
      to the static per-neuron bias (recovers r267, the stable baseline).
    - The gate has **NO learnable parameters** ⇒ it *cannot* learn to
      chase noise. This is the whole point: it structurally forbids the
      r277 failure mode.
    - ``beta`` controls sensitivity; ``ema_gamma`` smooths the volatility
      estimate. First timestep has no predecessor ⇒ vol_0 = 0 ⇒ g_0 = 1.

This is a strict superset of r277 in the low-noise limit (beta→0 ⇒ g_t≡1
⇒ exactly r277) and a strict superset of r267 in the high-noise limit
(g_t→0 ⇒ exactly static τ). It interpolates between them per-timestep
based on *observed* input predictability.

Hypotheses (PRD #10-115):

    H1: gated liquid τ recovers r277's WINS on toy_sin/structured
        (g_t≈1 there ⇒ near-full liquid).
    H2: gated liquid τ FIXES r277's random regression (g_t≈0 there ⇒
        collapses to stable static τ) — the headline test.
    H3: the gate value g_t is high on toy_sin/structured, low on random
        (mechanism check via diagnostics).
    H4: beta=0 exactly reproduces r277 (superset property).

API::

    PredictabilityGatedLiquidTauCfCCell(input_size, hidden_size,
        density=0.3, entropy_lambda=0.0, ste_temperature=1.0,
        liquid_tau_strength=1.0, pred_gate_beta=4.0, ema_gamma=0.5, ...)
"""

from __future__ import annotations

import torch

from lnn.core.liquid_tau_ste_cfc import LiquidTauSTECfCCell


class PredictabilityGatedLiquidTauCfCCell(LiquidTauSTECfCCell):
    """LiquidTauSTECfCCell with a parameter-free predictability gate on
    the liquid τ contribution.

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
        liquid_tau_strength: Base scale ``s`` on the τ gate.
        pred_gate_beta: Sensitivity ``beta`` of the predictability gate.
            0.0 ⇒ gate ≡ 1 ⇒ exactly r277. Higher ⇒ more aggressive
            collapse to static τ under volatility. Default 4.0.
        ema_gamma: EMA smoothing for the volatility estimate in [0, 1).
            0 ⇒ instantaneous |x_t - x_{t-1}|. Default 0.5.
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
        pred_gate_beta: float = 4.0,
        ema_gamma: float = 0.5,
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
            init_rec_scale=init_rec_scale,
            input_strength_init=input_strength_init,
            seed=seed,
        )
        if pred_gate_beta < 0:
            raise ValueError("pred_gate_beta must be >= 0")
        if not (0.0 <= ema_gamma < 1.0):
            raise ValueError("ema_gamma must be in [0, 1)")
        self.pred_gate_beta = float(pred_gate_beta)
        self.ema_gamma = float(ema_gamma)

    # ------------------------------------------------------------------
    # Predictability gate + gated liquid τ
    # ------------------------------------------------------------------
    def predictability_gate(self, vol: torch.Tensor) -> torch.Tensor:
        """Map a (B,) volatility estimate to a (B,1) gate in (0, 1].

        g = exp(-beta * vol). Smooth input (vol≈0) → 1; noisy → 0.
        """
        g = torch.exp(-self.pred_gate_beta * vol)  # (B,)
        return g.unsqueeze(-1)  # (B, 1)

    def get_tau_gated(
        self, x_t: torch.Tensor, h: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        """Input-dependent τ scaled by the predictability gate.

        Args:
            x_t: (B, d_in) current input.
            h: (B, d_h) previous hidden state.
            gate: (B, 1) predictability gate.

        Returns:
            (B, d_h) τ values.
        """
        raw_gate = self.W_tau(torch.cat([x_t, h], dim=-1))  # (B, d_h)
        logit = self.tau_per_neuron.unsqueeze(0) + (
            gate * self.liquid_tau_strength * raw_gate
        )
        raw = torch.sigmoid(logit)
        return self.tau_min + (self.tau_max - self.tau_min) * raw

    # ------------------------------------------------------------------
    # Forward (τ = predictability-gated liquid τ, per-timestep)
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """Run the cell; liquid τ contribution is gated by input
        predictability at every timestep.

        Args:
            x: (B, T, d_in) input sequence.
            h0: (B, d_h) initial hidden state. Defaults to zeros.
            return_aux: If True, also return a dict of diagnostics
                (including per-step gate + liquid-flow summaries).

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

        mask = self.get_neighborhood_mask()
        W_eff = mask * self.W_rec  # (d_h, d_h)
        alpha = self.get_alpha()  # (d_h,)

        vol = torch.zeros(B, device=device, dtype=dtype)  # EMA of |Δx|
        prev_x = None

        outputs = []
        tau_steps = [] if return_aux else None
        gate_steps = [] if return_aux else None
        for t in range(T):
            x_t = x[:, t, :]  # (B, d_in)
            # Causal input volatility (no peeking at future).
            if prev_x is not None:
                inst = (x_t - prev_x).abs().mean(dim=-1)  # (B,)
                vol = self.ema_gamma * vol + (1.0 - self.ema_gamma) * inst
            gate = self.predictability_gate(vol)  # (B, 1)
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
            "pred_gate_beta": self.pred_gate_beta,
            "ema_gamma": self.ema_gamma,
            "liquid_tau_strength": self.liquid_tau_strength,
            "alpha_mean": float(alpha.mean().item()),
            "alpha_std": float(alpha.std().item()),
        }
        return out, h, aux


__all__ = ["PredictabilityGatedLiquidTauCfCCell"]
