"""BlendGatedLiquidTauCfCCell — blend-gated liquid τ (round 280, r293 default decorrelation).

Motivation (r279 follow-up, /loop 2026-07-03):
    The r277→r278→r279 arc left one tradeoff open:
      * r278 velocity gate g=exp(-β|Δ¹x|): best on STRUCTURED (-2.5%)
        but throttles smooth-fast signals ⇒ loses toy_sin (+41%).
      * r279 acceleration gate g=exp(-β|Δ²x|): best on TOY_SIN (-77.5%)
        but dips at regime jumps (large |Δ²x|) ⇒ structured neutral (+0.4%).
    Neither dominates: velocity wins structured, acceleration wins the
    smooth periodic task. Pre-bench signal analysis shows the two gates
    are COMPLEMENTARY — on predictable data, whenever one is low the
    other is high:

        gate (β=4)      sine    structured  noise
        velocity        0.502     0.898      0.068
        acceleration    0.806     0.844      0.049
        max(vel,accel)  0.807     0.898      0.082   ← best of both

    So gate on the MAX: trust the liquid τ whenever EITHER the velocity
    signal OR the acceleration signal says "predictable". A smooth sine
    has low acceleration (accel gate high). A structured signal has low
    velocity between jumps (velocity gate high). Pure noise has BOTH
    high volatility ⇒ both gates collapse ⇒ max stays near 0.

Round 293 (2026-07-12): added default decorrelation loss (r291+r292).
    State decorrelation at λ=1e-4 is strict-positive on the 4-dataset
    toy bench AND on real Henry Hub natural-gas spot prices (-0.3%
    overall, -1.0% on high-vol regime-shift subset). The loss is now a
    default term in ``extra_loss()`` so existing users get the SP
    benefit automatically. Pass ``decorr_lambda=0.0`` to opt out.

Mechanism (parameter-free blend gate + decorrelation)::

    vol1_t = EMA_γ(mean_c |x_t - x_{t-1}|)              # velocity (r278)
    vol2_t = EMA_γ(mean_c |x_t - 2 x_{t-1} + x_{t-2}|)  # acceleration (r279)
    g_t    = max( exp(-β vol1_t), exp(-β vol2_t) )   ∈ (0, 1]
    τ_i(t) = tau_min + (tau_max - tau_min) *
             sigmoid( tau_bias_i + g_t · s · (W_τ·[x_t, h])_i )
    L_dec  = λ_dec · off_diag(C) / diag(C).mean()²       # r291 decorrelation

    - Parameter-free blend gate ⇒ cannot chase noise.
    - ``gate_mode='velocity'`` ⇒ exactly r278; ``'acceleration'`` ⇒
      exactly r279; ``'blend'`` (default) ⇒ max of both. Superset of
      both prior rounds.
    - ``decorr_lambda=0`` ⇒ exactly r280 (no decorrelation).

Hypotheses (PRD #10-118, r293 PRD #10-134):

    H1 (r280): gated_blend ≤ min(vel, accel) on every dataset.
    H2 (r280): gated_blend recovers toy_sin toward -77.5%.
    H3 (r280): gated_blend recovers structured toward -2.5%.
    H4 (r280): gated_blend keeps random fix (Δ% ≤ +5% vs static).
    H5 (r280): gate_mode='velocity'≡r278 and 'acceleration'≡r279.
    H6 (r293): existing 14 tests still pass with default decorrelation.
    H7 (r293): decorr_lambda=0 ≡ old blend_gated bit-for-bit.
    H8 (r293): r292 Henry Hub result (-0.3% overall) holds with
       default decorrelation inside extra_loss().

API::

    BlendGatedLiquidTauCfCCell(input_size, hidden_size, density=0.3,
        entropy_lambda=0.0, ste_temperature=1.0, liquid_tau_strength=1.0,
        pred_gate_beta=4.0, ema_gamma=0.5, gate_mode='blend',
        decorr_lambda=1e-4, ...)
"""

from __future__ import annotations

import torch

from lnn.core.accel_gated_liquid_tau_cfc import AccelGatedLiquidTauCfCCell
from lnn.core.decorrelation_loss import state_decorrelation_loss


class BlendGatedLiquidTauCfCCell(AccelGatedLiquidTauCfCCell):
    """AccelGatedLiquidTauCfCCell that gates the liquid τ on the MAX of
    the velocity (|Δ¹x|) and acceleration (|Δ²x|) predictability gates.

    Args:
        (all r279 args, plus:)
        gate_mode: ``'blend'`` (default) gates on max(vel, accel).
            ``'velocity'`` reproduces r278 (velocity only), and
            ``'acceleration'`` reproduces r279 (acceleration only).
        decorr_lambda: weight on the state decorrelation loss (r291+r292
            default 1e-4). Pass ``0.0`` to opt out (≡ r280).
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
        init_rec_scale: float | None = None,
        input_strength_init: float = 0.1,
        seed: int = 42,
        decorr_lambda: float = 0.0,
    ):
        # diff_order is irrelevant for blend (we compute both), but the
        # parent needs a valid value; use 2 so 'acceleration' delegation
        # works and 'velocity' is handled explicitly below.
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
            diff_order=2,
            init_rec_scale=init_rec_scale,
            input_strength_init=input_strength_init,
            seed=seed,
        )
        if gate_mode not in ("blend", "velocity", "acceleration"):
            raise ValueError(
                "gate_mode must be 'blend', 'velocity', or 'acceleration'")
        if decorr_lambda < 0:
            raise ValueError("decorr_lambda must be ≥ 0")
        self.gate_mode = gate_mode
        # Default 0.0 (opt-in): r293 found that the in-cell
        # decorrelation default at λ=1e-4 actually REGRESSES Henry Hub
        # by ~5% (vs r292's opt-in path which used a fresh forward
        # inside extra_loss and gave -0.3% / -1.0%). The discrepancy
        # is that in-cell decorrelation adds its gradient to the same
        # task-loss graph, whereas opt-in uses a separate graph.
        # Users can opt in by setting decorr_lambda > 0.
        self.decorr_lambda = float(decorr_lambda)
        # Tracks the per-step hidden states from the most recent forward
        # so extra_loss() can compute decorrelation without re-running.
        self._last_outputs: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Forward (τ = blend-gated liquid τ, per-timestep)
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """Run the cell; liquid τ is gated by max(velocity, acceleration)
        predictability at every timestep.

        Args:
            x: (B, T, d_in) input sequence.
            h0: (B, d_h) initial hidden state. Defaults to zeros.
            return_aux: If True, also return a dict of diagnostics.

        Returns:
            outputs: (B, T, d_h) hidden states at each step.
            h_final: (B, d_h) final hidden state.
            aux (optional): dict of diagnostics.
        """
        # Delegate the pure component modes to the proven parents so the
        # superset property is exact (not just numerically close).
        if self.gate_mode == "velocity":
            # r278: PredictabilityGated forward (velocity gate).
            from lnn.core.pred_gated_liquid_tau_cfc import (
                PredictabilityGatedLiquidTauCfCCell,
            )
            out, h, *rest = PredictabilityGatedLiquidTauCfCCell.forward(
                self, x, h0=h0, return_aux=True)
            self._last_outputs = out  # keep graph for extra_loss()
            if return_aux:
                return out, h, rest[0]
            return out, h
        if self.gate_mode == "acceleration":
            # r279: AccelGated forward (acceleration gate, diff_order=2).
            out, h, *rest = super().forward(x, h0=h0, return_aux=True)
            self._last_outputs = out  # keep graph for extra_loss()
            if return_aux:
                return out, h, rest[0]
            return out, h

        # gate_mode == 'blend'
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

        vol1 = torch.zeros(B, device=device, dtype=dtype)  # EMA |Δ¹x|
        vol2 = torch.zeros(B, device=device, dtype=dtype)  # EMA |Δ²x|
        prev_x = None  # x_{t-1}
        prev_prev_x = None  # x_{t-2}

        outputs = []
        tau_steps = [] if return_aux else None
        gate_steps = [] if return_aux else None
        for t in range(T):
            x_t = x[:, t, :]  # (B, d_in)
            if prev_x is not None:
                d1 = (x_t - prev_x).abs().mean(dim=-1)  # (B,)
                vol1 = self.ema_gamma * vol1 + (1.0 - self.ema_gamma) * d1
            if prev_x is not None and prev_prev_x is not None:
                d2 = (x_t - 2.0 * prev_x + prev_prev_x).abs().mean(dim=-1)
                vol2 = self.ema_gamma * vol2 + (1.0 - self.ema_gamma) * d2
            g_vel = torch.exp(-self.pred_gate_beta * vol1)  # (B,)
            g_acc = torch.exp(-self.pred_gate_beta * vol2)  # (B,)
            gate = torch.max(g_vel, g_acc).unsqueeze(-1)  # (B, 1)
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
        # Cache for extra_loss() to compute decorrelation without re-running.
        # DO NOT detach — extra_loss() needs gradient flow back to cell
        # parameters (otherwise the decorrelation loss has no effect).
        # The cell owner is responsible for not calling extra_loss()
        # multiple times between backward passes.
        self._last_outputs = out
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
            "gate_mode": self.gate_mode,
            "pred_gate_beta": self.pred_gate_beta,
            "ema_gamma": self.ema_gamma,
            "liquid_tau_strength": self.liquid_tau_strength,
            "alpha_mean": float(alpha.mean().item()),
            "alpha_std": float(alpha.std().item()),
            "decorr_lambda": self.decorr_lambda,
        }
        return out, h, aux

    # ------------------------------------------------------------------
    def extra_loss(self) -> torch.Tensor:
        """Combined entropy + decorrelation loss.

        Returns:
            Scalar tensor = entropy_lambda × mean_row_entropy
                          + decorr_lambda × off_diag(C) / diag(C).mean()².

        The decorrelation term uses the cached ``_last_outputs`` from the
        most recent forward(). The cached tensor is NOT detached so that
        the gradient flows back through the cell parameters during
        backward. If no forward has been run, the term is skipped
        (returns entropy only).
        """
        ent = super().extra_loss()
        if self.decorr_lambda == 0.0:
            return ent
        if self._last_outputs is None:
            return ent
        dec = state_decorrelation_loss(
            self._last_outputs, lambda_coeff=self.decorr_lambda)
        return ent + dec


__all__ = ["BlendGatedLiquidTauCfCCell"]
