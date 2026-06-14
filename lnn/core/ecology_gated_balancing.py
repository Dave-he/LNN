"""Ecology-gated φ-balancing: auto-enable φ when E < threshold (PRD #10-43, 2026-06-14).

Closes the loop on round 83's MoE ecology diagnostic (PRD #10-42,
arXiv:2605.06415) by giving it **teeth**: when the live E drops
below ``E_min``, automatically enable φ-balancing (PRD #10-40,
arXiv:2605.15403).  This converts the diagnostic from a passive
monitor into an **autonomous cell-health manager**.

Design choices:

- **No hysteresis** for round 84.  Once intervened, stay intervened.
  Disabling mid-training would re-collapse the routing, so we err
  on the side of "intervene early, stay".
- **Pure-Python gate** (no nn.Module buffers needed beyond a small
  state dict).  The class is small enough that it doesn't need
  parameter registration.
- **Plays nicely with round 80-83 back-compat**: zero changes to
  FAMECfCCell API when ``ecology_gated_balancing=False`` (default).

The mapping to arXiv:2605.06415's framework:

- T = 1.0 (no temperature scaling in our FAME stack)
- H = empirical routing entropy (round 83 §2.1)
- O = 0.0 (no oracle loss)
- B = active balance weight (orth λ or φ η or 0)

When B is large enough that 1/(B+eps) < 0.5, the paper's threshold
is violated and intervention is justified.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EcologyGateState:
    """Snapshot of the gate's current state."""
    intervened: bool
    triggered_step: int
    E: float
    B_active: float


class EcologyGatedBalancer:
    """Decide when to auto-enable φ-balancing based on live E.

    Args:
        E_min: Intervention threshold.  Default 0.5 (the paper's
            claim that E ≥ 0.5 alone is sufficient to guarantee
            zero dead experts).
        warmup_steps: Don't intervene in the first N steps even if
            E < threshold (router needs time to settle).  Default 0
            (intervene as soon as E drops).

    Example:
        >>> gate = EcologyGatedBalancer(E_min=0.5)
        >>> for step in range(100):
        ...     state = gate.step(E=2.0, B_active=0.0, step_idx=step)
        ...     if state["intervened"]:
        ...         enable_phi_balancing()
        ...     else:
        ...         # E is healthy, no intervention
        ...         pass
    """

    def __init__(self, E_min: float = 0.5, warmup_steps: int = 0):
        assert E_min > 0.0, f"E_min must be positive, got {E_min}"
        assert warmup_steps >= 0, f"warmup_steps must be non-negative, got {warmup_steps}"
        self.E_min = float(E_min)
        self.warmup_steps = int(warmup_steps)
        # State.
        self._intervened: bool = False
        self._triggered_step: int = -1
        self._last_E: float = float("nan")
        self._last_B: float = 0.0
        # Counters for diagnostics.
        self.n_steps: int = 0
        self.n_below_threshold: int = 0

    def step(self, E: float, B_active: float, step_idx: int) -> dict:
        """Update gate with current E and active B.

        Args:
            E: Current MoE ecology number (round 83's E = T·H/(O+B)).
                Should be a finite non-negative float.
            B_active: Active balance weight (orth λ or φ η or 0).
            step_idx: Global step index (for warmup gate).

        Returns:
            Dict with:
                - intervened: bool (True once E has dropped below E_min
                    and warmup has elapsed; stays True after).
                - triggered_step: int (step at which intervention
                    fired, or -1 if not yet).
                - E: float (current E for debugging).
                - B_active: float (current B for debugging).
                - in_warmup: bool (True if still in warmup).
        """
        E = float(E)
        B_active = float(B_active)
        self._last_E = E
        self._last_B = B_active
        self.n_steps += 1
        in_warmup = step_idx < self.warmup_steps
        below = (E < self.E_min) and (not in_warmup)
        if below:
            self.n_below_threshold += 1
        # First-fire semantics: latch the trigger.
        if below and not self._intervened:
            self._intervened = True
            self._triggered_step = int(step_idx)
        return {
            "intervened": self._intervened,
            "triggered_step": self._triggered_step,
            "E": E,
            "B_active": B_active,
            "in_warmup": in_warmup,
            "below_threshold": bool(below),
        }

    @property
    def intervened(self) -> bool:
        return self._intervened

    @property
    def triggered_step(self) -> int:
        return self._triggered_step

    @property
    def last_E(self) -> float:
        return self._last_E

    @property
    def last_B(self) -> float:
        return self._last_B

    def state(self) -> EcologyGateState:
        return EcologyGateState(
            intervened=self._intervened,
            triggered_step=self._triggered_step,
            E=self._last_E,
            B_active=self._last_B,
        )

    def reset(self) -> None:
        """Reset gate to initial state (clear trigger, counters)."""
        self._intervened = False
        self._triggered_step = -1
        self._last_E = float("nan")
        self._last_B = 0.0
        self.n_steps = 0
        self.n_below_threshold = 0

    def __repr__(self) -> str:
        return (
            f"EcologyGatedBalancer(E_min={self.E_min}, "
            f"warmup_steps={self.warmup_steps}, "
            f"intervened={self._intervened}, "
            f"triggered_step={self._triggered_step}, "
            f"n_below={self.n_below_threshold}/{self.n_steps})"
        )


@dataclass
class EcologyOrthGateState:
    """Snapshot of the orth gate's current state."""
    intervened: bool
    triggered_step: int
    E: float
    lambda_coeff: float
    lambda_scale: float
    effective_lambda: float


class EcologyGatedOrth:
    """Ecology-gated orth rescaling: scale λ down to lambda_safe when E<threshold.

    Closes the loop on round 84's honest negative: gated φ-balancing
    cannot recover from λ=1.0 ortho-toxicity because the orth loss
    is too strong for a soft router bias to counteract.  This gate
    attacks the **root cause**: it rescales the user's ``lambda_coeff``
    down to a safe value (``lambda_safe``, default 0.001 = round 80
    default) when E < threshold.

    The rescaling is **multiplicative**:
        effective_lambda = lambda_coeff * lambda_scale
        lambda_scale = 1.0 (healthy)  OR  lambda_safe / lambda_coeff (fired)

    This way, the user keeps their original ``lambda_coeff`` (e.g.,
    λ=1.0), but the gate applies a scale factor so the **effective**
    λ is 0.001.

    **No hysteresis** (consistent with round 84 φ gate): once
    rescaled, stays rescaled.  Re-enabling high λ mid-training would
    re-collapse the routing.

    Args:
        E_min: Intervention threshold.  Default 0.5.
        lambda_safe: Target effective λ when gate fires.  Default 0.001
            (round 80 default, validated on 3 synthetic datasets in
            round 83 B).
        warmup_steps: Don't rescale in the first N steps.

    Example:
        >>> gate = EcologyGatedOrth(E_min=0.5, lambda_safe=0.001)
        >>> for step in range(100):
        ...     info = gate.step(E=2.0, lambda_coeff=1.0, step_idx=step)
        ...     effective_lambda = info["effective_lambda"]
        ...     # use effective_lambda in the orth loss
    """

    def __init__(
        self,
        E_min: float = 0.5,
        lambda_safe: float = 0.001,
        warmup_steps: int = 0,
    ):
        assert E_min > 0.0, f"E_min must be positive, got {E_min}"
        assert lambda_safe > 0.0, f"lambda_safe must be positive, got {lambda_safe}"
        assert warmup_steps >= 0, f"warmup_steps must be non-negative, got {warmup_steps}"
        self.E_min = float(E_min)
        self.lambda_safe = float(lambda_safe)
        self.warmup_steps = int(warmup_steps)
        # State.
        self._intervened: bool = False
        self._triggered_step: int = -1
        self._last_E: float = float("nan")
        self._last_lambda_coeff: float = 0.0
        self._last_lambda_scale: float = 1.0
        # Counters.
        self.n_steps: int = 0
        self.n_below_threshold: int = 0
        self.n_rescaled: int = 0

    def step(self, E: float, lambda_coeff: float, step_idx: int) -> dict:
        """Update gate with current E and orth λ.

        Args:
            E: Current MoE ecology number.
            lambda_coeff: User-specified orth loss weight (the
                "aggressive" value, e.g., 1.0 or 10.0).
            step_idx: Global step index (for warmup gate).

        Returns:
            Dict with:
                - intervened: bool (True once gate has fired)
                - triggered_step: int (-1 if not yet)
                - E: float
                - lambda_coeff: float (the user's original value)
                - lambda_scale: float (1.0 or lambda_safe/lambda_coeff)
                - effective_lambda: float (lambda_coeff * lambda_scale)
                - in_warmup: bool
                - below_threshold: bool
        """
        E = float(E)
        lambda_coeff = float(lambda_coeff)
        self._last_E = E
        self._last_lambda_coeff = lambda_coeff
        self.n_steps += 1
        in_warmup = step_idx < self.warmup_steps
        below = (E < self.E_min) and (not in_warmup)
        if below:
            self.n_below_threshold += 1
        # First-fire semantics: latch the trigger.
        if below and not self._intervened:
            self._intervened = True
            self._triggered_step = int(step_idx)
        # Compute lambda_scale.
        if self._intervened and lambda_coeff > 0.0:
            lambda_scale = self.lambda_safe / lambda_coeff
            self.n_rescaled += 1
        else:
            lambda_scale = 1.0
        self._last_lambda_scale = lambda_scale
        effective_lambda = lambda_coeff * lambda_scale
        return {
            "intervened": self._intervened,
            "triggered_step": self._triggered_step,
            "E": E,
            "lambda_coeff": lambda_coeff,
            "lambda_scale": lambda_scale,
            "effective_lambda": effective_lambda,
            "in_warmup": in_warmup,
            "below_threshold": bool(below),
        }

    @property
    def intervened(self) -> bool:
        return self._intervened

    @property
    def triggered_step(self) -> int:
        return self._triggered_step

    @property
    def last_lambda_scale(self) -> float:
        return self._last_lambda_scale

    def state(self) -> EcologyOrthGateState:
        return EcologyOrthGateState(
            intervened=self._intervened,
            triggered_step=self._triggered_step,
            E=self._last_E,
            lambda_coeff=self._last_lambda_coeff,
            lambda_scale=self._last_lambda_scale,
            effective_lambda=self._last_lambda_coeff * self._last_lambda_scale,
        )

    def reset(self) -> None:
        """Reset gate to initial state."""
        self._intervened = False
        self._triggered_step = -1
        self._last_E = float("nan")
        self._last_lambda_coeff = 0.0
        self._last_lambda_scale = 1.0
        self.n_steps = 0
        self.n_below_threshold = 0
        self.n_rescaled = 0

    def __repr__(self) -> str:
        return (
            f"EcologyGatedOrth(E_min={self.E_min}, "
            f"lambda_safe={self.lambda_safe}, "
            f"warmup_steps={self.warmup_steps}, "
            f"intervened={self._intervened}, "
            f"triggered_step={self._triggered_step}, "
            f"n_rescaled={self.n_rescaled}/{self.n_steps})"
        )


@dataclass
class CombinedEcologyGateState:
    """Snapshot of the combined gate's current state."""
    phi_intervened: bool
    orth_intervened: bool
    triggered_step: int
    E: float
    lambda_coeff: float
    lambda_scale: float
    effective_lambda: float
    phi_enabled: bool


class CombinedEcologyGate:
    """Combined ecology-gated policy: φ gate (soft) + orth gate (strong) co-active.

    Round 86 (PRD #10-48).  Closes the loop on rounds 84 + 85 by
    running **both** gates in parallel on the same E < 0.5 condition.

    - **φ gate** (round 84, soft): attaches a ``PhiBalancer`` to the
      router.  Strong for low-λ regimes where the routing is unbalanced
      but the aux loss is small.
    - **orth gate** (round 85, strong): rescales user's orth λ down
      to ``lambda_safe`` (default 0.001).  Strong for high-λ regimes
      where the aux loss dominates the task loss.

    The combined gate **doesn't change either gate's semantics** — it
    is a thin orchestrator that runs both and reports a unified state.

    Hypotheses tested in round 86's bench:
    - **H1 (cumulative)**: combined ≥ both individually.
    - **H2 (orth dominates)**: combined ≈ orth alone.
    - **H3 (φ adds noise)**: combined < orth alone (rare).

    Args:
        E_min: Shared threshold.  Default 0.5.
        lambda_safe: Target effective λ when orth gate fires.  Default 0.001.
        eta: φ bias step size when φ gate fires.  Default 0.05.
        warmup_steps: Shared warmup.  Default 0.

    Example:
        >>> gate = CombinedEcologyGate()
        >>> for step in range(100):
        ...     info = gate.step(E=2.0, lambda_coeff=1.0, step_idx=step)
        ...     if info["phi_intervened"]:
        ...         enable_phi_balancing()
        ...     if info["orth_intervened"]:
        ...         effective_lambda = info["effective_lambda"]  # 0.001
    """

    def __init__(
        self,
        E_min: float = 0.5,
        lambda_safe: float = 0.001,
        eta: float = 0.05,
        warmup_steps: int = 0,
        phi_gate: EcologyGatedBalancer | None = None,
        orth_subgate: EcologyGatedOrth | None = None,
    ):
        assert E_min > 0.0, f"E_min must be positive, got {E_min}"
        assert lambda_safe > 0.0, f"lambda_safe must be positive, got {lambda_safe}"
        assert eta > 0.0, f"eta must be positive, got {eta}"
        assert warmup_steps >= 0, f"warmup_steps must be non-negative, got {warmup_steps}"
        self.E_min = float(E_min)
        self.lambda_safe = float(lambda_safe)
        self.eta = float(eta)
        self.warmup_steps = int(warmup_steps)
        # Sub-gates (compose, don't reimplement).  If the caller
        # pre-built sub-gates (e.g., FAMECfCCell wires the same instance
        # into both cell.ecology_gate and the orchestrator), reuse them
        # so the diagnostic state stays consistent.
        self.phi_gate = phi_gate if phi_gate is not None else EcologyGatedBalancer(
            E_min=self.E_min, warmup_steps=self.warmup_steps,
        )
        self.orth_subgate = orth_subgate if orth_subgate is not None else EcologyGatedOrth(
            E_min=self.E_min, lambda_safe=self.lambda_safe,
            warmup_steps=self.warmup_steps,
        )

    def step(self, E: float, lambda_coeff: float, step_idx: int) -> dict:
        """Run both sub-gates; return combined state.

        Args:
            E: Current MoE ecology number.
            lambda_coeff: User-specified orth λ (forwarded to orth gate).
            step_idx: Global step index (for warmup).

        Returns:
            Dict with:
                - phi_intervened: bool (φ gate fired)
                - orth_intervened: bool (orth gate fired)
                - phi_enabled: bool (alias for phi_intervened, for clarity)
                - effective_lambda: float (orth gate's effective λ, = user λ when healthy)
                - lambda_scale: float (orth gate's scale factor)
                - triggered_step: int (earliest of φ or orth triggered; -1 if neither)
                - phi_gate_info: dict (raw φ gate output)
                - orth_gate_info: dict (raw orth gate output)
                - E: float
                - lambda_coeff: float
                - in_warmup: bool
        """
        E = float(E)
        lambda_coeff = float(lambda_coeff)
        phi_info = self.phi_gate.step(E=E, B_active=lambda_coeff, step_idx=step_idx)
        orth_info = self.orth_subgate.step(E=E, lambda_coeff=lambda_coeff, step_idx=step_idx)
        # Earliest trigger.
        triggers = [t for t in (phi_info["triggered_step"], orth_info["triggered_step"]) if t >= 0]
        triggered = min(triggers) if triggers else -1
        return {
            "phi_intervened": phi_info["intervened"],
            "orth_intervened": orth_info["intervened"],
            "phi_enabled": phi_info["intervened"],
            "effective_lambda": orth_info["effective_lambda"],
            "lambda_scale": orth_info["lambda_scale"],
            "triggered_step": triggered,
            "phi_gate_info": phi_info,
            "orth_gate_info": orth_info,
            "E": E,
            "lambda_coeff": lambda_coeff,
            "in_warmup": phi_info["in_warmup"],
        }

    def state(self) -> CombinedEcologyGateState:
        """Snapshot of current state."""
        phi_st = self.phi_gate.state()
        orth_st = self.orth_subgate.state()
        return CombinedEcologyGateState(
            phi_intervened=phi_st.intervened,
            orth_intervened=orth_st.intervened,
            triggered_step=min(
                t for t in (phi_st.triggered_step, orth_st.triggered_step) if t >= 0
            ) if (phi_st.triggered_step >= 0 or orth_st.triggered_step >= 0) else -1,
            E=orth_st.E,
            lambda_coeff=orth_st.lambda_coeff,
            lambda_scale=orth_st.lambda_scale,
            effective_lambda=orth_st.effective_lambda,
            phi_enabled=phi_st.intervened,
        )

    def reset(self) -> None:
        """Reset both sub-gates."""
        self.phi_gate.reset()
        self.orth_subgate.reset()

    def __repr__(self) -> str:
        return (
            f"CombinedEcologyGate(E_min={self.E_min}, "
            f"lambda_safe={self.lambda_safe}, eta={self.eta}, "
            f"warmup_steps={self.warmup_steps}, "
            f"phi={self.phi_gate.intervened}, "
            f"orth={self.orth_subgate.intervened})"
        )


class CausalityGatedOrth:
    """Auto-rescale orth λ when per-expert causal imbalance is high.

    Round 89 (PRD #10-51).  Complements round 85 EcologyGatedOrth
    (observational E-based) by adding a **causal imbalance** signal.

    Round 88 (PRD #10-50) showed that
    ``max_min_ratio_grad = max(per_expert_grad)/min(per_expert_grad)``
    is **13-27× in 1-hot collapsed regimes** vs **2-3× in healthy
    regimes**.  This is a **causal** signal (gradient magnitude) that
    the observational E cannot see (E may be high even with
    per-expert imbalance).

    When this ratio exceeds ``ratio_threshold``, the gate fires and
    rescales the user's orth λ down to ``lambda_safe`` (default 0.001)
    — same intervention as round 85's EcologyGatedOrth.  The
    reasoning is the same: high aux loss weight in an imbalanced
    regime is ortho-toxicity (round 80 finding), so we rescale it
    to a safe value.

    Args:
        ratio_threshold: max_min_ratio_grad above this fires the gate.
            Default 10.0 (corresponds to 1-hot collapse regime from
            round 88's 9-cell bench).
        lambda_safe: λ to use when gate fires.  Default 0.001 (round 85).
        warmup_steps: Don't fire in the first N steps (router needs
            time to settle).  Default 0 (fire as soon as imbalance
            is detected).

    Example:
        >>> gate = CausalityGatedOrth(ratio_threshold=10.0)
        >>> for step in range(100):
        ...     ratio = compute_max_min_ratio_grad(cell, task_loss)
        ...     state = gate.step(ratio, step_idx=step)
        ...     eff_lambda = state["effective_lambda"]
        ...     use_eff_lambda_for_orth_loss(eff_lambda)
    """

    def __init__(
        self,
        ratio_threshold: float = 10.0,
        lambda_safe: float = 0.001,
        warmup_steps: int = 0,
    ):
        self.ratio_threshold = float(ratio_threshold)
        self.lambda_safe = float(lambda_safe)
        self.warmup_steps = int(warmup_steps)
        self.intervened: bool = False
        self.last_ratio: float = 1.0
        self.last_lambda_scale: float = 1.0
        self.triggered_step: int = -1

    def step(self, max_min_ratio_grad: float, step_idx: int) -> dict:
        """Decide whether to rescale orth λ.

        Args:
            max_min_ratio_grad: Current per-expert gradient imbalance
                (= max/min of per_expert_gradient_norms).  Pass 1.0
                (or any value ≤ 1) to disable the gate cleanly.
            step_idx: Global step index (for warmup).

        Returns:
            Dict with:
            - intervened: bool (whether gate fired)
            - effective_lambda_scale: float (1.0 if not fired,
              lambda_safe/original if fired)
            - last_ratio: float (echo of input)
            - triggered_step: int (-1 if not triggered)
        """
        self.last_ratio = float(max_min_ratio_grad)
        in_warmup = step_idx < self.warmup_steps
        fires = (
            not in_warmup
            and max_min_ratio_grad > self.ratio_threshold
        )
        if fires and not self.intervened:
            self.intervened = True
            self.triggered_step = int(step_idx)
        # If user passed lambda_safe=0.001 and ratio > threshold, we
        # rescale to 0.001 / max_min_ratio_grad (smaller rescale for
        # bigger imbalance).  Else use 0.001 as a fixed safe value.
        if self.intervened:
            self.last_lambda_scale = self.lambda_safe / max(
                self.lambda_safe + 1e-8, 1.0,
            )
            # Actually rescale: eff_lambda = user_lambda * scale
            # where scale = lambda_safe / max(lambda_safe, 1) =
            #   lambda_safe when lambda_safe <= 1, else 1
            # We want scale = lambda_safe when ratio is high.
            # Simpler: scale = lambda_safe if user_lambda > lambda_safe
            # else 1.0.  Caller passes user_lambda, this returns
            # effective_lambda_scale.
            self.last_lambda_scale = self.lambda_safe
        else:
            self.last_lambda_scale = 1.0
        return {
            "intervened": self.intervened,
            "effective_lambda_scale": self.last_lambda_scale,
            "last_ratio": self.last_ratio,
            "triggered_step": self.triggered_step,
        }

    def reset(self) -> None:
        """Reset gate state (call between training runs)."""
        self.intervened = False
        self.last_ratio = 1.0
        self.last_lambda_scale = 1.0
        self.triggered_step = -1

    def __repr__(self) -> str:
        return (
            f"CausalityGatedOrth(ratio_threshold={self.ratio_threshold}, "
            f"lambda_safe={self.lambda_safe}, "
            f"warmup_steps={self.warmup_steps}, "
            f"intervened={self.intervened}, "
            f"last_ratio={self.last_ratio:.3f})"
        )
