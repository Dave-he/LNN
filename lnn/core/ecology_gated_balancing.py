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
