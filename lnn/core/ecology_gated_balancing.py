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
