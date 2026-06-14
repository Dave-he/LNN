---
prd: 10-43
title: "Ecology-Gated φ-Balancing: auto-enable φ when E < 0.5"
date: 2026-06-14
status: draft
round: 84
authors: heyongxian
depends_on:
  - PRD #10-40 (φ-balancing, round 81)
  - PRD #10-42 (MoE ecology E diagnostic, round 83)
references:
  - arXiv:2605.06415  # E = T*H/(O+B) — paper that motivates the threshold
  - arXiv:2605.15403  # φ-Balancing (the intervention)
  - arXiv:2606.10703  # Causal Audit: observational ≠ causal
---

# PRD #10-43 — Ecology-Gated φ-Balancing: auto-enable φ when E < 0.5

## 0. One-liner

Close the loop on round 83's MoE ecology diagnostic: when the
live `E` drops below a configurable threshold, **automatically enable
φ-balancing** (round 81) on the affected cell.  This converts the
diagnostic from a passive monitor into an **autonomous cell-health
manager** that decides *when* to intervene, not just *whether* to.

## 1. Problem

Round 83 (PRD #10-42) gave us the **first theoretical diagnostic** for
our 5-layer LNN+MoE stack: `moe_ecology_diagnostic(B)` returns E, dead
count, per-expert utilization.  But it is **passive**: it tells you
the cell is unhealthy, but doesn't fix it.

Round 81 (PRD #10-40) gave us the **intervention**: `FAMECfC(phi_balance=True)`
adds an EMA-based mirror-descent bias to the router.  Smoke-bench:
K=3 top_k=1 task loss 0.125 vs 0.7595 baseline (-83.5%).

The two are **complementary but disconnected**: a user has to
(1) compute E from the cell, (2) decide it's too low, (3) manually
flip `phi_balance=True`.  We want a single **autonomous** component
that does this in real time.

## 2. Goal (Scope)

**Minimum-viable ecology-gated balancing for `FAMECfCCell`**:

- New flag `ecology_gated_balancing: bool = False` on `FAMECfCCell`.
  When `True`, the cell automatically enables `phi_balance` (and
  flips on a `PhiBalancer` if not already present) **at the first
  step where E drops below a configurable threshold**.
- New method `FAMECfCCell.set_ecology_threshold(E_min, B_ref)` —
  configures the threshold (default 0.5) and the reference B
  (default 0.001, the round 80 orth λ).
- New helper `EcologyGatedBalancer` (in `lnn/core/ecology_gated_balancing.py`)
  that wraps the gate logic: `step(E, B_active) → {"intervened": bool, "E": E, "B": B}`.
- New `scripts/bench_ecology_gated.py` — 3-cell × 3-dataset smoke
  bench: (a) baseline no-φ, (b) round 81 always-on φ (η=0.05),
  (c) **ecology-gated** φ (auto-enable when E < 0.5).
- **Back-compat**: zero changes when `ecology_gated_balancing=False`
  (the default).  All existing round 80-83 tests must still pass.

## 3. Out of Scope (Non-Goals)

- **Multi-cell coordination** (a network-level manager that toggles
  balancing across layers) — too complex for this round
- **Auto-disable when E recovers** — we only auto-enable; disable is manual
- **Adaptive η** (per-expert learning rate) — would need gradient H
- **Real MoE LLM reproduction** (1B+ params) — out of scope

## 4. Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `lnn/core/ecology_gated_balancing.py` exports `EcologyGatedBalancer` | TBD |
| 2 | `EcologyGatedBalancer.step(E, B)` returns intervention decision | TBD |
| 3 | `FAMECfCCell(ecology_gated_balancing=True)` auto-enables φ when E < 0.5 | TBD |
| 4 | `set_ecology_threshold(E_min, B_ref)` configures gate | TBD |
| 5 | 10+ unit tests in `tests/test_ecology_gated_balancing.py` | TBD |
| 6 | Smoke bench: 3 cells × 3 datasets (toy sin / random / structured) | TBD |
| 7 | Back-compat: `pytest tests/test_fame_cfc.py tests/test_orthogonality.py tests/test_moe_ecology.py tests/test_phi_balancing.py` all green | TBD |
| 8 | Honest-negative-friendly: report when gated φ LOSES to always-on φ | TBD |

## 5. Design

### 5.1 Module: `lnn/core/ecology_gated_balancing.py`

```python
class EcologyGatedBalancer:
    """Ecology-gated φ-balancing: auto-enable φ when E < threshold.
    
    The gate is **hysteresis-free** for simplicity: once enabled, stays
    enabled.  The reason is that turning φ off mid-training would
    re-collapse the routing, so we err on the side of "intervene
    early, stay intervened".
    
    Args:
        E_min: Threshold for intervention.  Default 0.5 (paper's value).
        warmup_steps: Don't intervene in the first N steps even if
            E < threshold (router needs time to settle).  Default 0
            (no warmup; gated by E alone).
    """
    def __init__(self, E_min: float = 0.5, warmup_steps: int = 0): ...
    
    @torch.no_grad()
    def step(self, E: float, B_active: float, step_idx: int) -> dict:
        """Update gate with current E and B.
        
        Returns dict with:
            - intervened: bool (True if φ-balancing should be active)
            - E: float (current E)
            - triggered_step: int (step at which intervention fired;
                -1 if not yet triggered)
        """
        ...
    
    def reset(self) -> None: ...
    @property
    def intervened(self) -> bool: ...
    @property
    def triggered_step(self) -> int: ...
```

### 5.2 Wire into `FAMECfCCell`

```python
def __init__(
    self, ..., phi_balance: bool = False, ema_alpha: float = 0.01,
    phi_step_size: float = 0.01, router_type: str = "learned",
    ecology_gated_balancing: bool = False,  # NEW
    ecology_E_min: float = 0.5,             # NEW
    ecology_warmup_steps: int = 0,          # NEW
):
    ...
    self.ecology_gated = (
        EcologyGatedBalancer(E_min=ecology_E_min, warmup_steps=ecology_warmup_steps)
        if ecology_gated_balancing else None
    )

def moe_ecology_diagnostic(self, B: float = 0.0, T: float = 1.0, O: float = 0.0) -> dict:
    # ... existing logic ...
    diag = {"E": ..., "dead_experts": ..., "utilization": ...}
    if self.ecology_gated is not None:
        gate_info = self.ecology_gated.step(diag["E"], B_active=B, step_idx=self.step_idx)
        diag["ecology_gate"] = gate_info
        # Auto-attach balancer if gate says intervene
        if gate_info["intervened"] and self.balancer is None and self.training:
            self.balancer = PhiBalancer(
                n_experts=self.n_experts,
                ema_alpha=self.ema_alpha,
                step_size=self.phi_step_size,
            )
            # Re-attach to router (forecastability)
            if hasattr(self.router, "set_balancer"):
                self.router.set_balancer(self.balancer)
    return diag
```

## 6. Files Touched

| File | Action | Lines |
|---|---|---|
| `lnn/core/ecology_gated_balancing.py` | **NEW** | ~80 |
| `lnn/core/fame_cfc.py` | MODIFY: add `ecology_gated_balancing` flag + gate wiring | ~+30 |
| `lnn/core/__init__.py` | MODIFY: export `EcologyGatedBalancer` | ~+1 |
| `tests/test_ecology_gated_balancing.py` | **NEW** | ~100 |
| `scripts/bench_ecology_gated.py` | **NEW** | ~200 |
| `docs/research/2026-06-14_ecology_gated_balancing_report.md` | **NEW** | ~150 |
| `docs/daily/2026-06-14_LNN_research_summary_v10.md` | **NEW** | ~50 |
| `README.md` | MODIFY: add Ecology-gated balancing section | ~+25 |

**Net**: 4 new + 3 modified = 7 files, ~635 lines (similar scope to round 83).

## 7. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Auto-attach balancer mid-training causes gradient issues | M | Use `PhiBalancer` (round 81) which is a no-grad bias; attaching after routing has run for some steps only affects the bias.  Tested in round 81. |
| E in toy data is "always high" (B≈0 → E≈1/eps) so gate never fires | H | We test on configs with high B (orth λ=1.0) so E drops below 0.5.  Bench explicitly constructs "E-drop" scenarios. |
| Gate fires too early in warmup, before routing settles | M | `warmup_steps` config; default 0 but user can set 50+ for early-step safety. |
| Back-compat: round 80-83 users see different behavior | L | New flag defaults to `False`; existing tests pass unchanged. |

## 8. Verification Plan

1. **Unit tests** (`tests/test_ecology_gated_balancing.py`):
   - `EcologyGatedBalancer` never fires when E > threshold
   - Fires exactly once when E first drops below threshold
   - Stays fired (hysteresis-free) after that
   - Respects `warmup_steps`
   - `FAMECfCCell(ecology_gated_balancing=True)` attaches balancer
     when E < threshold
   - Default `ecology_gated_balancing=False` is fully back-compat

2. **Smoke bench** (`scripts/bench_ecology_gated.py`):
   - 3 cells × 3 datasets × 3 conditions:
     - Cell A: `FAMECfCCell(phi_balance=False)` — baseline
     - Cell B: `FAMECfCCell(phi_balance=True, phi_step_size=0.05)` — always-on φ
     - Cell C: `FAMECfCCell(ecology_gated_balancing=True, ecology_E_min=0.5)` — gated
   - 3 datasets: toy sin, random, structured (same as round 83 B)
   - 3 seeds for variance
   - Force E < 0.5 by injecting orth λ=1.0 (paper's threshold region)
   - Report: final loss, intervened_step, E trajectory, dead_experts

3. **Regression**: full FAME+orth+φ+ecology test suite all green

## 9. Rollout

Single PR.  After landing:
- `lnn/core/ecology_gated_balancing.py` is the canonical reference impl
- `FAMECfCCell(ecology_gated_balancing=True)` is the one-liner API
- README adds "Ecology-gated balancing" section
- Back-compat: all existing code unchanged

## 10. Why this is the right next step (not other candidates)

- **Closes the loop on round 83**: passive diagnostic → active manager
- **Empirically testable**: 3×3×3 = 27 cells, runs in <5 min
- **Honest-negative-friendly**: if gated φ LOSES to always-on φ, we
  document it (consistent with rounds 73, 82, 83)
- **Scope**: similar to round 83, ~3-4h
- **Builds directly on round 80-83**: orth + φ + E + auto-gate = a
  complete "LNN+MoE autonomous health manager" stack

Other candidates rejected:
- **#10-44 Test E on real LLM training** — out of scope (no LLM data)
- **#10-45 Gradient-based H** — needs more research
- **#10-46 Test on vision** — out of scope (no vision data)
- **#10-7 LFM2.5-1.2B INT8** — deployment, needs full stack
- **Timeflies 2606.13571** — interesting but not directly relevant

## 11. Open Questions (to resolve in implementation)

- **Q1**: Should the gate have hysteresis (auto-disable when E recovers)?
  - **A1**: **No, no hysteresis for round 84**.  Once intervened,
    stay intervened.  Disabling mid-training would re-collapse the
    routing.  We err on the side of "intervene early, stay".
- **Q2**: Should the gate use the **empirical** E (cheap, default)
  or the **gradient-based** H (expensive, more accurate)?
  - **A2**: Empirical E (default).  Gradient-based H is a
    follow-up round (#10-45).
- **Q3**: What B to pass when computing E?
  - **A3**: The user's *active* balance weight — orth λ if using
    orth, φ η if using φ, 0 if neither.  The diagnostic accepts B
    as an arg, so the caller decides.  For the gate, we pass the
    round 80 default λ=0.001 unless the user overrides.
