---
prd: 10-48
title: "Combined Ecology Gates: orth (strong) + φ (soft) co-active 2-axis policy"
date: 2026-06-15
status: draft
round: 86
authors: heyongxian
depends_on:
  - PRD #10-37 (orthogonality constraint, round 80)
  - PRD #10-40 (φ-balancing, round 81)
  - PRD #10-42 (MoE ecology E diagnostic, round 83)
  - PRD #10-43 (Ecology-Gated φ, round 84)
  - PRD #10-44 (Ecology-Gated Orth, round 85)
references:
  - arXiv:2605.06415  # E = T*H/(O+B) — paper that motivates the threshold
  - arXiv:2606.03631  # AnchorMoE orthogonality (aux loss we rescale)
  - arXiv:2605.15403  # φ-Balancing (complementary intervention)
  - arXiv:2606.10703  # Causal Audit: observational ≠ causal
---

# PRD #10-48 — Combined Ecology Gates: 2-axis adaptive policy

## 0. One-liner

Run **both** the round 84 φ gate (soft intervention on router) **and**
the round 85 orth gate (strong intervention on aux loss weight)
**co-actively** when E < 0.5.  This combines the soft and strong
interventions into a **2-axis adaptive policy** that picks the right
intervention strength per regime.  Smoke-bench on round 85's hardest
case (λ=1.0 toy_sin) to test whether the combined gate is **strictly
better** than either gate alone.

## 1. Problem

Rounds 84-85 gave us **two independent gates** that fire on the same
E < 0.5 condition:

| Gate | Round | Intervention | Strength |
|---|---|---|---|
| `EcologyGatedBalancer` (φ) | 84 | attach `PhiBalancer` to router | soft (router bias) |
| `EcologyGatedOrth` | 85 | rescale orth λ to 0.001 | strong (aux loss) |

Round 85 honest-negative insight: **at λ=1.0 ortho-toxicity, the soft
φ gate cannot recover because the orth loss gradient is 20× larger
than the φ bias can counteract.**  Round 85's strong orth gate
**completely fixes** this (λ=1.0 toy_sin 0.7302 → 0.6285, -14%).

**Open question**: does **combining both** gates give **strictly
better** results than either alone?  Hypotheses:

- **H1 (cumulative)**: combined ≥ both individually, because each
  gate attacks a different part of the failure mode (routing vs loss).
- **H2 (orth dominates)**: combined ≈ orth alone, because orth is
  the stronger intervention and routing recovery is automatic once
  the loss is fixed.
- **H3 (φ hurts when orth is already small)**: combined < orth alone,
  because the φ balancer adds bias when the routing is already
  recovering on its own.

We don't know which is true.  This round tests H1 vs H2 vs H3.

## 2. Goal (Scope)

**Minimum-viable 2-axis combined ecology gate**:

- New flag `ecology_combined: bool = False` on `FAMECfCCell`.  When
  `True`, the cell runs **both** `EcologyGatedBalancer` AND
  `EcologyGatedOrth` in parallel.
- New helper method `FAMECfCCell.set_ecology_combined_config(...)` —
  configures both gates' threshold, warmup, lambda_safe, eta.
- New `scripts/bench_combined_gates.py` — 4 conditions × 3 datasets ×
  3 orth λ ∈ {0.1, 1.0, 10.0} to show:
  - Condition A: baseline (no gate, user λ)
  - Condition B: φ gate only
  - Condition C: orth gate only (round 85)
  - Condition D: **combined** (this round)
- **Acceptance criteria**:
  - At λ=0.1 (healthy): D ≈ A (no false-positive intervention)
  - At λ=1.0 (toxic): D ≤ min(B, C) (combined ≤ best single)
  - At λ=10.0 (extreme): D ≈ C (orth dominates, φ adds no value)
  - All 4 conditions backed by explicit test cases
- **Honest-negative-friendly**: report if combined is WORSE than
  orth alone (catches the "φ adds noise" failure mode)

## 3. Out of Scope (Non-Goals)

- **Different thresholds per gate** — we use the same `E_min=0.5` for
  both gates (consistent with rounds 84-85)
- **Hysteresis** — neither gate has hysteresis; combined is also
  latch-only
- **Per-layer gate config** — one gate per cell, applied to all layers
- **Real MoE LLM reproduction** — out of scope
- **Auto-tuning E_min** — fixed at 0.5 (the paper's threshold)

## 4. Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `CombinedEcologyGate` class wires both round 84 + round 85 gates | TBD |
| 2 | `FAMECfCCell(ecology_combined=True)` attaches both gates | TBD |
| 3 | Diagnostic includes both `ecology_gate_balancer` and `ecology_gate_orth` keys | TBD |
| 4 | When E > threshold, neither gate fires (no false positives) | TBD |
| 5 | When E < threshold, both gates fire co-actively | TBD |
| 6 | 12+ unit tests in `tests/test_combined_gates.py` | TBD |
| 7 | Smoke bench: 4 conditions × 3 datasets × 3 λ | TBD |
| 8 | Back-compat: `pytest tests/test_fame_cfc.py` all green | TBD |
| 9 | Honest-negative: report if combined is WORSE than orth alone | TBD |

## 5. Design

### 5.1 Module addition: `CombinedEcologyGate` in `lnn/core/ecology_gated_balancing.py`

```python
class CombinedEcologyGate:
    """Combine φ gate (soft) and orth gate (strong) into a 2-axis policy.
    
    Both gates fire when E < E_min.  The φ gate attaches a PhiBalancer
    to the router (soft bias), and the orth gate rescales λ down to
    lambda_safe (strong aux loss intervention).
    
    Args:
        E_min: Shared threshold for both gates.  Default 0.5.
        lambda_safe: Target λ when orth gate fires.  Default 0.001.
        eta: φ bias step size when φ gate fires.  Default 0.05.
        warmup_steps: Shared warmup.  Default 0.
    """
    def __init__(self, E_min=0.5, lambda_safe=0.001, eta=0.05,
                 warmup_steps=0): ...
    
    def step(self, E, lambda_coeff, step_idx) -> dict:
        """Run both gates; return combined dict with:
            - phi_gate_info: dict (from EcologyGatedBalancer)
            - orth_gate_info: dict (from EcologyGatedOrth)
            - effective_lambda: float (from orth gate)
            - phi_enabled: bool (from φ gate)
        """
        ...
```

### 5.2 Wire into `FAMECfCCell`

```python
def __init__(
    self, ...,
    ecology_gated_balancing: bool = False,  # round 84
    ecology_gated_orth: bool = False,       # round 85
    ecology_combined: bool = False,         # NEW round 86
    ...
):
    ...
    if ecology_combined:
        from lnn.core.ecology_gated_balancing import CombinedEcologyGate
        self.combined_gate = CombinedEcologyGate(
            E_min=self.ecology_E_min,
            lambda_safe=ecology_orth_lambda_safe,
            eta=phi_eta,
            warmup_steps=self.ecology_warmup_steps,
        )
        # When combined is on, also turn on individual flags for
        # backward-compatible code paths.
        ecology_gated_balancing = True
        ecology_gated_orth = True
    else:
        self.combined_gate = None
```

### 5.3 Combined `compute_orth_loss()` and φ activation

The existing methods (`compute_orth_loss` and `phi_balance` flag)
should automatically use the combined gate's outputs.  No new API
needed beyond `ecology_combined=True`.

### 5.4 Diagnostic output

```python
def moe_ecology_diagnostic(self, B=0.0) -> dict:
    ...
    out = { ... existing ... }
    if self.combined_gate is not None:
        combined_info = self.combined_gate.step(
            E=float(E.item()), lambda_coeff=B, step_idx=self._step_idx,
        )
        out["ecology_gate_combined"] = combined_info
        out["ecology_gate_balancer"] = combined_info["phi_gate_info"]
        out["ecology_gate_orth"] = combined_info["orth_gate_info"]
    elif self.balancer_gate is not None:
        ...
    elif self.orth_gate is not None:
        ...
    return out
```

## 6. Files Touched

| File | Action | Lines |
|---|---|---|
| `lnn/core/ecology_gated_balancing.py` | MODIFY: add `CombinedEcologyGate` class | +100 |
| `lnn/core/fame_cfc.py` | MODIFY: add `ecology_combined` flag, wire in diagnostic | +30 |
| `lnn/core/__init__.py` | MODIFY: export `CombinedEcologyGate` | +1 |
| `tests/test_combined_gates.py` | **NEW** | ~150 |
| `scripts/bench_combined_gates.py` | **NEW** | ~250 |
| `docs/research/2026-06-15_combined_gates_report.md` | **NEW** | ~150 |
| `docs/daily/2026-06-15_LNN_research_summary_v12.md` | **NEW** | ~50 |
| `README.md` | MODIFY: add Combined gates section | +30 |

**Net**: 3 new + 3 modified = 6 files, ~760 lines.

## 7. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Combined gate is WORSE than orth alone (φ adds noise) | M | Bench exposes this case (H3) — report honestly |
| Combined gate fires too aggressively (E is noisy) | M | `warmup_steps` (default 0, configurable) |
| Conflicting intervention signals (φ biasing UP while orth rescales λ DOWN) | L | They target different things: φ biases the router, orth rescales the aux loss.  No conflict. |
| `ecology_combined=True` accidentally turns on individual flags | M | Document this clearly: combined is a *superset* of the individual flags.  Test in back-compat. |

## 8. Verification Plan

1. **Unit tests** (`tests/test_combined_gates.py`):
   - `CombinedEcologyGate` never fires when E > threshold
   - Both gates fire when E < threshold
   - Effective lambda = 0.001 (orth gate) when fired
   - `phi_enabled = True` (φ gate) when fired
   - Latched (no hysteresis)
   - Warmup respected
   - `FAMECfCCell(ecology_combined=True)` attaches both gates
   - Diagnostic includes both keys

2. **Smoke bench** (`scripts/bench_combined_gates.py`):
   - 4 conditions (baseline, φ only, orth only, combined) × 3 datasets × 3 λ
   - Report: final loss, gate firing order, E trajectory

3. **Regression**: all FAME+orth+φ+ecology+gate tests still pass

## 9. Rollout

Single PR.  After landing:
- `CombinedEcologyGate` is the canonical reference impl
- `FAMECfCCell(ecology_combined=True)` is the recommended one-line opt-in
- README adds "Combined ecology gates" section
- Back-compat: zero changes when `ecology_combined=False` (default)

## 10. Why this is the right next step

- **Natural next step flagged by round 85 verdict**: "Next round
  (round 86): combine both gates for a 2-axis intervention policy
  that picks the right intervention per regime."
- **Empirically testable**: 4 conditions × 3 datasets × 3 λ = 36 cells
- **Honest-negative-friendly**: explicitly tests H1 (combined best)
  vs H2 (orth dominates) vs H3 (combined worse)
- **Closes the LNN+MoE stack** at the policy layer: 5 defenses
  (76-81) + 1 diagnostic (83) + 3 policies (84 φ + 85 orth + 86 combined)
- **Directly testable on round 85's hard case**: λ=1.0 toy_sin

Other candidates rejected:
- **#10-45 Gradient-based H** — needs more design
- **#10-46 Test on vision** — out of scope (no vision data)
- **#10-47 Causal importance gate** — needs more design
- **#10-7 LFM2.5-1.2B INT8** — deployment, needs full stack
- **Timeflies 2606.13571** — interesting but not directly relevant

## 11. Open Questions

- **Q1**: Should `ecology_combined=True` turn on the individual flags
  automatically?
  - **A1**: Yes (cleaner UX).  The combined gate is a strict superset.
- **Q2**: What if the user wants a different threshold for each gate?
  - **A2**: Use the individual flags separately (round 84 + 85).  Combined
    uses a shared threshold.
- **Q3**: Is the combined gate strictly better than orth alone?
  - **A3**: Unknown.  Bench will tell.  If H2 holds (orth dominates),
    combined is redundant; if H1 holds, combined is better.
- **Q4**: Does φ add value when orth is already rescaled?
  - **A4**: The φ gate's only role after orth fires is to **prevent
    the routing distribution from collapsing** even at the small
    effective λ=0.001.  Test it.
