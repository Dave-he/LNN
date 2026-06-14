---
PRD: #10-51
date: 2026-06-15
round: 89
title: Causal-imbalance-gated Orth policy (turn round 88 diagnostic into policy)
builds-on: [round 88 (PRD #10-50, per-expert H_grad), round 85 (EcologyGatedOrth, PRD #10-44)]
status: draft
---

# PRD #10-51 — Causal-imbalance-gated Orth Policy

## 1. Background

Round 88 (PRD #10-50) introduced per-expert gradient magnitude as a
**causal imbalance diagnostic**.  Key finding: in our 9-cell bench,
`max_min_ratio_grad` (= max(per_expert_grad) / min(per_expert_grad)) is
**13-27× in 1-hot collapsed regimes** vs **2-3× in healthy regimes**.

This diagnostic is currently **passive** — it reports but doesn't
**act**.  Round 85 (PRD #10-44) introduced `EcologyGatedOrth`, which
**rescales orth λ to 0.001 when E < 0.5** (per-MoE-layer E, the
**observational** collapse signal).  Round 89 closes the loop:
add a **causality-gated orth policy** that fires the orth gate when
**per-expert causal imbalance is high** (`max_min_ratio_grad > 10`),
even when the observational E looks healthy.

## 2. Goals

1. **CausalityGatedOrth class**: gate that fires when
   `max_min_ratio_grad > causality_ratio_threshold`.
2. **FAMECfCCell(causality_gated_orth=True, causality_ratio_threshold=10.0)**
   opt-in flag.
3. **moe_ecology_diagnostic()** returns `causality_gate` state dict
   when enabled.
4. **Bench**: 2 conditions × 3 datasets × 3 λ — compare
   orth-only vs orth+causality.

## 3. Design

### 3.1 New `CausalityGatedOrth` class

```python
class CausalityGatedOrth:
    """Gate that fires when per-expert gradient imbalance is high.

    Round 89 (PRD #10-51).  Complements round 85 EcologyGatedOrth
    (observational E-based) by adding a causal imbalance signal.

    Args:
        ratio_threshold: max_min_ratio_grad above this fires the gate.
            Default 10.0 (1-hot collapsed regime).
        lambda_safe: λ to use when gate fires.  Default 0.001 (round 85).
        warmup_steps: Skip gate for first N steps.
    """
    def __init__(self, ratio_threshold=10.0, lambda_safe=0.001, warmup_steps=0):
        self.ratio_threshold = float(ratio_threshold)
        self.lambda_safe = float(lambda_safe)
        self.warmup_steps = int(warmup_steps)
        self.intervened = False
        self.last_ratio = 1.0
        self.last_lambda_scale = 1.0

    def step(self, max_min_ratio_grad, step_idx, user_lambda):
        """Decide whether to rescale orth λ.
        Returns (effective_lambda, state_dict)."""
        ...
```

### 3.2 Wire into FAMECfCCell

```python
FAMECfCCell(
    ...,
    causality_gated_orth: bool = False,
    causality_ratio_threshold: float = 10.0,
)
```

New methods:
- `compute_orth_loss_causality(outs, user_lambda)` — uses
  CausalityGatedOrth to rescale λ

### 3.3 Tests (target ≥ 10 unit tests)

- CausalityGatedOrth: fires when ratio > threshold
- Doesn't fire when ratio ≤ threshold
- Respects warmup_steps
- Resets properly
- Effective λ = user_lambda when not fired, lambda_safe when fired
- FAMECfCCell(causality_gated_orth=True) wires it
- Diagnostic includes causality_gate state when enabled
- Combined with round 85 EcologyGatedOrth (safe superset)
- back-compat: causality_gated_orth=False doesn't break

### 3.4 Bench

`scripts/bench_causality_gated_orth.py`:
- 2 conditions × 3 datasets × 3 orth λ ∈ {0.1, 1.0, 10.0}
- Conditions: orth-only (round 85) vs orth+causality (round 89)
- Per cell: loss, E_emp, max_min_ratio_grad, causality_fired,
  orth_fired, effective λ

Hypotheses:
- **H1 (complementary to E gate)**: causality gate fires in cells
  where E gate doesn't (per-expert imbalance > 10 even when E ≥ 0.5)
- **H2 (safe superset)**: orth+causality never worse than orth alone
- **H3 (early collapse detection)**: causality gate fires before
  E drops below 0.5 (causal collapse precedes observational collapse)

## 4. Honesty section

1. **Honest-positive expected**: per-expert gradient is a
   different signal from per-MoE E, so they're complementary
2. **Honest-negative possible**: in toy regime, gradient and
   utilization are highly correlated (same 1-hot collapse), so
   causality gate may fire on the same cells as E gate
3. **2/5-epoch quick bench** — longer training may show more
   regime divergence
4. **3 synthetic datasets** — vision/NLP may show different
   per-expert gradient dynamics
5. **Threshold 10× chosen from round 88 bench** — may need tuning
   for other regimes

## 5. Files

| File | Action | Lines |
|---|---|---|
| `lnn/core/ecology_gated_balancing.py` | MODIFY: add `CausalityGatedOrth` | +60 |
| `lnn/core/fame_cfc.py` | MODIFY: add `causality_gated_orth` flag, `causality_ratio_threshold`, `compute_orth_loss_causality` | +30 |
| `lnn/core/__init__.py` | MODIFY: export `CausalityGatedOrth` | +1 |
| `tests/test_causality_gated_orth.py` | **NEW** | 10+ tests |
| `scripts/bench_causality_gated_orth.py` | **NEW** | 200 lines |
| `docs/prds/2026-06-15-lnn-round-89-a-causality-gated-orth.md` | **NEW** | this file |
| `docs/research/2026-06-15_causality_gated_orth_report.md` | **NEW** | bench + analysis |
| `docs/daily/2026-06-15_LNN_research_summary_v15.md` | **NEW** | digest v15 |
| `README.md` | MODIFY: add Causality-Gated Orth section | +25 |

## 6. Acceptance criteria

- 10+ unit tests pass
- 174+ full regression suite green
- Bench completes 9 cells with both conditions
- Honest-positive or honest-negative clearly reported
- Push to master via 140.82.112.4
- Memory file `lnn-round-89-causality-gated-orth.md` written
