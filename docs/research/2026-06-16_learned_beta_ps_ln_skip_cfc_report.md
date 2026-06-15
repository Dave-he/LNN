# Round 184 — LearnedBetaPS+LN+Skip-CfC — Research Report

**Date**: 2026-06-16
**Round**: 184
**Branch**: master
**Audit context (91-183)**: 45 strictly positive + 18 target-dep
+ 44 negatives = 107 mechanism classes.

## TL;DR

**NEGATIVE for Round 184**: Residual skip connection (replacing
CfC step with `h_t + Linear(LN(z))`) catastrophically regresses.
All 3 conditions are 100-300x worse than SOTA.

## What was tested

**lb_ps + LN + Skip** — replace CfC closed-form step with a
plain residual: `h_new = h_t + Linear(LN(z))`.

The motivation was:
- LN normalizes z (no scale issue) → residual adds
  well-conditioned correction
- h_t preserved by residual → gradient flow + magnitude

## Bench (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| lbps_lns_h3_75 | 1.1830±0.7361 ⚠️ | 0.8425±0.0263 ⚠️ | 0.6277±0.3815 ⚠️ | 26417 |
| lbps_lns_h2_75 | 0.3362±0.2347 ⚠️ | 0.5604±0.1925 ⚠️ | 0.1805±0.0006 | 23246 |
| lbps_lns_h5_75 | 0.7850±0.2088 ⚠️ | 0.7634±0.3148 ⚠️ | 0.2923±0.1045 | 32759 |

⚠️ = catastrophic regression

## Cross-round (best in class)

| Round | Mechanism | sin | structured |
|-------|-----------|-----|------------|
| 179 | lbps_ln_h2_75 (input LN) | **0.0035** | 0.0033 |
| 180 | lbps_ln_khl_2_5_2 | **0.0033** | 0.0058 |
| 180 | lbps_ln_khl_5_3_2 | 0.0198 | **0.0024** |
| **184** | **lbps_lns_h2_75 (residual skip)** | 0.3362 | 0.5604 |

**No NEW BESTS** — all 3 conditions are 100-300x worse than SOTA.

## Hypotheses revisited

- **H1 (positive)**: REJECTED. Residual + LN did not improve.
- **H2 (negative)**: CONFIRMED. Residual dilutes LN effect
  catastrophically.
- **H3 (mixed)**: REJECTED. Hurts both sin and structured.

## Why Residual Skip regresses catastrophically

### 1. Removed CfC step entirely
The original CfC step `h_new = τ·g + (1-τ)·h_branch` provides
a soft interpolation between g (new information) and h_branch
(recurrent update). The τ parameter has explicit time-scale
control.

By replacing this with `h_new = h_t + Linear(LN(z))`, we lose:
- Time-scale control
- Bounded output (h_branch is tanh-bounded)
- The non-linear interpolation between old and new

### 2. Residual projection can grow unboundedly
With `Linear(LN(z))` and no time scale, the residual magnitude
is unconstrained. After a few training steps, the projection
weights can grow large, causing the output to oscillate or
diverge.

### 3. h_t history dominates
Since h_t is added back via residual, it accumulates. Without
a mechanism to forget (like the CfC's (1-τ)·h_branch term), the
hidden state can grow without bound.

### 4. LayerNorm doesn't help here
LN normalizes the input z, but the output `h_t + Linear(LN(z))`
is not bounded — only the contribution from z is.

## Pattern (45 + 18 + 45 = 108 mechanism classes)

- **45 strictly positive** (unchanged)
- **18 target-dep** (unchanged)
- **45 negatives** (UP from 44, round 184 adds 1)
- Total: **108 mechanism classes**

## Critical implementation details

1. **Inherits from LearnedBetaPSLNCfCCell** — reuses LN-on-z
   + per-scale β.
2. **Residual proj** = `Linear(aug_total, hidden_size)`
3. **Residual init** = 0.1 scale on weight, 0 on bias (start
   like round 179).
4. **Removed CfC step entirely** — replaced with residual
5. **Tests** — 14/14 pass.

## Why this is a useful negative

1. **Identifies the risk of replacing CfC step** — the closed-
   form CfC interpolation is critical for stability.
2. **Documents that "residual" ≠ "improvement"** — adding a
   residual path without proper bounded dynamics destroys
   performance.
3. **Saves future exploration** — skip connection should be
   added INSIDE the CfC step (e.g., gated residual) not as a
   replacement.

## Next ideas (revised)

1. **lb_ps_ln + Gated residual** — gate the residual with a
   learnable scalar (init 0.1) inside the CfC step
2. **lb_ps_ln + Highway** — transform gate + carry gate
   (à la Highway networks)
3. **Different mechanism class** — try the next novel
   direction beyond lb_ps variants

## Files

- `lnn/core/learned_beta_ps_ln_skip_cfc.py` (~220 lines)
- `tests/test_learned_beta_ps_ln_skip_cfc.py` (14 tests)
- `scripts/bench_learned_beta_ps_ln_skip_cfc.py` (18-cell bench)
- `results/bench_learned_beta_ps_ln_skip_cfc.json`
- `docs/prds/2026-06-16-lnn-round-184-learned-beta-ps-ln-skip-cfc.md`
- `docs/research/2026-06-16_learned_beta_ps_ln_skip_cfc_report.md`

**Why:** Round 184 is NEGATIVE. Replacing CfC step with raw
residual destroys stability.

**How to apply:** Always keep the CfC closed-form step. If
adding residual, gate it or add it INSIDE the CfC step.
Audit becomes 108.
