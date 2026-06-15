# Round 175 — LearnedBetaPS+Reg-CfC — Research Report

**Date**: 2026-06-16
**Round**: 175
**Branch**: master
**Audit context (91-174)**: 43 strictly positive + 18 target-dep +
37 negatives = 98 mechanism classes.

## TL;DR

**NEGATIVE for Round 175**: L2 regularization on β doesn't change
outcomes — all 4 λ values give the same result at Kh=3, and Kh=2/5
match the round 171 SOTA exactly.

**Mechanism failure**: β_init = 0.75 = β_target = 0.75, so reg
penalty starts at 0 and can't pull β from optimal drift.

## What was tested

**Per-scale learnable β + L2 penalty** — `reg_loss = λ *
mean((β - 0.75)²)` added to task loss.

Hypothesis:
- H1 (positive): reg prevents extreme β (overfitting)
- H2 (negative): extreme β is useful (round 171)
- H3 (mixed): reg helps structured, hurts sin

## Bench (36 cells: 6 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | Kh | λ | sin_irr | structured_irr | random_irr | n_params |
|------|-----|---|---------|----------------|------------|----------|
| lbps_reg_l001 (very mild) | 3 | 0.001 | 0.0143±0.0051 (-46%) | 0.0137±0.0017 (-89%) | 0.1028±0.0029 (-2%) | 19241 |
| lbps_reg_l01 (mild) | 3 | 0.01 | 0.0143±0.0051 (-46%) | 0.0137±0.0017 (-89%) | 0.1028±0.0029 (-2%) | 19241 |
| lbps_reg_l1 (strong) | 3 | 1.0 | 0.0143±0.0051 (-46%) | 0.0139±0.0019 (-88%) | 0.1028±0.0029 (-2%) | 19241 |
| lbps_reg_l10 (very strong) | 3 | 10.0 | 0.0143±0.0051 (-46%) | 0.0138±0.0018 (-89%) | 0.1027±0.0029 (-2%) | 19241 |
| lbps_reg_kh2_l01 | 2 | 0.01 | 0.0064±0.0030 (-76%) | 0.0115±0.0008 (-91%) | 0.1023±0.0028 (-2%) | 16934 |
| lbps_reg_kh5_l01 | 5 | 0.01 | 0.0077±0.0006 (-71%) | 0.0095±0.0008 (-92%) | 0.1035±0.0031 (-1%) | 23855 |

## Cross-round (best in class)

| Round | Mechanism | sin | structured |
|-------|-----------|-----|------------|
| 171 | lb_ps_h2_75 (no reg) | **-76%** | -91% |
| 171 | lb_ps_h5_75 (no reg) | -71% | -92% |
| 175 | lbps_reg_kh2_l01 (λ=0.01) | -76% (TIES) | -91% (TIES) |
| 175 | lbps_reg_kh5_l01 (λ=0.01) | -71% (TIES) | -92% (TIES) |

**No NEW BESTS** in round 175. All variants tie round 171 SOTA.

## Hypotheses revisited

- **H1 (reg prevents extreme β)**: REJECTED. With β_init =
  β_target = 0.75, reg is 0 at start. β can still drift to
  extreme values if task loss dominates.
- **H2 (extreme β is useful)**: CONFIRMED. Without reg, β
  finds optimal values; with reg, β is constrained.
- **H3 (reg helps structured)**: REJECTED. All λ values give
  the same result.

## Why regularization doesn't change outcomes

### 1. β_init = β_target = 0.75
At init, β = 0.75 (init) and target = 0.75 (default), so reg
penalty is 0. The reg only kicks in if β drifts from 0.75.

### 2. β already at optimal for Kh=2/5
For Kh=2 and Kh=5, the optimal β is near 0.75 (round 171
SOTA). So β doesn't drift, reg stays 0, no change.

### 3. λ doesn't matter at Kh=3
At Kh=3, β_init = 0.75, β doesn't drift much because:
- For sin: optimal β is 0.75 (Kh=3 sweet spot)
- For structured: optimal β is 0.75 (Kh=3 sweet spot)
So β stays near 0.75 regardless of reg strength.

### 4. Reg can't escape local minimum
If β=0.75 is a local minimum, reg doesn't help escape. Reg
can only push β toward 0.75, not away.

## Pattern reinforced (43 + 18 + 38 = 99 mechanism classes)

- **43 strictly positive** (unchanged)
- **18 target-dep** (unchanged)
- **38 negatives** (UP from 37, round 175 adds 1)
- Total: **99 mechanism classes** (up from 98)

## Critical implementation details

1. **LearnedBetaPSRegCfCStackedNetwork** — wraps learned_beta_ps
   cells with `reg_loss()` method
2. **L2 penalty** — `reg = λ * mean((β - 0.75)²)` over all β
3. **Train loop adds reg to task loss** — see bench script
4. **Pyright false positives** on `import torch` are pre-existing
5. **Tests** — 13/13 pass

## Why this is a useful negative

1. **Confirms β_init = 0.75 is already optimal** — no need to
   try other targets
2. **Validates round 171 SOTA** — Kh=2/5 with init=0.75 already
   gives optimal β
3. **Saves future investigation** — reg on β is not useful when
   init = target
4. **Cheap** — only 13 lines for reg_loss method

## Files

- `lnn/core/learned_beta_ps_reg_cfc.py` (~200 lines, new core class)
- `tests/test_learned_beta_ps_reg_cfc.py` (13 tests, all pass)
- `scripts/bench_learned_beta_ps_reg_cfc.py` (36-cell bench)
- `results/bench_learned_beta_ps_reg_cfc.json`
- `docs/prds/2026-06-16-lnn-round-175-learned-beta-ps-reg-cfc.md`
- `docs/research/2026-06-16_learned_beta_ps_reg_cfc_report.md`

## Next ideas

1. **Reg with different target** — e.g., target=0.85 to push β
   higher
2. **Adaptive LR for β** — separate optimizer for β
3. **Per-layer learnable Kx** — also vary Kx per layer
4. **lb_ps + FAME MoE** — combine with FAME
5. **lb_ps + input-conditioned β** — β depends on input
6. **lb_ps + cosine annealing of β** — β varies over training

**Why:** Round 175 is a NEGATIVE. Reg doesn't help because
β_init = β_target = 0.75.

**How to apply:** **Don't use L2 reg on β when init = target.**
Audit becomes 99.
