# Round 204 — Spectral Dropout p=0.5 — Research Report

**Date**: 2026-06-16
**Round**: 204
**Branch**: master
**Audit context (91-203)**: 47 strictly positive + 24 target-dep
+ 55 negatives = 126 mechanism classes.

## TL;DR

**TARGET-DEPENDENT (25th) for Round 204**: Spectral dropout
p=0.5 (more aggressive than r203's p=0.3) is **worse than
r203 by mean** (-2.1% vs -5.0%).

- sin: -28.8% (slightly better than r203 -24.4%)
- struct: 0% (neutral)
- random: +10.2% (much worse than r203 +3.8%)
- mean: -2.1% (worse than r203 -5.0%)

**p=0.3 is the sweet spot.** p=0.5 is too aggressive,
hurting random data without giving enough extra sin benefit.

## What was tested

**Spectral dropout with higher p=0.5** (vs r203's p=0.3).
Same mechanism, just more aggressive dropout on the mask.

## Bench (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean | Δ vs CF | type |
|------|---------|----------------|------------|------|---------|------|
| cf | 0.0381 | 0.0001 | 0.0834 | 0.0405 | — | — |
| specdrop p=0.3 | 0.0288 | 0.0001 | 0.0866 | 0.0385 | -5.0% | **TD (r203)** |
| specdrop p=0.5 | 0.0271 | 0.0001 | 0.0919 | 0.0397 | -2.1% | **TD** |

## Per-dataset analysis

### sin_irr — slight improvement over r203
- cf: 0.0398 / 0.0363 (mean 0.0381)
- p=0.3: 0.0288 / 0.0289 (mean 0.0288)
- p=0.5: 0.0279 / 0.0263 (mean 0.0271)
- **-28.8%** vs r203's -24.4%

### structured_irr — neutral
- cf: 0.0001 / 0.0001 (mean 0.0001)
- p=0.3: 0.0000 / 0.0001 (mean 0.0001)
- p=0.5: 0.0000 / 0.0001 (mean 0.0001)

### random_irr — MUCH WORSE than r203
- cf: 0.0803 / 0.0866 (mean 0.0834)
- p=0.3: 0.0865 / 0.0867 (mean 0.0866, +3.8%)
- p=0.5: 0.0901 / 0.0937 (mean 0.0919, +10.2%)
- **+10.2%** vs r203's +3.8% — much worse

## Pattern (47 + 24 + 55 = 126 → 47 + 25 + 55 = 127)

- 47 strictly positive (unchanged)
- **25 target-dep** (UP from 24, +1)
- 55 negatives (unchanged)
- Total: **127 mechanism classes**

## Why p=0.5 is worse than p=0.3

Hypothesis: more dropout → better regularization.
Actually got: too much regularization hurts:

1. **Random**: p=0.5 zeros out half the spectral bins
   every step. Mask can't learn enough structure.
2. **Sin**: p=0.5 sometimes zeros out the dominant
   frequency
3. **Mean regression**: net effect is worse than p=0.3

## Critical implementation details

1. dropout_p=0.5 (vs r203's 0.3)
2. Same arch as r203

## Comparison r200-r204: spectral gating variants

| Round | Mechanism | sin | struct | random | mean | Verdict |
|-------|-----------|-----|--------|--------|------|---------|
| 200 | spec (REPLACE) | **-34.6%** | 0% | +12.2% | -2.5% | **TD** |
| 201 | addspec (ADD) | -22.8% | 0% | +11.2% | +0.5% | **NEG** |
| 202 | lambda (CONVEX) | -18.6% | 0% | +10.9% | +1.7% | **NEG** |
| 203 | specdrop p=0.3 | -24.4% | 0% | +3.8% | **-5.0%** | **TD** |
| 204 | specdrop p=0.5 | -28.8% | 0% | +10.2% | -2.1% | **TD** |

**r203 p=0.3 remains BEST** by mean improvement.

## Why this is a useful TD

1. Confirms p=0.3 is sweet spot — too much dropout hurts
2. Sin win slightly bigger with p=0.5 (smaller loss
   to dropout? No — random loss bigger)
3. Spectral gating has U-shape with dropout p:
   - p=0: too overfit
   - p=0.3: sweet spot
   - p=0.5: too aggressive

## Caveats

- 2 seeds, 30 epochs
- Tested on r187 stack only
- Tested on 3 datasets only
- Only 2 dropout values tested

## Next ideas

1. Tune dropout p — try 0.1, 0.2, 0.4
2. Per-feature dropout
3. Adaptive dropout (schedule)
4. Move to different axis (5 rounds on spectral done)

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_specdropout_high_cfc.py` (~165 lines)
- `tests/test_learned_beta_ps_ln_khlfft_specdropout_high_cfc.py` (12 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_specdropout_high_cfc.py` (18-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_specdropout_high_cfc.json`

**Why:** Round 204 is **TARGET-DEPENDENT (25th)** — p=0.5
is too aggressive. Mean improvement smaller than r203.

**How to apply:** Use p=0.3 (r203) not p=0.5. p=0.3 is the
sweet spot for spectral dropout on CfC.
