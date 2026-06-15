# Round 202 — Learnable λ Convex Spectral Gating — Research Report

**Date**: 2026-06-16
**Round**: 202
**Branch**: master
**Audit context (91-201)**: 47 strictly positive + 23 target-dep
+ 54 negatives = 124 mechanism classes.

## TL;DR

**NEGATIVE (55th) for Round 202**: Learnable λ convex
combination `(1-λ) * g_branch + λ * spectral_g` does NOT
recover random performance — random still regresses +10.9%
(similar to r200 spec). Sin improvement -18.6% (between r200
-34.6% and r201 -22.8%). Mean +1.7%.

**Disproves H1 (λ learns to route cleanly)**, confirms H2
(λ doesn't learn meaningfully). r200 REPLACE-style remains
the spectral gating winner.

## What was tested

**Learnable λ convex combination** of linear g_branch and
spectral_g(h_t):

```python
λ(z) = sigmoid(linear(z))          # [B, H] in [0, 1]
g_combined = (1-λ) * g_branch(z) + λ * spectral_g(h_t)
h_new = tau_eff * g_combined + (1 - tau_eff) * h_branch
```

Hypothesis: λ learns to use spectral on periodic data,
linear on noisy data → SP on all 3 datasets.

## Bench (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean | Δ vs CF | type |
|------|---------|----------------|------------|------|---------|------|
| cf | 0.0381 | 0.0001 | 0.0834 | 0.0405 | — | — |
| lambda | 0.0310 | 0.0000 | 0.0925 | 0.0412 | +1.7% | **NEG** |

## Per-dataset analysis

### sin_irr — BETWEEN r200 and r201
- cf: 0.0398 / 0.0363 (mean 0.0381)
- lambda: 0.0355 / 0.0266 (mean 0.0310)
- **-18.6%** (r200 was -34.6%, r201 was -22.8%)
- λ didn't reach r200's full spectral dominance

### structured_irr — neutral
- cf: 0.0001 / 0.0001 (mean 0.0001)
- lambda: 0.0000 / 0.0000 (mean 0.0000)
- Already near-perfect for both

### random_irr — SAME LOSS as r200/r201
- cf: 0.0803 / 0.0866 (mean 0.0834)
- lambda: 0.0857 / 0.0994 (mean 0.0925, +10.9%)
- λ didn't recover random performance

## Pattern (47 + 23 + 54 = 124 → 47 + 23 + 55 = 125)

- 47 strictly positive (unchanged)
- 23 target-dep (unchanged)
- **55 negatives** (UP from 54, +1)
- Total: **125 mechanism classes**

## Why learnable λ failed

Hypothesis was H1 (λ learns to use spectral on periodic,
linear on noisy → SP). Actually got H2: **λ doesn't learn
to route cleanly**:

1. **Spectral signal has higher variance** than linear
   signal — gradient pushes λ toward spectral
2. **λ initialized to ~0.5** (sigmoid of zero bias) —
   balanced but not committed
3. **Random data has no clear signal** for λ to extract,
   so λ doesn't help

## Critical implementation details

1. **Per-feature λ** [B, H] — not scalar, different mixing
   per hidden dimension
2. **Initialized to ~0.5** — balanced starting point
3. **+294 params per cell** vs baseline = +882 total
4. **Convex combination** — g_combined ∈ convex hull of
   {g, g_spec}

## Comparison r200-r202: spectral gating variants

| Round | Mechanism | sin | struct | random | mean | Verdict |
|-------|-----------|-----|--------|--------|------|---------|
| 200 | spec (REPLACE) | **-34.6%** | 0% | +12.2% | -2.5% | **TD** |
| 201 | addspec (ADD) | -22.8% | 0% | +11.2% | +0.5% | **NEG** |
| 202 | lambda (CONVEX) | -18.6% | 0% | +10.9% | +1.7% | **NEG** |

REPLACE > ADD > CONVEX in terms of sin improvement.
None recover random performance.

## Why this is a useful NEG

1. **Disproves H1** — even with learnable λ, cannot recover
   random performance
2. **Confirms H2** — λ doesn't learn to route cleanly,
   spectral gradient dominates
3. **Confirms r200** — REPLACE-style spectral is best
4. **Suggests** that to recover random, must use different
   mechanism entirely

## Next ideas

1. **Dataset-conditioned spectral** — disable spectral when
   noise detected
2. **Multi-resolution wavelets** — Sonnet-style approach
3. **Different axis entirely** — attention, state-space, etc.
4. **Move on** — r200 remains spectral gating winner

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_lambda_specgated_cfc.py` (~210 lines)
- `tests/test_learned_beta_ps_ln_khlfft_lambda_specgated_cfc.py` (12 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_lambda_specgated_cfc.py` (12-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_lambda_specgated_cfc.json`

**Why:** Round 202 is **NEGATIVE (55th)** — learnable λ
convex combination does NOT recover random performance.
Sin improvement is between r200 (-34.6%) and r201 (-22.8%).

**How to apply:** Learnable λ gating is not a clear win.
r200's REPLACE-style spectral gating remains the winner.
