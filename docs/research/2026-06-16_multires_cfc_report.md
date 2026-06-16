# Round 209 — Multi-Resolution Spectral Gating — Research Report

**Date**: 2026-06-16
**Round**: 209
**Branch**: master
**Audit context (91-208)**: 47 strictly positive + 26 target-dep
+ 58 negatives = 131 mechanism classes.

## TL;DR

**TARGET-DEPENDENT (27th) for Round 209**: Multi-resolution
spectral gating (2 scales) — sin -32.4% struct +19.5%
random -5.6% mean -6.2%.

## What was tested

**Multi-resolution spectral gating** (Sonnet 2026 style):
apply r200's spectral gating at TWO resolutions (full and
half FFT), combine with simple average.

## Bench (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean |
|------|---------|----------------|------------|------|
| cf | 0.0577 | 0.0041 | 0.0886 | 0.0501 |
| **multires (r209)** | **0.0390** | **0.0049** | **0.0836** | **0.0425** |

## Per-dataset analysis

### sin_irr — big win
- cf: 0.0552 / 0.0602 (mean 0.0577)
- r209: 0.0351 / 0.0428 (mean 0.0390, **-32.4%**)

### structured_irr — uneven
- cf: 0.0059 / 0.0023 (mean 0.0041)
- r209: 0.0089 / 0.0008 (mean 0.0049, **+19.5%**)
- High variance: seed 0 worse, seed 1 better

### random_irr — small win
- cf: 0.0949 / 0.0823 (mean 0.0886)
- r209: 0.0887 / 0.0784 (mean 0.0836, **-5.6%**)

## Pattern (47 + 26 + 58 = 131 → 47 + 27 + 58 = 132)

- 47 strictly positive (unchanged)
- **27 target-dep** (UP from 26, +1)
- 58 negatives (unchanged)
- Total: **132 mechanism classes**

## Comparison r200 vs r209

| Round | Mechanism | sin | struct | random | mean |
|-------|-----------|-----|--------|--------|------|
| 200 | spec single | -34.6% | 0% | +12.2% | -2.5% |
| **209** | **multires** | **-32.4%** | **+19.5%** | **-5.6%** | **-6.2%** |

Multi-resolution has **much better random** result.

## Why this is a useful TD

1. **First multi-resolution spectral mechanism in audit**
2. **Multi-scale structure helps sin-like data**
3. **Coarse scale interferes with multi-regime**

## Caveats

- 2 seeds, 30 epochs
- Struct has high variance
- Multi-scale is a meaningful new dimension

## Next ideas

1. **Three or more scales** — try 3-scale spectral
2. **Adaptive scale weighting** — learn per-scale weights
3. **Per-regime spectral** — different gating per regime
4. **Move away from spectral** — 7+ rounds on spectral done

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_multiresgated_cfc.py` (~250 lines)
- `tests/test_learned_beta_ps_ln_khlfft_multiresgated_cfc.py` (10 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_multiresgated_cfc.py` (12-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_multiresgated_cfc.json`

**Why:** Round 209 is **TARGET-DEPENDENT (27th)** — sin/random
improve, struct regresses unevenly.

**How to apply:** Use multi-resolution spectral gating for
sin-like or random data, NOT for structured/multi-regime.
