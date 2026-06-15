# PRD #10-149 — Round 187 — LearnedBetaPS+LN+Khl+FFT-CfC

**Date**: 2026-06-16
**Round**: 187
**Branch**: master
**Audit context (91-186)**: 45 strictly positive + 19 target-dep
+ 46 negatives = 110 mechanism classes.

## Background

Round 186 (FFT-CfC) was TARGET-DEPENDENT:
- **Helps sin** (best seed 0.0022 < SOTA 0.0033, 33% better)
- **Hurts structured** (0.0066-0.0075 vs SOTA 0.0024)

Round 180 (Kh ladder) is SOTA on both:
- `lbps_ln_khl_2_5_2`: sin 0.0033
- `lbps_ln_khl_5_3_2`: structured 0.0024

## Goal

**Combine FFT + Kh ladder** in a single architecture. The
hypothesis is that:
- FFT captures frequency-domain features (helps periodic
  data like sin)
- Kh ladder captures multi-time-scale features (handles
  regime changes in structured)

## Mechanism

**FFT input encoder** (from round 186) + **Kh ladder**
(from round 180):

```python
# Round 186:
x_clean = nan_to_num(x, nan=0)
x_fft = abs(rfft(x_clean, dim=1))
x_aug = cat([x, x_fft_pad], dim=-1)  # [B, T, 2D]
# Round 180: Kh ladder
h = stack_of_cells(x_aug, Kh_ladder=[2, 5, 2] or [5, 3, 2])
```

## Hypotheses

- **H1 (positive)**: FFT helps sin while Kh ladder handles
  structured → SP on both metrics
- **H2 (negative)**: FFT + Kh ladder don't compose (one
  dominates the other) → no improvement
- **H3 (mixed)**: helps one but hurts the other

## Configurations (2 conds)

1. `lbps_lnkhlfft_2_5_2`: Kh ladder [2,5,2] (sin-friendly)
2. `lbps_lnkhlfft_5_3_2`: Kh ladder [5,3,2] (structured-
   friendly)

Both with Kx=5, β=0.75, num_layers=3, FFT input features.

## Datasets

- sin_irr (D=2, T=32, missing_rate=0.3)
- structured_irr (D=2, T=32, missing_rate=0.3)
- random_irr (D=2, T=32, missing_rate=0.3)

## Bench

12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs.

## Success criteria

- **STRICTLY POSITIVE**: at least one cond beats SOTA on
  BOTH sin AND structured.
- **NEGATIVE**: no cond beats SOTA on either.
- **TARGET-DEP**: helps one, hurts the other.

## Why this direction is promising

1. **Round 186 found FFT helps sin per-seed** (0.0022 <
   SOTA 0.0033)
2. **Round 180 found Kh ladder handles structured**
   (0.0024 SOTA)
3. **Combined**: FFT + Kh ladder could be strictly
   positive on both

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_cfc.py` (~150 lines)
- `tests/test_learned_beta_ps_ln_khlfft_cfc.py` (12+ tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_cfc.py`
  (12-cell bench)
- `docs/research/2026-06-16_learned_beta_ps_ln_khlfft_cfc_report.md`

**Why:** Combine the two best mechanisms from rounds 180
(Kh ladder) and 186 (FFT) to potentially achieve SP on
both metrics.

**How to apply:** If SP, replace SOTA. If NEGATIVE, log
the negative. If TD, document dataset dependence.
