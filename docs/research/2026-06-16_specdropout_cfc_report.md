# Round 203 — Spectral Dropout Regularization — Research Report

**Date**: 2026-06-16
**Round**: 203
**Branch**: master
**Audit context (91-202)**: 47 strictly positive + 23 target-dep
+ 55 negatives = 125 mechanism classes.

## TL;DR

**TARGET-DEPENDENT (24th) for Round 203**: Spectral dropout
(during training, dropout on the spectral mask) is the
**BEST spectral gating variant by mean improvement**:

- sin: -24.4% (strict win, slightly smaller than r200 spec -34.6%)
- struct: 0% (neutral)
- random: **+3.8%** (much smaller loss than r200 spec +12.2%)
- mean: **-5.0%** (better than r200 spec -2.5%)

The dropout regularization **dramatically reduces random loss**
while preserving most of the sin win. End of the 3-round
NEG streak after r200.

## What was tested

**Spectral dropout** on CfC. Apply r200's spectral gating
but with random frequency dropout during training:

```python
H = FFT(h_t)
magnitude = |H|
mask = sigmoid(linear(magnitude))
# NEW: dropout on the mask during training
if self.training and self.dropout_p > 0:
    mask = F.dropout(mask, p=self.dropout_p, training=True)
g = IFFT(H * mask)
```

Hypothesis: dropout on the mask acts as regularization,
prevents overfitting on noisy data (random) while preserving
periodic sensitivity (sin).

## Bench (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean | Δ vs CF | type |
|------|---------|----------------|------------|------|---------|------|
| cf | 0.0381 | 0.0001 | 0.0834 | 0.0405 | — | — |
| spec | 0.0249 | 0.0000 | 0.0936 | 0.0395 | -2.5% | **TD (r200)** |
| specdrop | 0.0288 | 0.0001 | 0.0866 | 0.0385 | **-5.0%** | **TD** |

## Per-dataset analysis

### sin_irr — strict win (slightly smaller than r200)
- cf: 0.0398 / 0.0363 (mean 0.0381)
- spec: 0.0269 / 0.0230 (mean 0.0249)
- specdrop: 0.0288 / 0.0289 (mean 0.0288)
- **-24.4%** vs r200's -34.6% (smaller but still strict win)

### structured_irr — neutral
- cf: 0.0001 / 0.0001 (mean 0.0001)
- spec: 0.0000 / 0.0000 (mean 0.0000)
- specdrop: 0.0000 / 0.0001 (mean 0.0001)
- Already near-perfect for all

### random_irr — DRAMATIC IMPROVEMENT over r200
- cf: 0.0803 / 0.0866 (mean 0.0834)
- spec: 0.0907 / 0.0965 (mean 0.0936, +12.2%)
- specdrop: 0.0865 / 0.0867 (mean 0.0866, +3.8%)
- **+3.8% loss vs r200's +12.2%** — much smaller

## Pattern (47 + 23 + 55 = 125 → 47 + 24 + 55 = 126)

- 47 strictly positive (unchanged)
- **24 target-dep** (UP from 23, +1)
- 55 negatives (unchanged)
- Total: **126 mechanism classes**

## Why spectral dropout helps

Hypothesis: spectral mask overfits without regularization.
Dropout in the spectral mask acts as regularization:

1. **Random data has uniform spectrum** — dropout makes
   spectral mask sparse, easier to ignore
2. **Sin data has concentrated spectrum** — mask learns
   dominant frequencies despite dropout
3. **The +3.8% random loss** (vs +12.2% for r200) shows
   dropout effectively reduces overfitting on noisy data

## Critical implementation details

1. **Dropout on the MASK** (not on the FFT output directly)
2. **Active in training mode only** — eval uses full mask
3. **dropout_p=0.3** — moderate dropout
4. **No other changes from r200** — same FFT, same masking

## Comparison r200-r203: spectral gating variants

| Round | Mechanism | sin | struct | random | mean | Verdict |
|-------|-----------|-----|--------|--------|------|---------|
| 200 | spec (REPLACE) | **-34.6%** | 0% | +12.2% | -2.5% | **TD** |
| 201 | addspec (ADD) | -22.8% | 0% | +11.2% | +0.5% | **NEG** |
| 202 | lambda (CONVEX) | -18.6% | 0% | +10.9% | +1.7% | **NEG** |
| 203 | specdrop (REG) | -24.4% | 0% | **+3.8%** | **-5.0%** | **TD** |

**r203 specdrop has the BEST mean** (-5.0% vs r200 -2.5%).

## Why this is a useful TD

1. **Best spectral variant by mean** — confirms dropout
   regularization helps spectral gating
2. **Random recovery is dramatic** — +12.2% → +3.8% loss
3. **Sin still wins** — strict per-dataset win
4. **Dropout p is a hyperparameter** — could tune for SP

## Caveats

- 2 seeds, 30 epochs
- Tested on r187 stack only
- Tested on 3 datasets only
- dropout_p=0.3 fixed (could try 0.1, 0.5)

## Next ideas

1. **Tune dropout_p** — try 0.1, 0.2, 0.4, 0.5
2. **Combine with r187 baseline g_branch** (replace specdrop
   for r200's g_branch path)
3. **Per-feature dropout** — different p per hidden dimension
4. **Dropout on FFT magnitude** (not mask) — different signal
5. **Try on PhysioNet** — irregular time series

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_specdropout_cfc.py` (~190 lines)
- `tests/test_learned_beta_ps_ln_khlfft_specdropout_cfc.py` (12 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_specdropout_cfc.py` (18-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_specdropout_cfc.json`

**Why:** Round 203 is **TARGET-DEPENDENT (24th)** with
best mean improvement among spectral variants. Dropout
regularization prevents overfitting on noisy data.

**How to apply:** Use spectral dropout for spectral gating
on CfC. Better generalization than plain spectral gating.
Tune dropout_p for application.
