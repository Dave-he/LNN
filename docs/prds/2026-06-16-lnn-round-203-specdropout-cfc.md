# PRD #10-165 — Round 203 — Spectral Dropout Regularization

**Date**: 2026-06-16
**Round**: 203
**Branch**: master
**Audit context (91-202)**: 47 strictly positive + 23 target-dep
+ 55 negatives = 125 mechanism classes.

## Background

After 3 spectral gating variants (r200 spec REPLACE = TD,
r201 addspec ADD = NEG, r202 lambda CONVEX = NEG), round 203
adds **spectral dropout regularization** to address the
consistent failure on random data.

The hypothesis: spectral gating overfits because the mask
has too many parameters and no regularization. Adding dropout
in the frequency domain should:
1. Prevent the mask from overfitting to noise (helps random)
2. Encourage the mask to learn robust frequency structure
3. Act as a regularizer (may give SP)

## Goal

Test if dropout on the spectral mask during training gives
better generalization than r200 plain spectral gating.

## Mechanism

```python
# r200 spec: g = IFFT(FFT(h_t) * sigmoid(linear(|FFT(h_t)|)))
# r203 specdrop: g = IFFT(FFT(h_t) * dropout(sigmoid(linear(|FFT(h_t)|)), p))
# where dropout is active in train mode, identity in eval mode.

mask = sigmoid(linear(|FFT(h_t)|))  # [B, n_freq]
if self.training and self.dropout_p > 0:
    mask = F.dropout(mask, p=self.dropout_p, training=True)
g = IFFT(FFT(h_t) * mask)
```

The dropout zeros out random frequency bins in the mask,
forcing the model to be robust to missing frequencies.

## Configurations (3 conds)

1. `cf`: r187 baseline (linear g_branch)
2. `spec`: r200 spectral gating (no dropout)
3. `specdrop`: r203 spectral gating + dropout p=0.3

## Result (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean | Δ vs CF | type |
|------|---------|----------------|------------|------|---------|------|
| cf | 0.0381 | 0.0001 | 0.0834 | 0.0405 | — | — |
| spec | 0.0249 | 0.0000 | 0.0936 | 0.0395 | -2.5% | **TD (r200)** |
| specdrop | 0.0288 | 0.0001 | 0.0866 | 0.0385 | **-5.0%** | **TD** |

Per-dataset (specdrop vs cf):
- sin_irr: 0.0381 → 0.0288 (**-24.4%**, smaller than r200 spec -34.6% but bigger than r201/r202)
- structured_irr: 0.0001 → 0.0001 (~0%, neutral)
- random_irr: 0.0834 → 0.0866 (**+3.8%**, MUCH SMALLER loss than r200 spec +12.2%)

## Verdict

**TARGET-DEPENDENT (24th)** — spectral dropout is the
**BEST spectral gating variant** in terms of mean improvement:

1. **Sin improvement**: -24.4% (strict win, slightly smaller than r200 spec -34.6%)
2. **Random improvement**: +3.8% loss (vs r200 spec +12.2%, r201 addspec +11.2%, r202 lambda +10.9%)
3. **Mean**: -5.0% (BETTER than r200 spec -2.5%)

The dropout regularization **dramatically reduces random loss**
(from +12.2% to +3.8%) at the cost of slightly smaller sin win.

## Pattern (47 + 23 + 55 = 125 → 47 + 24 + 55 = 126)

- 47 strictly positive (unchanged)
- **24 target-dep** (UP from 23, +1)
- 55 negatives (unchanged)
- Total: **126 mechanism classes**

## Why spectral dropout helps (vs r200 spec)

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
   — keeps the gradient signal intact
2. **Active in training mode only** — eval uses full mask
3. **dropout_p=0.3** — moderate dropout
4. **No other changes from r200** — same FFT, same masking
5. **Cell has self.training flag** — dropout active iff
   training

## Comparison r200-r203: spectral gating variants

| Round | Mechanism | sin | struct | random | mean | Verdict |
|-------|-----------|-----|--------|--------|------|---------|
| 200 | spec (REPLACE) | **-34.6%** | 0% | +12.2% | -2.5% | **TD** |
| 201 | addspec (ADD) | -22.8% | 0% | +11.2% | +0.5% | **NEG** |
| 202 | lambda (CONVEX) | -18.6% | 0% | +10.9% | +1.7% | **NEG** |
| 203 | specdrop (REG) | -24.4% | 0% | **+3.8%** | **-5.0%** | **TD** |

**r203 specdrop has the BEST mean** (-5.0% vs r200 -2.5%).
Best random recovery (only +3.8% loss).
Slightly smaller sin win than r200 (-24.4% vs -34.6%).

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
**best mean improvement** among spectral variants:
sin -24.4% (strict win), random +3.8% (much smaller
loss than r200 +12.2%), mean -5.0% (better than r200
-2.5%). Spectral dropout regularization prevents overfitting.

**How to apply:** Use spectral dropout for spectral gating
on CfC. Better generalization than plain spectral gating.
Tune dropout_p for application.
