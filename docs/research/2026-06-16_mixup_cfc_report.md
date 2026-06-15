# Round 197 — Mixup Data Augmentation for CfC — Research Report

**Date**: 2026-06-16
**Round**: 197
**Branch**: master
**Audit context (91-196)**: 47 strictly positive + 21 target-dep
+ 51 negatives = 119 mechanism classes.

## TL;DR

**NEGATIVE for Round 197**: Mixup (Zhang et al 2018) is
**catastrophic** for CfC on time series prediction. Mean
+130-164% degradation. All 3 α values hurt all 3 datasets
(sin by 270-360%, random by 37-51%, structured near zero
both baseline and mixup). **Sample-level interpolation
doesn't apply to time series regression** — the data is
a continuous sequence, not a set of independent samples.

## What was tested

**Mixup** (Zhang et al 2018) is a sample-level augmentation:
```python
# Sample λ from Beta(α, α)
lam = sample_mixup_lambda(alpha, B, device)
# Permute batch
idx = torch.randperm(B)
# Mix input
x_mixed = lam * x + (1-lam) * x[idx]
# Forward
y = cfc_net(x_mixed)
# Mixup loss
loss = lam * MSE(y, t) + (1-lam) * MSE(y, t[idx])
```

This is **sample-level** (interpolates between two random
samples), as opposed to input/hidden noise (intra-sample
additive) or DropConnect (weight-level).

## Bench (24 cells: 4 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean | Δ vs MSE | type |
|------|---------|----------------|------------|------|----------|------|
| mse (α=0) | 0.0381 | 0.0001 | 0.0834 | 0.0405 | — | — |
| mixup_01 | 0.1750 | 0.0000 | 0.1263 | 0.1004 | +163.8% | **NEG** |
| mixup_02 | 0.1416 | 0.0000 | 0.1216 | 0.0877 | +130.5% | **NEG** |
| mixup_04 | 0.1693 | 0.0000 | 0.1146 | 0.0947 | +148.7% | **NEG** |

## Per-dataset analysis

### sin_irr
- mse: 0.0381
- mixup_01: 0.1750 (+359%)  ⚠️ CATASTROPHIC
- mixup_02: 0.1416 (+272%)  ⚠️ CATASTROPHIC
- mixup_04: 0.1693 (+344%)  ⚠️ CATASTROPHIC

**Mixup is destructive on sin** — interpolation creates
unrealistic phase-shifted curves the model can't fit.

### structured_irr
- mse: 0.0001 (near perfect, baseline)
- mixup_01-04: 0.0000-0.0001 (neutral)

**Neutral on structured** because the model already fits
the 2-regime pattern perfectly with no augmentation.

### random_irr
- mse: 0.0834
- mixup_01: 0.1263 (+51%)
- mixup_02: 0.1216 (+46%)
- mixup_04: 0.1146 (+37%)

**All mixup conds hurt random** by 37-51%.

## Pattern (47 + 21 + 51 = 119 → 47 + 21 + 52 = 120)

- 47 strictly positive (unchanged)
- 21 target-dep (unchanged)
- **52 negatives** (UP from 51, +1)
- Total: **120 mechanism classes**

## Why Mixup fails for time series

1. **Mixup is designed for classification** — interpolating
   between two images and their one-hot labels makes sense
   semantically.
2. **Time series prediction is point-wise regression** — each
   (t, x_t, y_t) is a sample. Interpolating between two random
   sequences creates UNREALISTIC inputs that don't exist in
   the data distribution.
3. **Mixup loss is contradictory for regression** —
   `lam * MSE(y, t) + (1-lam) * MSE(y, t[idx])` forces the
   model to match two different targets simultaneously,
   which is impossible when t and t[idx] are semantically
   different (different phases, different amplitudes).
4. **Interpolation smooths out discriminative features** —
   sin/cos curves mixed with other sin/cos curves at
   different phases produce garbage inputs the model
   can't fit.

## Why this is a useful negative

1. **Confirms task-appropriate augmentation matters** —
   Mixup works for images, fails for time series regression
2. **Closes the augmentation-aug loop** — after 5+ rounds
   of various augmentations, the right augmentation for
   CfC on these datasets is **input noise (round 192)**,
   not sample interpolation
3. **Confirms that sample-level interpolation doesn't apply
   here** — the data is a continuous sequence, not a set
   of independent samples

## Critical implementation details

1. **Beta distribution sampling** — `gamma1 / (gamma1 + gamma2)`
   where gamma1, gamma2 ~ Gamma(α, 1)
2. **Per-sample λ** — different λ for each sample in batch
3. **Permutation-based mixing** — `idx = randperm(B)`, mix
   x with `x[idx]`
4. **Tuple return for mixup loss** — forward returns
   `(y, idx, lam)` so loss can match both targets
5. **Same param count as round 187** — Mixup adds no params

## Comparison with r192-r196

| Round | Mechanism | Best sin | Best struct | Best random | Best mean | Verdict |
|-------|-----------|----------|-------------|-------------|-----------|---------|
| 192 | input noise | -16% | +6% | -26% | -24% | **SP** |
| 193 | hidden noise | -20% | -16% | +21% | +17% | TD |
| 194 | combined | +8% | -25% | +14% | +12% | TD |
| 196 | dropconnect | -14% (dc05) | +63% (dc05) | -3% (dc20) | 0% | **NEG** |
| 197 | mixup | +272% | 0% | +37% | +130% | **NEG** |

**Mixup is the worst augmentation** tested in the r192-197
audit. Pivot away from sample-level augmentation.

## Hypotheses revisited

- **H1 (positive, Mixup helps regression)**: REJECTED.
  Mean is +130-164% worse.
- **H2 (target-dep, helps structured)**: REJECTED.
  Structured is neutral because baseline is already perfect.
- **H3 (negative, hurt by phase interpolation)**: CONFIRMED.
  Sin catastrophic because of phase interpolation.

## Caveats

- **2 seeds only** — would benefit from 3-5 seeds
- **3 α values only** — could try α=0.05, 0.3
- **Tested on round 187 stack only** — may not generalize
- **Tested on 3 datasets only** — not PhysioNet/UEA/UCR

## Next ideas

1. **Cutmix for time series** — zero out contiguous
   time intervals, fill with other sample's intervals
2. **Channel mixup** — mix along feature dim only (not time)
3. **Time mixup** — same x and y, but mix along time dim
4. **Mixup at hidden state level** — not input
5. **Pivot to a different paradigm** — try a different
   mechanism class entirely (not augmentation)

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_mixup_cfc.py` (~155 lines)
- `tests/test_learned_beta_ps_ln_khlfft_mixup_cfc.py` (13 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_mixup_cfc.py` (24-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_mixup_cfc.json`

**Why:** Round 197 is **NEGATIVE** (Mixup +130-164% mean
degradation). Sample-level interpolation doesn't apply to
time series regression.

**How to apply:** Don't use Mixup for time series regression.
Use input noise (round 192) instead.
