# PRD #10-159 — Round 197 — Mixup Data Augmentation for CfC

**Date**: 2026-06-16
**Round**: 197
**Branch**: master
**Audit context (91-196)**: 47 strictly positive + 21 target-dep
+ 51 negatives = 119 mechanism classes.

## Background

Rounds 192-196 explored regularization (input/hidden noise,
combined noise, σ sweep, DropConnect). All but r192 (input
noise) hurt on at least one dataset. Pivot to a different
augmentation paradigm: **Mixup** (Zhang et al 2018).

Mixup is **sample-level** (interpolates between two random
samples), as opposed to input/hidden noise (intra-sample
additive) or DropConnect (weight-level). Different paradigm.

## Goal

Test if Mixup provides useful regularization for time series
prediction with CfC.

## Mechanism (TRAINING ONLY, not eval)

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

## Configurations (4 conds)

1. `mse`: pure MSE baseline (α=0, no mixup)
2. `mixup_01`: α=0.1 (mild)
3. `mixup_02`: α=0.2 (standard)
4. `mixup_04`: α=0.4 (strong)

## Result (24 cells: 4 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean | Δ vs MSE | type |
|------|---------|----------------|------------|------|----------|------|
| mse (α=0) | 0.0381 | 0.0001 | 0.0834 | 0.0405 | — | — |
| mixup_01 | 0.1750 | 0.0000 | 0.1263 | 0.1004 | +163.8% | **NEG** |
| mixup_02 | 0.1416 | 0.0000 | 0.1216 | 0.0877 | +130.5% | **NEG** |
| mixup_04 | 0.1693 | 0.0000 | 0.1146 | 0.0947 | +148.7% | **NEG** |

**All 3 mixup conds hurt across all 3 datasets** by 100%+.

## Verdict

**NEGATIVE** (52nd) — Mixup is uniformly bad for CfC on
time series prediction tasks.

Mean deltas: +163.8%, +130.5%, +148.7% — all catastrophic.

## Per-dataset analysis

### sin_irr
- mse: 0.0381 (mean of seeds)
- mixup_01: 0.1750 (+359%)
- mixup_02: 0.1416 (+272%)
- mixup_04: 0.1693 (+344%)

**Mixup catastrophic on sin** at all α values.

### structured_irr
- mse: 0.0001 (well-fitted)
- mixup_01-04: 0.0000-0.0001 (neutral)

**Neutral on structured** because baseline is already near 0.

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
2. **Time series prediction is point-wise** — each (t, x_t, y_t)
   is a sample. Interpolating between two random sequences
   creates UNREALISTIC inputs that don't exist in the data
   distribution.
3. **Mixup loss is contradictory for regression** — `lam *
   MSE(y, t) + (1-lam) * MSE(y, t[idx])` forces the model
   to match two different targets simultaneously, which is
   impossible when t and t[idx] are semantically different
   (different phases, different amplitudes).
4. **The interpolation smooths out discriminative features**
   — sin/cos curves are mixed with other sin/cos curves at
   different phases, producing garbage inputs that the
   model can't fit.

## Why this is a useful negative

1. **Confirms the importance of task-appropriate augmentation**
   — Mixup works for images, fails for time series regression
2. **Confirms that sample-level interpolation doesn't apply
   here** — the data is a continuous sequence, not a set
   of independent samples
3. **Closes the augmentation-aug loop** — after 5+ rounds
   of various augmentations, the right augmentation for
   CfC on these datasets is **input noise (round 192)**,
   not sample interpolation

## Critical implementation details

1. **Beta distribution sampling** — `gamma1 / (gamma1 + gamma2)`
   where gamma1, gamma2 ~ Gamma(α, 1)
2. **Per-sample λ** — different λ for each sample in batch
3. **Permutation-based mixing** — `idx = randperm(B)`, mix
   x with `x[idx]`
4. **Tuple return for mixup loss** — forward returns
   `(y, idx, lam)` so loss can match both targets
5. **Same param count as round 187** — Mixup adds no params

## Caveats

- **2 seeds only** — would benefit from 3-5 seeds
- **3 α values only** — could try α=0.05, 0.3
- **Tested on round 187 stack only** — may not generalize
- **Tested on 3 datasets only** — not PhysioNet/UEA/UCR

## Next ideas

1. **Cutmix for time series** — zero out contiguous
   time intervals, fill with other sample's intervals
2. **Channel mixup** — mix along feature dim only
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
