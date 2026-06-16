# Round 208 — Stable Diagonal SSM — Research Report

**Date**: 2026-06-16
**Round**: 208
**Branch**: master
**Audit context (91-207)**: 47 strictly positive + 26 target-dep
+ 57 negatives = 130 mechanism classes.

## TL;DR

**NEGATIVE (58th) for Round 208**: Stable diagonal SSM
(sigmoid-A bounded [0,1]) **fixes r207's NaN but causes
regression on all 3 datasets**.

- sin: +728% (catastrophic)
- struct: +96%
- random: +78%

## What was tested

**Stable diagonal SSM** with sigmoid(A) bounded to [0,1].

## Bench (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr |
|------|---------|----------------|------------|
| cf | 0.0466 | 0.0024 | 0.0917 |
| **stablessm (r208)** | **0.3859** | **0.0047** | **0.1635** |

## Per-dataset analysis

### sin_irr — catastrophic
- cf: 0.0454 / 0.0478 (mean 0.0466)
- r208: 0.3902 / 0.3815 (mean 0.3859, **+728%**)

### structured_irr — regression
- cf: 0.0028 / 0.0019 (mean 0.0024)
- r208: 0.0029 / 0.0064 (mean 0.0047, **+96%**)

### random_irr — regression
- cf: 0.0907 / 0.0926 (mean 0.0917)
- r208: 0.2186 / 0.1083 (mean 0.1635, **+78%**)

## Pattern (47 + 26 + 57 = 130 → 47 + 26 + 58 = 131)

- 47 strictly positive (unchanged)
- 26 target-dep (unchanged)
- **58 negatives** (UP from 57, +1)
- Total: **131 mechanism classes**

## Why stable SSM still fails

1. **Bounded A limits expressiveness** — h_ssm decays too fast
2. **g_combined = h_branch * g_ssm** — when g_ssm is small,
   this zeros out h_branch
3. **CfC's tanh branch is already strong** — SSM doesn't add
   useful signal

## Why this is a useful NEG

1. **Confirms SSM is not a good addition to CfC's gating**
2. **Spectral gating (r200-r205) is the better axis**
3. **CfC's tanh branch is hard to improve via multiplicative
   gating**

## Caveats

- 2 seeds, 30 epochs
- Hidden=12, lr=1e-2, batch_size=16
- No NaN this time (stability confirmed)

## Next ideas

1. **Move away from SSM axis** — both r207 and r208 fail
2. **Spectral gating** (r200-r205) remains best axis
3. **Multi-scale FFT** — apply spectral at multiple resolutions
4. **Conv1D over hidden states** — classical CNN approach

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_stablessm_cfc.py` (~250 lines)
- `tests/test_learned_beta_ps_ln_khlfft_stablessm_cfc.py` (10 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_stablessm_cfc.py` (12-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_stablessm_cfc.json`

**Why:** Round 208 is **NEGATIVE (58th)** — stability fixed
but all 3 datasets regress.

**How to apply:** Do NOT add SSM to CfC. Spectral gating
remains the best axis.
