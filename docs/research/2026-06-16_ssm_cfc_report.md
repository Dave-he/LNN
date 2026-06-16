# Round 207 — Diagonal SSM on CfC — Research Report

**Date**: 2026-06-16
**Round**: 207
**Branch**: master
**Audit context (91-206)**: 47 strictly positive + 26 target-dep
+ 56 negatives = 129 mechanism classes.

## TL;DR

**NEGATIVE (57th) for Round 207**: Diagonal SSM added to
CfC DIVERGES. 4/6 cells NaN, 2/6 cells exploded to 1e18+.

## What was tested

**Diagonal state-space model (SSM)** as gating component.
Linear recurrence: `h_ssm_t = A * h_ssm + B * x_t`.

## Bench (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr |
|------|---------|----------------|------------|
| cf | 0.0502 | 0.0060 | 0.0889 |
| **ssm (r207)** | **NaN** | **5.4e22** | **2.3e18** |

## Per-dataset analysis

### sin_irr — NaN
- cf: 0.0469 / 0.0534 (mean 0.0502)
- r207: NaN / NaN

### structured_irr — exploded
- cf: 0.0042 / 0.0077 (mean 0.0060)
- r207: NaN / 5.4e22

### random_irr — exploded
- cf: 0.0948 / 0.0830 (mean 0.0889)
- r207: NaN / 2.3e18

## Pattern (47 + 26 + 56 = 129 → 47 + 26 + 57 = 130)

- 47 strictly positive (unchanged)
- 26 target-dep (unchanged)
- **57 negatives** (UP from 56, +1)
- Total: **130 mechanism classes**

## Why SSM diverges

1. **Diagonal SSM has no stability constraints** — A can grow
   to softplus(big) ≈ huge
2. **Recurrence h_ssm_t = A * h_ssm + B * x can explode**
3. **No normalization on h_ssm**
4. **C * h_ssm amplifies** whatever h_ssm becomes

## Why this is a useful NEG

1. **Confirms diagonal SSM needs explicit stability**
2. **CfC's `tanh(linear(z))` for h_branch is naturally bounded
   — SSM has no such bound**
3. **Suggests r208: SSM with A<1 constraint**

## Caveats

- 2 seeds, 30 epochs
- Hidden=12, lr=1e-2, batch_size=16
- 4/6 cells NaN, 2/6 cells exploded

## Next ideas

1. **Sigmoid A** — bound A to [0,1]
2. **Layer norm on h_ssm**
3. **HiPPO initialization**
4. **Gated SSM** — Mamba-style
5. **Move to different axis**

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_ssm_cfc.py` (~250 lines)
- `tests/test_learned_beta_ps_ln_khlfft_ssm_cfc.py` (11 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_ssm_cfc.py` (12-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_ssm_cfc.json`

**Why:** Round 207 is **NEGATIVE (57th)** — diagonal SSM
diverges without explicit stability constraints.

**How to apply:** Use sigmoid(A) [0,1] or layer norm on
h_ssm or HiPPO init for next attempt.
