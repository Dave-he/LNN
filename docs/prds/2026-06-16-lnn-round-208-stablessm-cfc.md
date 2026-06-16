# PRD #10-170 — Round 208 — Stable Diagonal SSM on CfC

**Date**: 2026-06-16
**Round**: 208
**Branch**: master
**Audit context (91-207)**: 47 strictly positive + 26 target-dep
+ 57 negatives = 130 mechanism classes.

## Background

r207's diagonal SSM diverged (NaN, 1e22). Fix with
**sigmoid(A) bounded to [0,1]** for guaranteed stability.

Hypothesis: bounded A prevents divergence while preserving
the linear SSM's representational power.

## Goal

Test if a stable diagonal SSM (sigmoid-A) added to CfC
improves over the r187 baseline.

## Mechanism

```python
A = sigmoid(linear_A(z))     # [B, H], in [0,1] (key fix)
B = linear_B(x_t)            # [B, H]
C = linear_C(z)              # [B, H]
h_ssm = A * h_ssm + B         # [B, H], stable recurrence
g_ssm = C * h_ssm             # [B, H]
g_combined = h_branch * g_ssm
h_new = τ_eff * g_combined + (1-τ_eff) * h_branch
```

## Configurations (2 conds)

1. `cf`: r187 baseline
2. `stablessm`: r208 (stable SSM)

## Result (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr |
|------|---------|----------------|------------|
| cf | 0.0466 | 0.0024 | 0.0917 |
| **stablessm (r208)** | **0.3859** | **0.0047** | **0.1635** |

Per-dataset (r208 vs cf):
- sin: 0.0466 → 0.3859 (**+728%**)
- struct: 0.0024 → 0.0047 (**+96%**)
- random: 0.0917 → 0.1635 (**+78%**)

## Verdict

**NEGATIVE (58th)** — stability fixed (no NaN) but all 3
datasets regress.

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
