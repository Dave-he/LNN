# PRD #10-169 — Round 207 — Diagonal SSM on CfC

**Date**: 2026-06-16
**Round**: 207
**Branch**: master
**Audit context (91-206)**: 47 strictly positive + 26 target-dep
+ 56 negatives = 129 mechanism classes.

## Background

After 6 rounds on spectral (r200-r205) and 1 on attention (r206),
pivot to a third axis: state-space models (Mamba/S4 style).

Hypothesis: linear state-space recurrence provides a
complementary gating signal to CfC's nonlinear branch.

## Goal

Test if a diagonal SSM component (linear recurrence) added
to CfC improves over the r187 baseline.

## Mechanism

```python
A = softplus(linear_A(z))     # [B, H], positive decay
B = linear_B(x_t)              # [B, H]
C = linear_C(z)                # [B, H]
h_ssm_t = A * h_ssm_{t-1} + B  # [B, H]
g_ssm = C * h_ssm_t            # [B, H]
g_combined = h_branch * g_ssm  # element-wise
h_new = τ_eff * g_combined + (1-τ_eff) * h_branch
```

## Configurations (2 conds)

1. `cf`: r187 baseline
2. `ssm`: r207 (diagonal SSM hybrid)

## Result (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr |
|------|---------|----------------|------------|
| cf | 0.0502 | 0.0060 | 0.0889 |
| **ssm (r207)** | **NaN** | **5.4e22** | **2.3e18** |

Per-dataset (r207):
- sin: NaN (both seeds)
- struct: NaN, 5.4e22 (seed 1 DIVERGED)
- random: NaN, 2.3e18 (seed 1 DIVERGED)

## Verdict

**NEGATIVE (57th)** — diagonal SSM diverges. 4/6 cells NaN,
2/6 cells exploded to 1e18+.

## Pattern (47 + 26 + 56 = 129 → 47 + 26 + 57 = 130)

- 47 strictly positive (unchanged)
- 26 target-dep (unchanged)
- **57 negatives** (UP from 56, +1)
- Total: **130 mechanism classes**

## Why SSM diverges

1. **Diagonal SSM has no stability constraints** — A can grow
   to softplus(big) ≈ huge
2. **Recurrence h_ssm_t = A * h_ssm + B * x can explode**
3. **No normalization on h_ssm** (unlike layer norm on z)
4. **C * h_ssm amplifies** whatever h_ssm becomes

## Why this is a useful NEG

1. **Confirms diagonal SSM needs explicit stability** (norm,
   constraint A<1, or HiPPO initialization)
2. **CfC's `tanh(linear(z))` for h_branch is naturally bounded
   (-1, 1) — SSM has no such bound**
3. **Suggests r208: SSM with A<1 constraint**

## Caveats

- 2 seeds, 30 epochs
- Hidden=12, lr=1e-2, batch_size=16
- 4/6 cells NaN, 2/6 cells exploded

## Next ideas

1. **Sigmoid A** — bound A to [0,1] for stability
2. **Layer norm on h_ssm** — normalize after each step
3. **HiPPO initialization** — proper SSM init
4. **Gated SSM** — Mamba-style selective gating
5. **Move to different axis entirely**

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_ssm_cfc.py` (~250 lines)
- `tests/test_learned_beta_ps_ln_khlfft_ssm_cfc.py` (11 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_ssm_cfc.py` (12-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_ssm_cfc.json`

**Why:** Round 207 is **NEGATIVE (57th)** — diagonal SSM
diverges. Softplus(A) can grow unbounded, h_ssm explodes.

**How to apply:** Do NOT use unbounded diagonal SSM.
