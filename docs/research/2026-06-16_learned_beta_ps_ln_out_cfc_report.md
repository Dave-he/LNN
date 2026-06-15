# Round 183 — LearnedBetaPS+LNout-CfC — Research Report

**Date**: 2026-06-16
**Round**: 183
**Branch**: master
**Audit context (91-182)**: 45 strictly positive + 18 target-dep +
43 negatives = 106 mechanism classes.

## TL;DR

**NEGATIVE for Round 183**: Output LayerNorm (post-CfC) doesn't
beat Input LayerNorm (round 179). High variance, regression on
sin, ties on structured.

## What was tested

**lb_ps + Output LN** — apply LayerNorm to h_new (output of CfC
closed-form) instead of z (input to CfC linear projections).

## Bench (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | Kh | sin_irr | structured_irr | random_irr | n_params |
|------|-----|---------|----------------|------------|----------|
| lbps_lno_h3_75 | 3 | 0.0077±0.0037 | 0.1385±0.0760 ⚠️ | 0.1734±0.0076 | 19337 |
| lbps_lno_h2_75 | 2 | 0.0100±0.0003 | 0.1146±0.1119 ⚠️ | 0.1728±0.0074 | 17030 |
| lbps_lno_h5_75 | 5 | 0.0055±0.0017 | 0.0052±0.0033 | 0.1736±0.0074 | 23951 |

⚠️ = high variance / regression

## Cross-round (best in class)

| Round | Mechanism | sin | structured |
|-------|-----------|-----|------------|
| 179 | lbps_ln_h2_75 (input LN) | **0.0035** | 0.0033 |
| 180 | lbps_ln_khl_2_5_2 (input LN + Kh ladder) | **0.0033** | 0.0058 |
| 180 | lbps_ln_khl_5_3_2 | 0.0198 | **0.0024** |
| **183** | **lbps_lno_h5_75 (output LN)** | 0.0055 | 0.0052 |

**No NEW BESTS**.

## Hypotheses revisited

- **H1 (post-LN normalizes h_new magnitude)**: PARTIAL. h_new
  IS normalized but magnitude info is lost.
- **H2 (post-LN differs from pre-LN, captures different signal)**:
  REJECTED. Post-LN captures LESS useful signal than pre-LN.
- **H3 (post-LN redundant with pre-LN)**: PARTIAL. Post-LN is
  worse, not redundant.

## Why Output LN regresses

### 1. Post-LN loses magnitude information
Pre-LN normalizes the INPUT features (different scales).
Post-LN normalizes the OUTPUT h_new, removing the magnitude
information needed for prediction.

### 2. Compound LN effect in stacked network
With pre-LN: each layer's z is normalized. h_new is NOT
normalized → magnitude preserved between layers.
With post-LN: each layer's h_new is normalized. Next layer's
z is computed from normalized h_t → may be redundant or
harmful.

### 3. High variance on structured
lbps_lno_h2_75 structured: 0.0027 vs 0.2265 (84x spread
between seeds). The LN normalization destabilizes training
for some seeds.

### 4. lno_h5_75 closest to SOTA but doesn't beat
Best individual result: 0.0019 (one seed of lbps_lno_h5_75)
— close to SOTA 0.0024 but not better on average.

## Pattern (45 + 18 + 44 = 107 mechanism classes)

- **45 strictly positive** (unchanged)
- **18 target-dep** (unchanged)
- **44 negatives** (UP from 43, round 183 adds 1)
- Total: **107 mechanism classes** (up from 106)

## Critical implementation details

1. **LayerNorm on h_new (post-CfC)** — applied AFTER closed-
   form CfC step.
2. **Same closed-form CfC** as round 171/179.
3. **Tests** — 12/12 pass.

## Why this is a useful negative

1. **Confirms Input LN is the right position** — Post-LN loses
   magnitude info needed for prediction.
2. **Identifies post-LN instability** — high variance on
   structured (84x spread).
3. **Saves future post-LN exploration** — no need to try other
   post-LN variants.

## Files

- `lnn/core/learned_beta_ps_ln_out_cfc.py` (~210 lines)
- `tests/test_learned_beta_ps_ln_out_cfc.py` (12 tests)
- `scripts/bench_learned_beta_ps_ln_out_cfc.py` (18-cell bench)
- `results/bench_learned_beta_ps_ln_out_cfc.json`
- `docs/prds/2026-06-16-lnn-round-183-learned-beta-ps-ln-out-cfc.md`
- `docs/research/2026-06-16_learned_beta_ps_ln_out_cfc_report.md`

## Next ideas

1. **lb_ps + RMSNorm** — different normalization
2. **lb_ps_ln + skip connections** — add residual to avoid
   vanishing gradients
3. **lb_ps_ln + Pre-LN (transformer-style)** — LN before each
   sublayer
4. **Different mechanism class entirely** — try next novel
   direction beyond lb_ps variants

**Why:** Round 183 is NEGATIVE. Output LN loses magnitude info
needed for prediction.

**How to apply:** Stick with Input LN (round 179/180). Audit
becomes 107.
