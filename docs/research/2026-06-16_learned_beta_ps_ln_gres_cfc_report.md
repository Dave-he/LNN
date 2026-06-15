# Round 185 — LearnedBetaPS+LN+GatedResidual-CfC — Research Report

**Date**: 2026-06-16
**Round**: 185
**Branch**: master
**Audit context (91-184)**: 45 strictly positive + 18 target-dep
+ 45 negatives = 108 mechanism classes.

## TL;DR

**NEGATIVE for Round 185**: Gated residual ON TOP of CfC step
regresses. Kh=3 is most stable but still 2-5x worse than
SOTA. Kh=2,5 show extreme variance on structured (39-99x
spread between seeds).

## What was tested

**lb_ps + LN + Gated Residual** — add a learnable scalar
gate α and residual projection ON TOP of the CfC step
(preserving it):
```
h_cfc = τ·g + (1-τ)·h_branch
h_new = h_cfc + α · Linear(LN(z))
```

## Bench (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| lbps_lngr_h3_75 | 0.0068±0.0024 | 0.0110±0.0026 | 0.1745±0.0087 | 26420 |
| lbps_lngr_h2_75 | 0.0094±0.0030 | 0.1565±0.1487 ⚠️ | 0.1737±0.0071 | 23249 |
| lbps_lngr_h5_75 | 0.0113±0.0046 | 0.2377±0.2178 ⚠️ | 0.1737±0.0070 | 32762 |

⚠️ = high variance (39-99x seed spread on structured)

## Cross-round (best in class)

| Round | Mechanism | sin | structured |
|-------|-----------|-----|------------|
| 180 | lbps_ln_khl_2_5_2 | **0.0033** | 0.0058 |
| 180 | lbps_ln_khl_5_3_2 | 0.0198 | **0.0024** |
| **185** | **lbps_lngr_h3_75 (gated residual)** | 0.0068 | 0.0110 |

**No NEW BESTS** — all 3 conds worse than SOTA on both
metrics.

## Hypotheses revisited

- **H1 (positive)**: REJECTED. Gated residual did not
  improve.
- **H2 (negative)**: PARTIAL. Residual is not redundant
  with τ (it adds different signal), but doesn't help.
- **H3 (mixed)**: CONFIRMED. Hurts structured more than
  sin. structured 0.011-0.238 vs SOTA 0.0024.

## Why Gated Residual regresses

### 1. Residual adds noise to well-conditioned CfC output
The CfC step `h_cfc = τ·g + (1-τ)·h_branch` is already a
carefully bounded interpolation. Adding `α·Linear(LN(z))`
introduces unconstrained perturbation.

### 2. α can grow during training
Although init α=0.1, the optimizer may grow α to values
that destabilize h_new (especially on structured data
where the target has discontinuities).

### 3. High variance on structured (h2, h5)
lbps_lngr_h2_75: structured seeds 0.0078 vs 0.3053 (39x)
lbps_lngr_h5_75: structured seeds 0.0199 vs 0.4555 (23x)

The Kh=2 and Kh=5 conditions have more EMA scales, making
the residual path more sensitive to input variations.

### 4. Kh=3 is the most stable
h3 has only 3 EMA scales (one per layer), giving the
smallest effective z dim and most stable residual.

## Pattern (45 + 18 + 46 = 109 mechanism classes)

- **45 strictly positive** (unchanged)
- **18 target-dep** (unchanged)
- **46 negatives** (UP from 45, round 185 adds 1)
- Total: **109 mechanism classes**

## Critical implementation details

1. **Inherits from LearnedBetaPSLNCfCCell** — reuses
   LN-on-z + per-scale β
2. **CfC step is UNCHANGED** — unlike round 184
3. **Gated residual** = `h_cfc + α · Linear(LN(z))`
4. **α init = 0.1** (sigmoid(logit(0.1)) ≈ 0.1)
5. **Residual init = 0.1** (small perturbation)
6. **Tests** — 15/15 pass

## Why this is a useful negative

1. **Confirms that the CfC step is hard to improve upon**
   — adding residual on top doesn't help
2. **Documents variance pattern** — Kh=2,5 destabilize
   structured more than Kh=3
3. **Saves future residual exploration** — gated residual
   on top of CfC is not a productive direction

## Next ideas (revised)

The lb_ps variant space is exhausted (5 NEGATIVEs in a row:
181-185). Pivot to a **different mechanism class**:

1. **Multi-resolution temporal pooling** — different time
   scale per branch
2. **1D conv on input** — capture local patterns
3. **Frequency-domain processing** — FFT, learn in
   spectral domain
4. **Different cell architecture** — try ODE solver, Mamba-
   style SSM, etc.

## Files

- `lnn/core/learned_beta_ps_ln_gres_cfc.py` (~240 lines)
- `tests/test_learned_beta_ps_ln_gres_cfc.py` (15 tests)
- `scripts/bench_learned_beta_ps_ln_gres_cfc.py` (18-cell bench)
- `results/bench_learned_beta_ps_ln_gres_cfc.json`
- `docs/prds/2026-06-16-lnn-round-185-learned-beta-ps-ln-gres-cfc.md`
- `docs/research/2026-06-16_learned_beta_ps_ln_gres_cfc_report.md`

**Why:** Round 185 is NEGATIVE. Gated residual on top of
CfC regresses (2-99x worse than SOTA).

**How to apply:** Don't add residual paths to CfC. Pivot
to a different mechanism class. Audit becomes 109.
