# PRD #10-173 — Round 211 — 3-Scale Adaptive Spectral on CfC

**Date**: 2026-06-16
**Round**: 211
**Branch**: master
**Audit context (91-210)**: 48 strictly positive + 27 target-dep
+ 58 negatives = 133 mechanism classes.

## Background

r210 (3-scale spectral with simple average) was the 48th SP.
r211 extends r210 with **learned per-scale weights** via
softmax(linear(z)).

Hypothesis: adaptive weights should beat uniform average.

## Goal

Test if 3-scale adaptive spectral (learned weights) provides
better or different results than r210 (simple average).

## Mechanism

```python
# 3 scales (same as r210)
g1, g2, g3 = 3-scale computation

# ADAPTIVE WEIGHTS (NEW vs r210)
w = softmax(linear_weight(z))  # [B, 3]
g_combined = w[0]*g1 + w[1]*g2 + w[2]*g3

h_new = τ_eff * g_combined + (1-τ_eff) * h_branch
```

## Configurations (3 conds)

1. `cf`: r187 baseline
2. `3spectral`: r210 (simple average)
3. `3spectraladapt`: r211 (learned weights)

## Result (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs, hidden=16)

| Cond | sin_irr | structured_irr | random_irr | mean |
|------|---------|----------------|------------|------|
| cf | 0.0627 | 0.0066 | 0.0972 | 0.0555 |
| 3spectral (r210) | 0.0423 | 0.0020 | 0.0851 | 0.0431 |
| **3spectraladapt (r211)** | **0.0435** | **0.0025** | **0.0854** | **0.0438** |

Per-dataset (r211 vs cf):
- sin: -30.6% ✓
- structured: -62.1% ✓
- random: -12.1% ✓
- mean: -34.9%

## Verdict

**STRICTLY POSITIVE (49th)** 🎉 — ALL 3 datasets improve.

## Pattern (48 + 27 + 58 = 133 → **49 + 27 + 58 = 134**)

- **49 strictly positive** (UP from 48, +1) 🎉
- 27 target-dep (unchanged)
- 58 negatives (unchanged)
- Total: **134 mechanism classes**

## Why adaptive doesn't beat simple average

In same bench (r210 vs r211):
- sin: r210 -32.5% vs r211 -30.6% (slight loss)
- struct: r210 -69.7% vs r211 -62.1% (slight loss)
- random: r210 -12.4% vs r211 -12.1% (similar)

Adaptive weights don't significantly improve over uniform.

## Caveats

- 2 seeds, 30 epochs
- Hidden=16, lr=1e-2, batch_size=16
- Adaptive weights add params but no benefit in this regime

## Next ideas

1. **4-scale or 5-scale** — push scale count higher
2. **3-scale + spectral dropout** (combine with r203)
3. **Per-task adaptive weights** — different weights per dataset
4. **PhysioNet test** — real-world data

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_3spectraladapt_cfc.py` (~250 lines)
- `tests/test_learned_beta_ps_ln_khlfft_3spectraladapt_cfc.py` (10 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_3spectraladapt_cfc.py` (18-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_3spectraladapt_cfc.json`

**Why:** Round 211 is **STRICTLY POSITIVE (49th)** — 3-scale
adaptive spectral improves all 3 datasets.

**How to apply:** Use 3-scale spectral (simple average
suffices — learned weights don't help in 1D).
