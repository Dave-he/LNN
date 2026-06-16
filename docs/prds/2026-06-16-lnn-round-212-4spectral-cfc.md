# PRD #10-174 — Round 212 — 4-Scale Spectral on CfC

**Date**: 2026-06-16
**Round**: 212
**Branch**: master
**Audit context (91-211)**: 49 strictly positive + 27 target-dep
+ 58 negatives = 134 mechanism classes.

## Background

r210 (3-scale spectral with simple average) and r211 (3-scale
adaptive) were consecutive SPs (48th and 49th). Both improved
all 3 datasets.

| Round | Mechanism | sin | struct | random | mean | Verdict |
|-------|-----------|-----|--------|--------|------|---------|
| r200 | 1-scale spectral | -17.5% | -61.9% | -5.3% | TD | TD 26th |
| r209 | 2-scale (full, half) | -32.4% | +19.5% | -5.6% | TD | TD 27th |
| r210 | 3-scale (full, half, quarter) | -32.5% | -69.7% | -12.4% | **SP 48th** | SP |
| r211 | 3-scale adaptive weights | -30.6% | -62.1% | -12.1% | **SP 49th** | SP |
| **r212** | **4-scale (adds eighth)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

## Goal

Test if pushing to 4 scales (adding an eighth-FFT scale for
ultra-coarse regime structure) provides additional benefit.

## Mechanism

```python
# 4 scales (vs r210's 3)
H1 = FFT(h_t)  # full
H2 = H1[:, :hidden_size//4+1]  # half
H3 = H1[:, :hidden_size//8+1]  # quarter
H4 = H1[:, :hidden_size//16+1]  # eighth (NEW)
g1, g2, g3, g4 = 4-scale computation
g_combined = (g1 + g2 + g3 + g4) / 4
h_new = τ_eff * g_combined + (1-τ_eff) * h_branch
```

## Configurations (3 conds)

1. `cf`: r187 baseline
2. `3spectral`: r210 (3-scale, simple average)
3. `4spectral`: r212 (4-scale, simple average)

## Result (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs, hidden=16)

| Cond | sin_irr | structured_irr | random_irr | mean |
|------|---------|----------------|------------|------|
| cf | 0.0715 | 0.0092 | 0.0989 | 0.0599 |
| 3spectral (r210) | 0.0356 | 0.0018 | 0.0860 | 0.0411 |
| **4spectral (r212)** | **0.0416** | **0.0027** | **0.0843** | **0.0429** |

Per-dataset (r212 vs cf):
- sin: -41.9% ✓
- structured: -70.3% ✓
- random: -14.8% ✓
- mean: -42.3%

## Verdict

**STRICTLY POSITIVE (50th)** 🎉 — ALL 3 datasets improve.

## Pattern (49 + 27 + 58 = 134 → **50 + 27 + 58 = 135**)

- **50 strictly positive** (UP from 49, **+1**) 🎉
- 27 target-dep (unchanged)
- 58 negatives (unchanged)
- Total: **135 mechanism classes**

## Caveats

- 2 seeds, 30 epochs
- Hidden=16, lr=1e-2, batch_size=16
- 4-scale ~16% slower than 3-scale
- 4-scale slightly worse on sin/struct vs 3-scale
  (but better on random)

## Caveats

- 2 seeds, 30 epochs
- Hidden=16, lr=1e-2, batch_size=16
- For hidden=16, scale 4 (eighth) only has 2 frequencies
  (borderline trivial)

## Next ideas (post-r212)

1. **3-scale + spectral dropout** — combine r210 (SP) with r203 (TD)
2. **PhysioNet test** — real-world data
3. **Per-task adaptive weights** — different weights per dataset
4. **4-scale adaptive weights** — combine r211 + r212

**Why:** Round 212 is **STRICTLY POSITIVE (50th)** — 4-scale
spectral improves all 3 datasets. 3 SPs in a row from the
spectral axis (r210, r211, r212).

**How to apply:** Use 3-scale spectral (best balance of
improvement and cost). 4-scale is also SP but slightly worse
on sin/struct.
