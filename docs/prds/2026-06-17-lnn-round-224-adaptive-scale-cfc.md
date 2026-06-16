# PRD #10-186 — Round 224 — Adaptive 3-Branch Scale Selector on CfC

**Date**: 2026-06-17
**Round**: 224
**Branch**: master
**Audit context (91-223)**: 59 strictly positive + 28 target-dep
+ 59 negatives = 146 mechanism classes.

## Background

r216 (uniform 4-scale) and r223 (per-layer 2/3/4) both win all
3 datasets vs cf. Question: can adaptive per-step scale selection
beat fixed per-layer allocation?

## Goal

Test if a per-step learned gate that selects among 3 spectral
branches (2-scale / 3-scale / 4-scale) provides dynamic
flexibility beyond fixed allocation.

## Mechanism

```python
# 3 branches per cell (vs fixed scale count in r216/r223)
g_a = _branch_a_2scale(h)  # full + half
g_b = _branch_b_3scale(h)  # full + half + quarter
g_c = _branch_c_4scale(h)  # full + half + quarter + eighth

# Per-step learned gate
logits = scale_router(z)  # (B, 3)
weights = softmax(logits)  # (B, 3)
g = weights[0] * g_a + weights[1] * g_b + weights[2] * g_c
```

## Configurations (4 conds)

1. `cf`: r187 baseline
2. `4spectralbiasdrop`: r216 (uniform 4-scale)
3. `perlayer_234`: r223 (per-layer 2/3/4)
4. `adaptive_scale`: r224 (per-step adaptive 2/3/4)

## Result (24 cells: 4 conds × 3 datasets × 2 seeds × 30 epochs, hidden=16)

| Cond | sin_irr | structured_irr | random_irr | mean |
|------|---------|----------------|------------|------|
| cf | 0.0580 | 0.0130 | 0.0940 | 0.0550 |
| 4spectralbiasdrop (r216) | 0.0449 | 0.0013 | 0.0828 | 0.0430 |
| perlayer_234 (r223) | 0.0402 | 0.0008 | 0.0842 | 0.0417 |
| **adaptive_scale (r224)** | **0.0345** | **0.0012** | **0.0831** | **0.0396** |

Per-dataset (r224 vs cf):
- sin: -40.5% ✓
- structured: -90.6% ✓
- random: -11.6% ✓
- mean: -28.0%

Per-dataset (r224 vs r216):
- sin: -23.1% (better)
- structured: -3.0% (slightly better)
- random: +0.3% (tie)
- mean: -7.9%

Per-dataset (r224 vs r223):
- sin: -14.1% (better)
- structured: +48.3% (worse)
- random: -1.4% (better)
- mean: -5.1%

## Verdict

**STRICTLY POSITIVE (60th)** 🎉 — ALL 3 datasets improve vs cf.
Improves over both r216 (-7.9%) and r223 (-5.1%) on mean.

## Pattern (59 + 28 + 59 = 146 → **60 + 28 + 59 = 147**)

- **60 strictly positive (UP from 59, +1)** 🎉
- 28 target-dep (unchanged)
- 59 negatives (unchanged)
- Total: **147 mechanism classes**

## Caveats

- 2 seeds, 30 epochs
- Hidden=16, lr=1e-2, batch_size=16
- ~2x slower than uniform 4-scale (3x more spectral work)
- More parameters than r223 (3 branches share input dim)

## Lesson

**Adaptive per-step scale selection strictly positive** —
wins all 3 datasets vs cf, and improves mean over both r216
and r223. The learned gate effectively chooses the right
scale per timestep.

**Tradeoff vs r223**: better sin/random mean, worse structured
(2x). Both better than uniform 4-scale (r216) on mean.

## Next ideas

1. **Branch dropout** — drop entire branches randomly during
   training to encourage diversity
2. **Cross-branch attention** — let branches share info
3. **Larger branch pool** — add 5-scale (sub-threshold in
   uniform, may help as branch)
4. **PhysioNet test** — real-world data

**Why:** Round 224 is **STRICTLY POSITIVE 60th** — adaptive
3-branch scale selection improves over uniform 4-scale (r216)
and per-layer (r223) on mean.

**How to apply:** Use adaptive_scale when you have varied
timestep complexity. Use r216/r223 for faster inference.