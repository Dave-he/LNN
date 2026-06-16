# PRD #10-182 — Round 220 — 4-Scale Spectral + Bias + Dropout + MAX Combination

**Date**: 2026-06-16
**Round**: 220
**Branch**: master
**Audit context (91-219)**: 55 strictly positive + 28 target-dep
+ 59 negatives = 142 mechanism classes.

## Background

r216 (4-scale + bias + dropout, simple average) is the canonical
spectral variant. Natural question: does max combination
(winner-take-all per channel) help?

## Goal

Test if max combination (vs r216's simple average) improves
4-scale + bias + dropout spectral.

## Mechanism

```python
# Same as r216 except for combination
H1, H2, H3, H4 = 4-scale spectral gating
# For each scale: bias + mask + dropout (same as r216)

# NEW: MAX combination (vs avg in r216)
g_combined = max(g1, g2, g3, g4)  # element-wise across scales

h_new = τ_eff * g_combined + (1-τ_eff) * h_branch
```

## Configurations (3 conds)

1. `cf`: r187 baseline
2. `4spectralbiasdrop`: r216 (simple average)
3. `4spectralbiasdrop_max`: r220 (max combination)

## Result (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs, hidden=16)

| Cond | sin_irr | structured_irr | random_irr | mean |
|------|---------|----------------|------------|------|
| cf | 0.0583 | 0.0075 | 0.0904 | 0.0521 |
| 4spectralbiasdrop (r216) | 0.0435 | 0.0018 | 0.0831 | 0.0428 |
| **4spectralbiasdrop_max (r220)** | **0.0422** | **0.0023** | **0.0833** | **0.0426** |

Per-dataset (r220 vs cf):
- sin: -27.6% ✓
- structured: -69.3% ✓
- random: -7.9% ✓
- mean: -18.2%

## Verdict

**STRICTLY POSITIVE (56th)** 🎉 — ALL 3 datasets improve vs cf.

## Pattern (55 + 28 + 59 = 142 → **56 + 28 + 59 = 143**)

- **56 strictly positive (UP from 55, +1)** 🎉
- 28 target-dep (unchanged)
- 59 negatives (unchanged)
- Total: **143 mechanism classes**

## Caveats

- 2 seeds, 30 epochs
- Hidden=16, lr=1e-2, batch_size=16

## Lesson

**MAX and AVG are roughly equivalent** in this regime.
Both win 3/3 vs cf.

## Next ideas

1. **Learnable per-scale temperature** — different "softness" per scale
2. **Wavelet basis** — different frequency decomposition
3. **Phase-only spectral gating** — use phase not magnitude
4. **PhysioNet test** — real-world data

**Why:** Round 220 is **STRICTLY POSITIVE 56th** — max combination
improves all 3 datasets vs cf. Max and avg are roughly equivalent.

**How to apply:** Use r216 (avg) for smoother output, r220 (max)
for more discriminative. Both win 3/3 vs cf.
