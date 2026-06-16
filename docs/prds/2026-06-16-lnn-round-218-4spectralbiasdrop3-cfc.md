# PRD #10-180 — Round 218 — 4-Scale Spectral + Bias + Dropout p=0.3

**Date**: 2026-06-16
**Round**: 218
**Branch**: master
**Audit context (91-217)**: 54 strictly positive + 27 target-dep
+ 59 negatives = 140 mechanism classes.

## Background

After r216 (4-scale + bias + dropout p=0.2) was the canonical
spectral variant, natural question: does more aggressive
dropout (p=0.3) help?

## Goal

Test if pushing dropout from p=0.2 (r216) to p=0.3 improves
4-scale + bias spectral.

## Mechanism

```python
# Same as r216 except dropout_p=0.3
# Reuses r216's FourScaleSpectralBiasDropCfCCell
# Just calls make with dropout_p=0.3 instead of 0.2
```

## Configurations (3 conds)

1. `cf`: r187 baseline
2. `4spectralbiasdrop`: r216 (dropout p=0.2)
3. `4spectralbiasdrop3`: r218 (dropout p=0.3)

## Result (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs, hidden=16)

| Cond | sin_irr | structured_irr | random_irr | mean |
|------|---------|----------------|------------|------|
| cf | 0.0555 | 0.0028 | 0.0971 | 0.0518 |
| 4spectralbiasdrop (p=0.2, r216) | 0.0411 | 0.0007 | 0.0844 | 0.0421 |
| **4spectralbiasdrop3 (p=0.3, r218)** | **0.0339** | **0.0084** | **0.0824** | **0.0416** |

Per-dataset (r218 vs cf):
- sin: -38.9% ✓
- structured: +200% ✗ REGRESSED
- random: -15.1% ✓
- mean: -19.7%

Per-dataset (r218 vs r216):
- sin: -17.5% ✓
- structured: +1100% ✗ MASSIVE REGRESSION
- random: -2.4% ✓
- mean: -1.2%

## Verdict

**TARGET-DEPENDENT 28th** 🎯 — dropout sweet spot is dataset-dependent.

## Pattern (54 + 27 + 59 = 140 → **54 + 28 + 59 = 141**)

- 54 strictly positive (unchanged)
- **28 target-dep (UP from 27, +1)** 🎯
- 59 negatives (unchanged)
- Total: **141 mechanism classes**

## Why target-dependence?

- sin data: more dropout (p=0.3) helps — regularizes noise
- random data: more dropout (p=0.3) helps — regularizes noise
- structured data: less dropout (p=0.2) better — preserves structure

**CRITICAL SEED INSTABILITY on struct**: r218 s0=0.0158 vs s1=0.0010.
Higher dropout creates seed sensitivity on structured data.

## Caveats

- 2 seeds, 30 epochs
- Hidden=16, lr=1e-2, batch_size=16

## Lesson

**p=0.2 (r216) is the safer default.** Only push to p=0.3
when you know the data is noisy.

## Next ideas

1. **Per-scale adaptive dropout p** — different p per scale
2. **Per-dataset dropout selection** — meta-learn
3. **Different bias parameterization** — per-scale (not per-frequency)
4. **PhysioNet test** — real-world data

**Why:** Round 218 is **TARGET-DEPENDENT 28th** — dropout
sweet spot is dataset-dependent. p=0.3 helps noisy data
but destabilizes structured.

**How to apply:** Use p=0.2 (r216) for unknown data. Use p=0.3
only for known-noisy data.
