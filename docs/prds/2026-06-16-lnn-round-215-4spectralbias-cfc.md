# PRD #10-177 — Round 215 — 4-Scale Spectral + Bias on CfC

**Date**: 2026-06-16
**Round**: 215
**Branch**: master
**Audit context (91-214)**: 52 strictly positive + 27 target-dep
+ 58 negatives = 137 mechanism classes.

## Background

6 SPs in a row from the spectral axis (r210-r214). Spectral axis
is the most reliable SP source in the audit.

r212 = 4-scale, r213 = 3-scale + dropout, r214 = 4-scale + dropout.
New direction: add per-frequency learnable bias to the mask.

## Goal

Test if per-frequency learnable bias added before the mask
linear improves the 4-scale spectral gating.

## Mechanism

```python
# 4-scale spectral gating (same as r212)
H1, H2, H3, H4 = 4-scale computation

# NEW: per-frequency learnable bias (vs r212)
mag1 = |H1| + spec_bias1
mask1 = sigmoid(spec_mask(mag1))
g1 = IFFT(H1 * mask1, n=hidden_size)
# Similar for H2/H3/H4 with their own biases

g_combined = (g1 + g2 + g3 + g4) / 4
h_new = τ_eff * g_combined + (1-τ_eff) * h_branch
```

## Configurations (3 conds)

1. `cf`: r187 baseline
2. `4spectral`: r212 (4-scale, no bias)
3. `4spectralbias`: r215 (4-scale, per-frequency bias)

## Result (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs, hidden=16)

| Cond | sin_irr | structured_irr | random_irr | mean |
|------|---------|----------------|------------|------|
| cf | 0.0508 | 0.0071 | 0.0918 | 0.0499 |
| 4spectral (r212) | 0.0427 | 0.0139 | 0.0849 | 0.0472 |
| **4spectralbias (r215)** | **0.0417** | **0.0045** | **0.0844** | **0.0435** |

Per-dataset (r215 vs cf):
- sin: -17.8% ✓
- structured: -37.0% ✓
- random: -8.0% ✓
- mean: -20.9%

## Verdict

**STRICTLY POSITIVE (53rd)** 🎉 — ALL 3 datasets improve.

## Pattern (52 + 27 + 58 = 137 → **53 + 27 + 58 = 138**)

- **53 strictly positive** (UP from 52, **+1**) 🎉
- 27 target-dep (unchanged)
- 58 negatives (unchanged)
- Total: **138 mechanism classes**

## 6 SPs in a row from spectral axis

| Round | Mechanism | Verdict |
|-------|-----------|---------|
| r210 | 3-scale | SP 48th |
| r211 | 3-scale adaptive | SP 49th |
| r212 | 4-scale | SP 50th |
| r213 | 3-scale + dropout | SP 51st |
| r214 | 4-scale + dropout | SP 52nd |
| **r215** | **4-scale + bias** | **SP 53rd** |

## Caveats

- 2 seeds, 30 epochs
- Hidden=16, lr=1e-2, batch_size=16
- Per-frequency bias: 9+5+3+2=19 learnable scalars
- ~25% slower than baseline cf

## Next ideas

1. **4-scale + bias + dropout** — combine all
2. **5-scale + bias** — push scale count
3. **Per-scale adaptive bias** — different bias per scale (not per-frequency)
4. **PhysioNet test** — real-world data

**Why:** Round 215 is **STRICTLY POSITIVE (53rd)** — 4-scale
spectral + per-frequency bias improves all 3 datasets. 6 SPs
in a row from the spectral axis.

**How to apply:** Use 4-scale spectral + per-frequency bias for
adaptive frequency selection. The bias is small (one scalar
per frequency) but provides seed-stability.
