# PRD #10-183 — Round 221 — 3-Scale Spectral + Bias + Dropout on CfC

**Date**: 2026-06-16
**Round**: 221
**Branch**: master
**Audit context (91-220)**: 56 strictly positive + 28 target-dep
+ 59 negatives = 143 mechanism classes.

## Background

r216 (4-scale + bias + dropout) is the canonical spectral variant.
r213 was 3-scale + dropout (no bias). Natural question: does
3-scale + bias + dropout (combining r213+r215) work as well as
r216's 4-scale?

## Goal

Test if 3-scale + bias + dropout is competitive with r216's
4-scale variant.

## Mechanism

```python
# 3 scales (vs 4 in r216)
H1, H2, H3 = 3-scale computation  # full, half, quarter

# Per-frequency bias + dropout (same as r216)
for each scale:
    mag = |H| + bias
    mask = sigmoid(linear(mag))
    if training: mask = F.dropout(mask, p=0.2)
    g = IFFT(H * mask, n=hidden)

g_combined = (g1+g2+g3) / 3
h_new = τ_eff * g + (1-τ_eff) * h_branch
```

## Configurations (3 conds)

1. `cf`: r187 baseline
2. `4spectralbiasdrop`: r216 (4-scale)
3. `3spectralbiasdrop`: r221 (3-scale)

## Result (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs, hidden=16)

| Cond | sin_irr | structured_irr | random_irr | mean |
|------|---------|----------------|------------|------|
| cf | 0.0648 | 0.0048 | 0.0939 | 0.0545 |
| 4spectralbiasdrop (r216) | 0.0476 | 0.0010 | 0.0839 | 0.0442 |
| **3spectralbiasdrop (r221)** | **0.0423** | **0.0031** | **0.0869** | **0.0441** |

Per-dataset (r221 vs cf):
- sin: -34.7% ✓
- structured: -35.4% ✓
- random: -7.5% ✓
- mean: -19.1%

## Verdict

**STRICTLY POSITIVE (57th)** 🎉 — ALL 3 datasets improve vs cf.

## Pattern (56 + 28 + 59 = 143 → **57 + 28 + 59 = 144**)

- **57 strictly positive (UP from 56, +1)** 🎉
- 28 target-dep (unchanged)
- 59 negatives (unchanged)
- Total: **144 mechanism classes**

## Caveats

- 2 seeds, 30 epochs
- Hidden=16, lr=1e-2, batch_size=16
- ~10% faster than 4-scale

## Lesson

**3-scale and 4-scale are roughly equivalent** in this regime.

## Next ideas

1. **2-scale + bias + dropout** — minimal scale
2. **Adaptive per-layer scale count** — different per layer
3. **Cross-scale attention** — let scales attend to each other
4. **PhysioNet test** — real-world data

**Why:** Round 221 is **STRICTLY POSITIVE 57th** — 3-scale +
bias + dropout improves all 3 datasets vs cf. 3-scale and
4-scale are roughly equivalent.

**How to apply:** Use 3-scale for faster inference (~10%),
4-scale for marginal improvement on struct.
