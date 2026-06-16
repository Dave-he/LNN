# PRD #10-179 — Round 217 — 5-Scale Spectral + Bias + Dropout on CfC

**Date**: 2026-06-16
**Round**: 217
**Branch**: master
**Audit context (91-216)**: 54 strictly positive + 27 target-dep
+ 58 negatives = 139 mechanism classes.

## Background

7 SPs in a row from the spectral axis (r210-r216). Natural
question: does pushing to 5 scales (sixteenth added) improve
on r216's 4-scale + bias + dropout?

## Goal

Test if a 5th scale (sixteenth, 1 frequency for hidden=16)
on top of r216's 4-scale + bias + dropout helps.

## Mechanism

```python
# 5 scales (vs 4 in r216)
H1, H2, H3, H4, H5 = 5-scale computation
# H5 = H1[:, :hidden_size // 32 + 1] = 1 freq for hidden=16

# Per-frequency bias + dropout (same as r216)
for each scale:
    mag = |H| + bias
    mask = sigmoid(linear(mag))
    if training: mask = F.dropout(mask, p=0.2)
    g = IFFT(H * mask, n=hidden)

g_combined = (g1+...+g5) / 5
h_new = τ_eff * g + (1-τ_eff) * h_branch
```

## Configurations (3 conds)

1. `cf`: r187 baseline
2. `4spectralbiasdrop`: r216 (4-scale, bias, drop)
3. `5spectralbiasdrop`: r217 (5-scale, bias, drop)

## Result (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs, hidden=16)

| Cond | sin_irr | structured_irr | random_irr | mean |
|------|---------|----------------|------------|------|
| cf | 0.0597 | 0.0031 | 0.0885 | 0.0504 |
| 4spectralbiasdrop (r216) | 0.0381 | 0.0018 | 0.0854 | 0.0418 |
| **5spectralbiasdrop (r217)** | **0.0422** | **0.0083** | **0.0862** | **0.0456** |

Per-dataset (r217 vs cf):
- sin: -29.3% ✓
- structured: +167.7% ✗ REGRESSED
- random: -2.6% ✓
- mean: -9.5%

Per-dataset (r217 vs r216):
- sin: +10.8% ✗ REGRESSED
- structured: +361.1% ✗ REGRESSED
- random: +0.9% ✗
- mean: +9.1% (worse)

## Verdict

**NEGATIVE 59th** 🎯 — 5-scale regresses vs 4-scale on ALL 3 datasets.

## Pattern (54 + 27 + 58 = 139 → **54 + 27 + 59 = 140**)

- 54 strictly positive (unchanged)
- 27 target-dep (unchanged)
- **59 negatives (UP from 58, +1)** 🎯
- Total: **140 mechanism classes**

## Why 5-scale hurts

For hidden=16, the 5th scale (sixteenth) has only **1 frequency**.
This is too few to be useful — the mask linear (1→1) doesn't
provide meaningful frequency selection.

The 4-scale sweet spot:
- Scale 1 (full, 9 freqs) — most info
- Scale 2 (half, 5 freqs) — broad
- Scale 3 (quarter, 3 freqs) — medium
- Scale 4 (eighth, 2 freqs) — narrow but still useful

5th scale at 1 freq is **sub-threshold** — the model can't
distinguish this from bias noise.

## Caveats

- 2 seeds, 30 epochs
- Hidden=16, lr=1e-2, batch_size=16
- 5-scale ~15% slower than 4-scale

## Lesson

**4-scale is the sweet spot** for the spectral axis.

## Next ideas

1. **Adaptive scale weights** (not simple avg) — round 211 was 3-scale adaptive
2. **Spectral L2 regularization** — penalize mask norm
3. **Different bias parameterization** — per-scale (not per-frequency) bias
4. **PhysioNet test** — real-world data

**Why:** Round 217 is **NEGATIVE 59th** — 5-scale regresses on
all 3 datasets vs r216's 4-scale. The 4-scale sweet spot is
confirmed.

**How to apply:** Don't push past 4 scales for hidden=16.
Use 4-scale + bias + dropout (r216) as the canonical
spectral variant.
