# PRD #10-176 — Round 214 — 4-Scale Spectral + Dropout on CfC

**Date**: 2026-06-16
**Round**: 214
**Branch**: master
**Audit context (91-213)**: 51 strictly positive + 27 target-dep
+ 58 negatives = 136 mechanism classes.

## Background

5 SPs in a row from the spectral axis (r210-r213). Spectral
axis is the most reliable SP source in the audit.

r212 = 4-scale spectral SP 50th, r213 = 3-scale + dropout SP 51st.
Natural combination: 4-scale + dropout.

## Goal

Test if combining r212's 4-scale spectral gating with r213's
spectral mask dropout (p=0.2) provides the most robust result.

## Mechanism

```python
# 4-scale spectral gating (same as r212)
H1, H2, H3, H4 = 4-scale computation
mask1, mask2, mask3, mask4 = sigmoid(linear(|H|))

# Per-scale dropout (same as r213)
if self.training:
    mask1 = F.dropout(mask1, p=0.2)
    mask2 = F.dropout(mask2, p=0.2)
    mask3 = F.dropout(mask3, p=0.2)
    mask4 = F.dropout(mask4, p=0.2)

g_combined = (g1 + g2 + g3 + g4) / 4
h_new = τ_eff * g_combined + (1-τ_eff) * h_branch
```

## Configurations (3 conds)

1. `cf`: r187 baseline
2. `4spectral`: r212 (4-scale, no dropout)
3. `4spectraldrop`: r214 (4-scale, dropout p=0.2)

## Result (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs, hidden=16)

| Cond | sin_irr | structured_irr | random_irr | mean |
|------|---------|----------------|------------|------|
| cf | 0.0556 | 0.0058 | 0.0874 | 0.0496 |
| 4spectral (r212) | 0.0426 | 0.0012 | 0.0870 | 0.0436 |
| **4spectraldrop (r214)** | **0.0412** | **0.0011** | **0.0844** | **0.0422** |

Per-dataset (r214 vs cf):
- sin: -25.9% ✓
- structured: -81.1% ✓
- random: -3.5% ✓
- mean: -36.8%

## Verdict

**STRICTLY POSITIVE (52nd)** 🎉 — ALL 3 datasets improve.

## Pattern (51 + 27 + 58 = 136 → **52 + 27 + 58 = 137**)

- **52 strictly positive** (UP from 51, **+1**) 🎉
- 27 target-dep (unchanged)
- 58 negatives (unchanged)
- Total: **137 mechanism classes**

## 5 SPs in a row from spectral axis

| Round | Mechanism | Verdict |
|-------|-----------|---------|
| r210 | 3-scale | SP 48th |
| r211 | 3-scale adaptive | SP 49th |
| r212 | 4-scale | SP 50th |
| r213 | 3-scale + dropout | SP 51st |
| **r214** | **4-scale + dropout** | **SP 52nd** |

## Caveats

- 2 seeds, 30 epochs
- Hidden=16, lr=1e-2, batch_size=16
- 4-scale + dropout ~25% slower than baseline

## Next ideas

1. **5-scale + dropout** — push to 5 scales
2. **Per-scale adaptive dropout** — different p per scale
3. **PhysioNet test** — real-world data
4. **Combine spectral with QuITE embedding** (r102)

**Why:** Round 214 is **STRICTLY POSITIVE (52nd)** — 4-scale
spectral + dropout improves all 3 datasets. 5 SPs in a row
from the spectral axis.

**How to apply:** Use 4-scale spectral + dropout p=0.2 for the
most robust multi-scale regularization. Combines 4-scale
spectral richness with dropout stability.
