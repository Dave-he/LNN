# Round 169 — LayerDecay-H4-CfC (Kh=4 with constant β) — Research Report

**Date**: 2026-06-15
**Round**: 169
**Branch**: master
**Audit context (91-168)**: 41 strictly positive + 17 target-dep +
35 negatives = 93 mechanism classes.

## TL;DR

**42nd STRICTLY POSITIVE**: **`ld_constant_h3_finer`** (3-layer,
Kx=5, Kh=3, constant β ∈ {0.75, 0.85, 0.95}) achieves **sin -72%
NEW BEST** (1pp better than round 168's -71%).

Notable side finding: `ld_constant_h4_wide` ties structured
**-91% (round 165 best)** but only achieves sin -49%.

**Kh=3 is the SWEET SPOT** — Kh=4 and Kh=5 REGRESS. Kx=6 doesn't
help with Kh=3. β values matter — {0.75, 0.85, 0.95} > {0.7,
0.85, 0.95}.

## Bench (42 cells: 7 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| ld_constant_h4_default | 0.0103±0.0009 (-65%) | 0.0177±0.0017 (-89%) | 0.1023±0.0032 (-2%) | 21691 |
| ld_constant_h4_wide | 0.0139±0.0034 (-49%) | **0.0136±0.0004 (-91%)** | 0.1028±0.0033 (-2%) | 21691 |
| ld_constant_h4_narrow | 0.0118±0.0002 (-58%) | 0.0317±0.0002 (-81%) | 0.1046±0.0023 (-2%) | 21691 |
| ld_constant_h3_k6 (Kx=6) | 0.0118±0.0014 (-58%) | 0.0192±0.0020 (-88%) | 0.1042±0.0026 (-1%) | 21053 |
| ld_constant_h3_wider | 0.0151±0.0020 (-43%) | 0.0213±0.0057 (-87%) | 0.1033±0.0027 (-1%) | 19387 |
| **ld_constant_h3_finer** | **0.0075±0.0001 (-72% NEW BEST)** | 0.0184±0.0015 (-87%) | 0.1029±0.0028 (-1%) | 19387 |
| ld_constant_h5 | 0.0221±0.0062 (-23%) | 0.0154±0.0007 (-90%) | 0.1042±0.0039 (-1%) | 23995 |

## Cross-round (best in class)

| Round | Mechanism | Kh | β values | sin | structured |
|-------|-----------|-----|----------|-----|------------|
| 165 | hb_xh_deep_h2_k5 | 2 | {0.7, 0.95} | -63% | **-91%** |
| 167 | ld_reverse_k5 | 2 | REVERSE [0.99, 0.5] | -69% | -82% |
| 168 | ld_constant_h3 | 3 | {0.7, 0.85, 0.95} | -71% | -84% |
| **169** | **ld_constant_h3_finer** | **3** | **{0.75, 0.85, 0.95}** | **-72% NEW BEST** | -87% |

## Side finding: ld_constant_h4_wide

While the headline is h3_finer, **`ld_constant_h4_wide`** (Kh=4,
β ∈ {0.5, 0.7, 0.85, 0.99}) achieves **structured -91% (TIED
with round 165)** — but at the cost of sin -49% (much worse).

This shows that **wider β range on h-side** is essential for
structured data (which has regime switches and needs both very
slow and very fast time-scales).

## Hypotheses revisited

- **H1 (Kh=4 helps)**: REJECTED. Kh=4 regressed on sin (-65% vs
  round 168's -71%).
- **H2 (Kx=6 helps)**: REJECTED. Kx=6 regressed on sin (-58%).
- **H3 (wider β range)**: REJECTED for sin. CONFIRMED for
  structured — wider β achieves -91% structured (round 165 tied).
- **H4 (Kh=5 saturates)**: CONFIRMED. Kh=5 regressed severely
  on sin (-23%).

## Pattern reinforced (42 + 17 + 35 = 94 mechanism classes)

- **42 strictly positive** (was 41): added **ld_constant_h3_finer
  (42nd)**
- **17 target-dep** (unchanged)
- **35 negatives** (unchanged)
- Total: 94 mechanism classes

## Critical implementation details

1. **Same LayerDecayCfCStackedNetwork as round 167** — only
   changes betas_h list (3 vs 4 vs 5 values, different β values)
2. **7 factory functions** for various Kh + β range combinations
3. **Pyright false positives** on `import torch` are pre-existing
4. **Tests** — 11/11 pass

## Why h3_finer beats h3 default

### 1. β values closer to slow regime help sin
{0.75, 0.85, 0.95} all sit in the slow-to-medium range.
{0.7, 0.85, 0.95} includes a fast β (0.7) which adds variance.

### 2. 3 time-scales is enough
With 3 distinct β values, the model can already capture
smooth/medium/very-slow regimes. The added β value (0.7) just
introduces noise.

### 3. Kh=3 is the sweet spot for the missing_rate=0.3 + sin task
The number of EMAs needed is bounded by the data complexity. Sin
data doesn't need 4 or 5 distinct time-scales.

## Files

- `lnn/core/layer_decay_h4_cfc.py` (~80 lines, re-export)
- `tests/test_layer_decay_h4_cfc.py` (11 tests, all pass)
- `scripts/bench_layer_decay_h4_cfc.py` (42-cell bench)
- `results/bench_layer_decay_h4_cfc.json`
- `docs/prds/2026-06-15-lnn-round-169-layer-decay-h4-cfc.md`
- `docs/research/2026-06-15_layer_decay_h4_cfc_report.md`

## Next ideas

1. **ld_constant_h3_optimal** — grid search β ∈ {0.7-0.95} for
   Kh=3 to find the absolute optimum
2. **ld_constant_h3 + 4-layer** — does h3 help 4-layer?
3. **Combine h3_finer with Kx=6** — orthogonal push
4. **ld_constant_h3_per_feature** — per-feature β on h-side
5. **Adaptive β (learned) with h3_finer init** — gradient-descent
   on the β values themselves

**Why:** Kh=3 with finer β ∈ {0.75, 0.85, 0.95} beats Kh=3 with
default β ∈ {0.7, 0.85, 0.95} by 1pp on sin. **β values matter**,
not just Kh count. Kh=4 and Kh=5 REGRESS.

**How to apply:** For best **sin** alone, use `ld_constant_h3_finer`
(3-layer, Kx=5, Kh=3, constant β ∈ {0.75, 0.85, 0.95}) — 19387
params achieves -72%. For best **structured** alone, use
`ld_constant_h4_wide` (Kh=4, β ∈ {0.5, 0.7, 0.85, 0.99}) — 21691
params achieves -91% (TIED round 165). For best of BOTH, use
round 165's `hb_xh_deep_h2_k5` (-63%/-91%) — round 165's
double-best remains the SOTA.
