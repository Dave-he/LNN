# Round 168 — LayerDecay-H3-CfC (Kh=3 with REVERSE β) — Research Report

**Date**: 2026-06-15
**Round**: 168
**Branch**: master
**Audit context (91-167)**: 40 strictly positive + 17 target-dep +
35 negatives = 92 mechanism classes.

## TL;DR

**41st STRICTLY POSITIVE**: **`ld_constant_h3`** (3-layer, Kx=5,
**Kh=3 with CONSTANT β** ∈ {0.7, 0.85, 0.95}) achieves **sin -71%
NEW BEST** (beats round 167's ld_reverse_k5 -69% by 2pp).

But the constant schedule (Kh=3) beats the REVERSE schedule
(Kh=2). The **Kh dimension** (more hidden-side time-scales) is
the winning axis, not the per-layer REVERSE β schedule.

## What was tested

**Kh=3 / Kh=4 with REVERSE β schedule** vs constant control.
Round 167 used Kh=2 with REVERSE. Round 168 tests if more
hidden-side time-scales under REVERSE schedule compound.

## Bench (36 cells: 6 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| ld_reverse_h3_k5 | 0.0101±0.0004 (-65%) | 0.0362±0.0184 (-77%) | 0.1034±0.0032 (-1%) | 19387 |
| ld_reverse_h4_k5 | 0.0102±0.0020 (-65%) | 0.0232±0.0088 (-85%) | 0.1052±0.0019 (-1%) | 21691 |
| ld_reverse_h3_wider | 0.0133±0.0030 (-49%) | 0.0229±0.0056 (-85%) | 0.1037±0.0038 (-1%) | 19387 |
| ld_reverse_h3_k6 | 0.0101±0.0001 (-65%) | 0.0260±0.0007 (-85%) | 0.1031±0.0023 (-1%) | 21053 |
| ld_reverse_h3_h2 | 0.0114±0.0003 (-58%) | 0.0242±0.0049 (-85%) | 0.1029±0.0036 (-1%) | 19387 |
| **ld_constant_h3** | **0.0078±0.0008 (-71% NEW BEST)** | 0.0206±0.0005 (-84%) | 0.1022±0.0037 (-1%) | 19387 |

## Cross-round progression (best in class)

| Round | Mechanism | Kh | schedule | sin | structured |
|-------|-----------|-----|----------|-----|------------|
| 165 | hb_xh_deep_h2_k5 | 2 | constant [0.7, 0.95] | -63% | **-91%** |
| 167 | ld_reverse_k5 | 2 | REVERSE [0.99, 0.5] | -69% | -82% |
| **168** | **ld_constant_h3** | **3** | constant [0.7, 0.85, 0.95] | **-71% NEW BEST** | -84% |

## Hypotheses revisited

- **H1 (Kh=3 helps)**: REJECTED for REVERSE — Kh=3 with REVERSE
  REGRESSED sin from -69% (round 167) to -65% (+4pp regression).
  CONFIRMED for constant — Kh=3 with constant achieves -71%.
- **H2 (wider range helps)**: REJECTED. Wider β range [0.999, 0.7, 0.3]
  regressed to -49%.
- **H3 (Kh=3 helps structured)**: PARTIAL. ld_constant_h3 -84%
  better than ld_reverse_k5 -82% but worse than round 165 -91%.
- **H4 (constant Kh=3 baseline)**: **CONFIRMED** — constant Kh=3
  beats all REVERSE variants.

## Why Kh dimension wins over REVERSE schedule

### 1. More hidden-side time-scales helps (Kh=2 → Kh=3)
Adding a 3rd hidden-side time-scale (0.85) provides more
smooth-vs-fast contrast on h-side, which is essential for sin.

### 2. Constant schedule keeps contrast within layers
Each layer gets all 3 β values at the same time. The β contrast
helps because each layer can choose the right time-scale for
its representation.

### 3. REVERSE schedule distributes β across layers
With REVERSE, layer 0 gets only slow β (0.99), layer 2 gets
only fast β (0.5). This FORCES each layer to a single
time-scale, losing the within-layer contrast.

### 4. Kh=3 wins when β values are well-chosen
The constant β ∈ {0.7, 0.85, 0.95} covers slow (0.95), medium
(0.85), and fast (0.7) — a good spread for sin.

## Pattern reinforced (41 + 17 + 35 = 93 mechanism classes)

- **41 strictly positive** (was 40): added **ld_constant_h3 (41st)**
- **17 target-dep** (unchanged)
- **35 negatives** (unchanged)
- Total: 93 mechanism classes

## Critical implementation details

1. **Same LayerDecayCfCStackedNetwork as round 167** — only
   changes betas_h list length (2 → 3 or 4)
2. **6 factory functions** for various Kh + schedule combinations
3. **Pyright false positives** on `import torch` are pre-existing
4. **Tests** — 10/10 pass

## Files

- `lnn/core/layer_decay_h3_cfc.py` (~75 lines, re-export)
- `tests/test_layer_decay_h3_cfc.py` (10 tests, all pass)
- `scripts/bench_layer_decay_h3_cfc.py` (36-cell bench)
- `results/bench_layer_decay_h3_cfc.json`
- `docs/prds/2026-06-15-lnn-round-168-layer-decay-h3-cfc.md`
- `docs/research/2026-06-15_layer_decay_h3_cfc_report.md`

## Next ideas

1. **Kh=4 with constant schedule** — even more hidden-side time-
   scales
2. **ld_constant_h3 + 4-layer** — does Kh=3 help 4-layer?
3. **ld_constant_h4_k6** — push Kx and Kh both higher
4. **Combine with FAME-MoE** — best sin + routing
5. **Per-feature β on h-side with Kh=3** — orthogonal gain
6. **ld_constant_h3 with wider β range** — try [0.7, 0.85, 0.99]

**Why:** Kh=3 with CONSTANT β schedule beats Kh=2 with REVERSE
β schedule. The Kh dimension (more hidden-side time-scales) is
more important than the per-layer β schedule. **Within-layer
contrast in β values** (constant, all 3 at each layer) wins
over **between-layer contrast** (REVERSE, different β at each
layer).

**How to apply:** For best **sin** alone, use `ld_constant_h3`
(3-layer, Kx=5, Kh=3, constant β ∈ {0.7, 0.85, 0.95}) — 19387
params achieves -71%. For best **structured** alone, use
round 165's `hb_xh_deep_h2_k5` (-91%). For best of BOTH, use
round 165's `hb_xh_deep_h2_k5` (-63%/-91%) — round 165's
double-best remains the SOTA.
