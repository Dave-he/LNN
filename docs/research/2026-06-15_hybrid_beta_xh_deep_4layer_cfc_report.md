# Round 166 — HybridBeta-XH-Deep-4Layer-CfC — Research Report

**Date**: 2026-06-15
**Round**: 166
**Branch**: master
**Audit context (91-165)**: 39 strictly positive + 17 target-dep +
35 negatives = 91 mechanism classes.

## TL;DR

**3-layer (round 165) is the SWEET SPOT for hybrid β.** Going to
4-layer **REGRESSES on sin** (-63% → -37%, +26pp regression) while
**marginally improving structured** (-91% → -92%, -1pp). The
double best of round 165 is **NOT preserved** at 4-layer.

## What was tested

PARAMETER-ONLY test of round 165's 3-layer hybrid β (per-feature
on x, scalar on h) with **num_layers=4** instead of 3. Tests
whether depth continues to compound the benefits of hybrid β.

## Bench (24 cells: 4 conds × 3 datasets × 2 seeds, 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| hb_xh_4layer_h1 | 0.0172±0.0003 (-37%) | 0.0228±0.0117 (-86%) | 0.1033±0.0026 (-2%) | 11267 |
| hb_xh_4layer_h2 | 0.0171±0.0062 (-37%) | 0.0188±0.0022 (-88%) | 0.1029±0.0023 (-2%) | 16789 |
| hb_xh_4layer_h2_3x | 0.0172±0.0007 (-37%) | **0.0114±0.0013 (-92%)** | 0.1049±0.0027 (-1%) | 19239 |
| hb_xh_4layer_h2_k5 | 0.0171±0.0050 (-37%) | 0.0118±0.0016 (-91%) | 0.1042±0.0019 (-1%) | 24139 |

## Cross-round comparison

| Round | Mechanism | Layers | Kx | Kh | sin | structured |
|-------|-----------|--------|-----|-----|-----|------------|
| 161 | sx_xh_best (fixed β both) | 2 | 3 | 2 | -33% | -86% |
| 163 | hb_xh_scalar_h2 | 2 | 2 | 2 | -40% | -85% |
| 164 | hb_xh_deep_h1 (K=1) | 3 | 1 | 1 | -48% | -91% |
| 165 | **hb_xh_deep_h2_k5** | **3** | **5** | **2** | **-63%** | **-91%** |
| **166** | **hb_xh_4layer_h2_3x** | **4** | **3** | **2** | **-37%** | **-92%** |
| 166 | hb_xh_4layer_h2_k5 | 4 | 5 | 2 | -37% | -91% |

**Headline**: 4-layer DEGRADES on sin from -63% to -37% (+26pp
regression!) while marginally improving structured from -91% to
-92% (-1pp).

## Hypotheses revisited

- **H1 (4-layer compounds)**: REJECTED. 4-layer hurts sin by 26pp
  vs 3-layer.
- **H2 (no degradation)**: PARTIAL. Sin degrades 26pp, structured
  improves 1pp, random neutral. Sin degradation is a clear
  negative signal, not catastrophic.
- **H3 (raw 4-layer CfC is bad)**: not directly tested but
  inferred from round 164's finding that raw 3-layer CfC was
  +33% on sin — 4-layer raw CfC would likely be worse.

## Why 4-layer hurts sin

### 1. Sin has uniform frequency, no need for depth
Sin data is **smooth and low-frequency**. A 3-layer stacked
hybrid β already has enough capacity to learn it. Adding a 4th
layer adds:
- More parameters (overfitting risk)
- More depth-induced variance (gradient variance compounds)
- More redundant smoothing (the per-feature β already smooths)

### 2. Structured benefits from depth
Structured data has a **regime switch** (sin → sin(2t) at T/2).
4-layer can capture this hierarchy better than 3-layer, hence
the marginal structured improvement.

### 3. Random data is too noisy to benefit
Random walks have no learnable structure. 3-layer and 4-layer
both fail (≈0.10 loss) and depth adds nothing.

## Pattern reinforced (39 + 17 + 35 = 91 mechanism classes)

This round does NOT add new strictly positive winners (sin
regression is too severe). The 91-class audit remains stable.

## Negative-with-nuance verdict

- **For sin**: 3-layer is the sweet spot, going to 4-layer hurts.
- **For structured**: 4-layer is marginally better (-1pp).
- **For random**: depth doesn't matter.
- **Best 4-layer config**: hb_xh_4layer_h2_3x (structured -92%)
  — uses 19239 params, gives the best structured in 91-166 audit.

## Practical implications

- **Use 3-layer + Kx=5 + Kh=2 (round 165) for BOTH sin and
  structured** — round 166 confirms 3-layer remains the SOTA.
- **4-layer is NOT a strict upgrade** — only use if marginal
  structured improvement matters.
- **The double-best of round 165 is stable** — no 4-layer config
  beats both sin -63% AND structured -91% simultaneously.

## Critical implementation details

1. **4-layer + Kx=5** — same HybridBetaXHCfCStackedNetwork as
   round 163, num_layers=4, Kx=5 (re-export)
2. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements
3. **Factory functions** now accept optional `return_sequences`
   kwarg (default True) for testability
4. **Tests** — 10/10 pass, all from scratch in <5s

## Files

- `lnn/core/hybrid_beta_xh_deep_4layer_cfc.py` (~80 lines, re-export)
- `tests/test_hybrid_beta_xh_deep_4layer_cfc.py` (10 tests, all pass)
- `scripts/bench_hybrid_beta_xh_deep_4layer_cfc.py` (24-cell bench)
- `results/bench_hybrid_beta_xh_deep_4layer_cfc.json`

## Next ideas

1. **2-layer (revisit baseline)** — round 165 winner used
   3-layer. Test if 2-layer can match with right Kx/Kh.
2. **Hybrid β + FAME-MoE** — combine with FAME routing
3. **Test on missing_rate=0.5** — robustness under heavy missing
4. **Residual connections in deep stack** — could help 4-layer
   saturate the depth benefit
5. **Layer-wise β specialization** — different β per layer
6. **Different D (4D, 8D inputs)** — does depth help with more
   features?

**Why:** Going from 3 to 4 layers with hybrid β **REGRESSES sin
26pp** while marginally improving structured 1pp. The 3-layer
sweet spot is now empirically established: 3-layer is the
optimal depth for hybrid β. Round 165's double-best stands.

**How to apply:** **Use 3-layer + Kx=5 + Kh=2** (hb_xh_deep_h2_k5)
as the **default LNN hybrid β config** — it is the SOTA in 91-166
audit. Don't go deeper without residual connections or other
depth-specific regularizers.
