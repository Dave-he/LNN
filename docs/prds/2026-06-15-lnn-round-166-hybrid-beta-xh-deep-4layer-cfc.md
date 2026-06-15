# PRD #10-128 — Round 166 HybridBeta-XH-Deep-4Layer-CfC (4-Layer Stacked)

**Date**: 2026-06-15
**Round**: 166
**Audit context (91-165)**: 39 strictly positive + 17 target-dep +
35 negatives = 91 mechanism classes.

## Motivation

Rounds 156-165 (10 rounds, 23+ strictly positive winners) explored
EMA-based augmentations. The best in class is **HybridBeta-XH-Deep-HighK**
(round 165) which achieved:
- sin **-63% (NEW BEST)** with hb_xh_deep_h2_k5 (3-layer, Kx=5, Kh=2)
- structured **-91% (tied NEW BEST)** with hb_xh_deep_h2_k5

Round 165's winner used 3-layer. Round 164 showed 3-layer
compounds the benefits. The natural test: does **4-layer stacking**
compound the benefits even further?

Round 164 also showed that **raw 3-layer CfC is WORSE on sin
(+33%)** — the hybrid β augmentation is essential. So 4-layer
testing with raw CfC would be a control. We'll test 4-layer with
hybrid β to see if depth continues to help.

## Mechanism

Same as round 165's HybridBeta-XH-Deep-HighK, but with **4 layers**
instead of 3::

    For each of 4 layers:
        # Per-feature learned β on x-side (round 163/164/165):
        beta_x_k,d = sigmoid(beta_x_k_raw[d])  # shape [Kx, D]
        # SCALAR fixed β on h-side (round 163/164/165):
        beta_h_k = fixed value (e.g. 0.7, 0.95)
        # Input-side EMAs (per-feature):
        ema_x_k,t[d] = beta_x_k,d * ema_x_k,t-1[d] + (1 - beta_x_k,d) * x_t[d]
        # Hidden-state EMAs (scalar):
        ema_h_k,t[d] = beta_h_k * ema_h_k,t-1[d] + (1 - beta_h_k) * h_t[d]
        z_t = cat(aug_x_t, aug_h_t)
        h_t = CfC(z_t)

### Variants (4 conds)

1. **hb_xh_4layer_h1**: 4-layer, Kx=1, Kh=1, scalar β=0.9 on h
2. **hb_xh_4layer_h2**: 4-layer, Kx=2, Kh=2, scalar β ∈ {0.7, 0.95} on h
3. **hb_xh_4layer_h2_3x**: 4-layer, Kx=3, Kh=2, scalar β ∈ {0.7, 0.95} on h
4. **hb_xh_4layer_h2_k5**: 4-layer, Kx=5, Kh=2, scalar β ∈ {0.7, 0.95} on h (round 165 best config)

## Hypotheses

- **H1 (4-layer compounds)**: 4-layer hb_xh_4layer_h2_k5 beats
  3-layer round 165 hb_xh_deep_h2_k5 on at least one dimension.
- **H2 (no degradation)**: 4-layer doesn't catastrophically
  hurt on any dataset.
- **H3 (raw 4-layer CfC is bad)**: raw 4-layer CfC is WORSE than
  3-layer (round 164's finding extends).

## Bench plan (24 cells)

4 conds × 3 datasets × 2 seeds × 30 epochs:
- Datasets: sin_irr, structured_irr, random_irr.
- Compare to round 165 hb_xh_deep_h2_k5 (sin -63%, structured -91%).

## Success criteria

- **STRICTLY POSITIVE** if all 3 datasets improve vs CfC.
- **4-LAYER BREAKTHROUGH** if a cond beats round 165 on BOTH
  sin AND structured.
- **NEGATIVE** if any dataset degrades ≥30% (catastrophic).

## Files

- `lnn/core/hybrid_beta_xh_deep_4layer_cfc.py` (re-export)
- `tests/test_hybrid_beta_xh_deep_4layer_cfc.py` (~10 tests)
- `scripts/bench_hybrid_beta_xh_deep_4layer_cfc.py` (24-cell bench)
- `docs/research/2026-06-15_hybrid_beta_xh_deep_4layer_cfc_report.md`
- `memory/lnn-round-166-hybrid-beta-xh-deep-4layer-cfc.md`

## Why this is cheap

- 100% code reuse from round 163.
- The only new code is factory functions for 4-layer.
- Total new code: <100 lines.
