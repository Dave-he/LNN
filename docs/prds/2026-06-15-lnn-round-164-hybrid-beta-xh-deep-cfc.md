# PRD #10-126 — Round 164 HybridBeta-XH-Deep-CfC (3-Layer Stacked)

**Date**: 2026-06-15
**Round**: 164
**Audit context (91-163)**: 31 strictly positive + 17 target-dep +
34 negatives = 82 mechanism classes.

## Motivation

Rounds 156-163 (8 rounds, 14+ strictly positive winners) explored
EMA-based augmentations. The best in class is **HybridBeta-XH-CfC**
(round 163) which achieved:
- sin **-40% (NEW BEST)** with hb_xh_scalar_h2
- structured **-88% (NEW BEST)** with hb_xh_best

All these winners used **2-layer stacked networks**. The natural
test: does **3-layer stacking** compound the benefits?

This is a parameter-only test (num_layers=3 instead of 2) — no
new core code, just bench deeper variants of round 163's winners.

## Mechanism

Same as round 163's HybridBeta-XH-CfC, but with **3 stacked cells**
instead of 2::

    For each of 3 layers:
        # Per-feature learned β on x-side (round 163):
        beta_x_k,d = sigmoid(beta_x_k_raw[d])  # shape [Kx, D]
        # SCALAR fixed β on h-side (round 163):
        beta_h_k = fixed value (e.g. 0.7, 0.95)
        # Input-side EMAs (per-feature):
        ema_x_k,t[d] = beta_x_k,d * ema_x_k,t-1[d] + (1 - beta_x_k,d) * x_t[d]
        # Hidden-state EMAs (scalar):
        ema_h_k,t[d] = beta_h_k * ema_h_k,t-1[d] + (1 - beta_h_k) * h_t[d]
        z_t = cat(aug_x_t, aug_h_t)
        h_t = CfC(z_t)

### Variants (4 conds)

1. **hb_xh_deep_h1**: 3-layer, Kx=1, Kh=1, scalar β=0.9 on h
2. **hb_xh_deep_h2**: 3-layer, Kx=2, Kh=2, scalar β ∈ {0.7, 0.95} on h
3. **hb_xh_deep_h2_3x**: 3-layer, Kx=3, Kh=2, scalar β ∈ {0.7, 0.95} on h
4. **hb_xh_deep_best**: 3-layer, Kx=3, Kh=2, scalar β ∈ {0.7, 0.95} on h

## Hypotheses

- **H1 (depth compounds)**: 3-layer hb_xh_deep_best beats
  2-layer round 163 hb_xh_best on at least one dimension.
- **H2 (no degradation)**: 3-layer doesn't catastrophically
  hurt on any dataset.
- **H3 (stable)**: training is stable with 3 layers.

## Bench plan (24 cells)

4 conds × 3 datasets × 2 seeds × 30 epochs:
- Datasets: sin_irr, structured_irr, random_irr.
- Compare to round 163 hb_xh_best (sin -29%, structured -88%).

## Success criteria

- **STRICTLY POSITIVE** if all 3 datasets improve vs CfC.
- **DEPTH BREAKTHROUGH** if a cond beats round 163 on BOTH
  sin AND structured.
- **NEGATIVE** if any dataset degrades ≥30% (catastrophic).

## Files

- `lnn/core/hybrid_beta_xh_deep_cfc.py` (re-exports round 163)
- `tests/test_hybrid_beta_xh_deep_cfc.py` (~10 tests)
- `scripts/bench_hybrid_beta_xh_deep_cfc.py` (24-cell bench)
- `docs/research/2026-06-15_hybrid_beta_xh_deep_cfc_report.md`
- `memory/lnn-round-164-hybrid-beta-xh-deep-cfc.md`

## Why this is cheap

- 100% code reuse from round 163's HybridBetaXHCfCStackedNetwork.
- The only new code is a test for num_layers=3.
- The bench script creates 4 conditions with num_layers=3.
- Total new code: <100 lines.
