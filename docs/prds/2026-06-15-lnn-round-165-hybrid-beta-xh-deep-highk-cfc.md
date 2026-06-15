# PRD #10-127 — Round 165 HybridBeta-XH-Deep-HighK-CfC (3-Layer + High K)

**Date**: 2026-06-15
**Round**: 165
**Audit context (91-164)**: 35 strictly positive + 17 target-dep +
35 negatives = 87 mechanism classes.

## Motivation

Rounds 156-164 (9 rounds, 19+ strictly positive winners) explored
EMA-based augmentations. The best in class is **HybridBeta-XH-Deep**
(round 164) which achieved:
- sin **-48% (NEW BEST)** with hb_xh_deep_h1 (K=1)
- structured **-91% (NEW BEST)** with hb_xh_deep_h1 (K=1)

Round 164's winner used the SIMPLEST K=1. But round 161's
2-layer winner used K=3 (sx_xh_best). The natural test: what
about K=4 or K=5 with 3-layer? Does adding more time-scales
help structured even more?

## Mechanism

Same as round 164's HybridBeta-XH-Deep-CfC, but with HIGHER Kx::

    For each of 3 layers:
        # Per-feature learned β on x-side (round 163/164):
        beta_x_k,d = sigmoid(beta_x_k_raw[d])  # shape [Kx, D]
        # SCALAR fixed β on h-side (round 163/164):
        beta_h_k = fixed value (e.g. 0.7, 0.95)
        # Input-side EMAs (per-feature):
        ema_x_k,t[d] = beta_x_k,d * ema_x_k,t-1[d] + (1 - beta_x_k,d) * x_t[d]
        # Hidden-state EMAs (scalar):
        ema_h_k,t[d] = beta_h_k * ema_h_k,t-1[d] + (1 - beta_h_k) * h_t[d]
        z_t = cat(aug_x_t, aug_h_t)
        h_t = CfC(z_t)

### Variants (4 conds)

1. **hb_xh_deep_h1_k4**: 3-layer, Kx=4, Kh=1, scalar β=0.9 on h
2. **hb_xh_deep_h1_k5**: 3-layer, Kx=5, Kh=1, scalar β=0.9 on h
3. **hb_xh_deep_h2_k4**: 3-layer, Kx=4, Kh=2, scalar β ∈ {0.7, 0.95} on h
4. **hb_xh_deep_h2_k5**: 3-layer, Kx=5, Kh=2, scalar β ∈ {0.7, 0.95} on h

## Hypotheses

- **H1 (high K helps)**: Kx=4 or Kx=5 with 3-layer beats
  round 164 hb_xh_deep_h1 (Kx=1) on at least one dimension.
- **H2 (no degradation)**: high K doesn't catastrophically
  hurt on any dataset.
- **H3 (stable)**: training is stable with high K and 3 layers.

## Bench plan (24 cells)

4 conds × 3 datasets × 2 seeds × 30 epochs:
- Datasets: sin_irr, structured_irr, random_irr.
- Compare to round 164 hb_xh_deep_h1 (sin -48%, structured -91%).

## Success criteria

- **STRICTLY POSITIVE** if all 3 datasets improve vs CfC.
- **HIGH-K BREAKTHROUGH** if a cond beats round 164 on BOTH
  sin AND structured.
- **NEGATIVE** if any dataset degrades ≥30% (catastrophic).

## Files

- `lnn/core/hybrid_beta_xh_deep_highk_cfc.py` (re-export)
- `tests/test_hybrid_beta_xh_deep_highk_cfc.py` (~10 tests)
- `scripts/bench_hybrid_beta_xh_deep_highk_cfc.py` (24-cell bench)
- `docs/research/2026-06-15_hybrid_beta_xh_deep_highk_cfc_report.md`
- `memory/lnn-round-165-hybrid-beta-xh-deep-highk-cfc.md`

## Why this is cheap

- 100% code reuse from round 163/164.
- The only new code is factory functions for high K.
- Total new code: <100 lines.
