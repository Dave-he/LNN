# PRD #10-125 — Round 163 HybridBeta-XH-CfC (Scalar β on h + Per-Feature β on x)

**Date**: 2026-06-15
**Round**: 163
**Audit context (91-162)**: 27 strictly positive + 17 target-dep +
34 negatives = 78 mechanism classes.

## Motivation

Round 161 (Stacked-EMA-XH-CfC, Kx=3, Kh=2, **fixed β**) achieved
**BOTH best sin -33% AND best structured -86%** simultaneously.

Round 162 (LearnedBeta-XH-CfC, Kx=3, Kh=2, **per-feature β on
BOTH**) achieved **NEW BEST structured -90%** but TRADES sin
(-15% vs -33% round 161).

The hypothesis from round 162 finding: per-feature β on the hidden
state hurts sin (overfits on uniform-frequency data). The natural
test: use **per-feature β on x-side** (helps structured) +
**scalar β on h-side** (helps sin). Best of both worlds.

## Mechanism

For each step, augment BOTH input x and hidden state h with
multi-scale EMAs, but with **MIXED β parameterization**::

    # Per-feature learned β on x-side (round 157/162):
    beta_x_k,d = sigmoid(beta_x_k_raw[d])  # shape [Kx, D]

    # SCALAR fixed β on h-side (round 161):
    beta_h_k = fixed value (not learned, like 0.7/0.95/0.5/0.9/0.99)

    # Input-side EMAs (round 162 best):
    ema_x_k,t[d] = beta_x_k,d * ema_x_k,t-1[d] + (1 - beta_x_k,d) * x_t[d]
    aug_x_t = [x_t, ema_x_1,t - x_t, ..., ema_x_Kx,t - x_t]

    # Hidden-state EMAs (round 161 best, scalar β):
    ema_h_k,t[d] = beta_h_k * ema_h_k,t-1[d] + (1 - beta_h_k) * h_t[d]
    aug_h_t = [h_t, ema_h_1,t - h_t, ..., ema_h_Kh,t - h_t]

    z_t = cat(aug_x_t, aug_h_t)

### Variants (4 conds)

1. **hb_xh_scalar_h1**: Kx=1, Kh=1, **per-feature β on x,
   scalar β=0.9 on h**, both diff
2. **hb_xh_scalar_h2**: Kx=2, Kh=2, **per-feature β on x, scalar
   β ∈ {0.7, 0.95} on h** (round 160's best h config), both diff
3. **hb_xh_scalar_h2_3x**: Kx=3, Kh=2, **per-feature β on x,
   scalar β ∈ {0.7, 0.95} on h** (round 161's best config
   replicated but with per-feature x), both diff
4. **hb_xh_best**: Kx=3, Kh=2, per-feature β on x, scalar β on
   h, BOTH DIFF, balanced config

## Hypotheses

- **H1 (best of both)**: hb_xh_best achieves BOTH round 161's
  sin (-33%) AND round 162's structured (-90%) simultaneously.
- **H2 (no trade-off)**: per-feature β on x does NOT hurt sin
  (only the per-feature β on h hurt sin in round 162).
- **H3 (stable)**: training is stable across all 3 datasets.

## Bench plan (24 cells)

4 conds × 3 datasets × 2 seeds × 30 epochs:
- Datasets: sin_irr, structured_irr, random_irr.
- Compare to round 161 (sx_xh_best) and round 162 (lb_xh_best).

## Success criteria

- **STRICTLY POSITIVE** if all 3 datasets improve vs CfC.
- **BREAKTHROUGH+** if a cond achieves BOTH sin ≤-30% AND
  structured ≤-85% (matches or beats round 161 on sin AND
  round 162 on structured).
- **NEGATIVE** if any dataset degrades ≥30% (catastrophic).

## Files

- `lnn/core/hybrid_beta_xh_cfc.py` (~310 lines)
- `tests/test_hybrid_beta_xh_cfc.py` (~25 tests)
- `scripts/bench_hybrid_beta_xh_cfc.py` (24-cell bench)
- `docs/research/2026-06-15_hybrid_beta_xh_cfc_report.md`
- `memory/lnn-round-163-hybrid-beta-xh-cfc.md`
