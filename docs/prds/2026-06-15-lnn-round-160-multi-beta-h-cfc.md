# PRD #10-122 — Round 160 MultiBeta-H-CfC (Multi-Scale Hidden State EMA)

**Date**: 2026-06-15
**Round**: 160
**Audit context (91-159)**: 21 strictly positive + 16 target-dep +
28 negatives = 65 mechanism classes.

## Motivation

Rounds 156-159 (4 rounds, 7 strictly positive winners) all
explored EMA-style mechanisms. The audit reveals:
- Round 156 (EMA-X): K=1 scalar β on input x — 17th positive (-42%)
- Round 157 (LearnedBeta): per-feature learnable β on input x — 18th positive (-63%)
- Round 158 (MultiBeta): K=2/K=3 fixed β on input x — 19th/20th positive (-60%/-65%)
- Round 159 (EMA-H): K=1 scalar β on hidden state h — 21st positive (-77%)

The natural next step is the **CROSS-PRODUCT**: apply K=2 or K=3
multi-scale EMA pattern to the hidden state h (vs input x in
round 158). This tests whether multi-scale EMA works as well on
h-side as on x-side.

## Mechanism

For each step, augment the hidden state h with K EMAs at
different β values::

    # At step t, for each k in 0..K-1:
    ema_h_k,t = beta_k * ema_h_k,t-1 + (1 - beta_k) * h_t
    # Build augmented hidden state:
    aug_h_t = f_concat(h_t, ema_h_1,t, ..., ema_h_K,t)  # variants

K=2: β ∈ {0.7, 0.95} (short, long)
K=3: β ∈ {0.5, 0.9, 0.99} (short, medium, long)

This is the CROSS of round 158 (multi-scale input) and round 159
(h-side EMA). It tests if multi-scale h-side strictly improves
over single-scale h-side.

### Variants (mirror round 158)

1. **mbh_diff_2**: aug_h = [h_t, ema_h_1 - h_t, ema_h_2 - h_t] (3H input)
2. **mbh_concat_2**: aug_h = [h_t, ema_h_1, ema_h_2] (3H input)
3. **mbh_diff_3**: aug_h = [h_t, ema_h_1 - h_t, ema_h_2 - h_t, ema_h_3 - h_t] (4H input)
4. **mbh_concat_3**: aug_h = [h_t, ema_h_1, ema_h_2, ema_h_3] (4H input)

## Hypotheses

- **H1 (multi-scale h-side outperforms single)**: mbh_* improves
  on eh_diff (round 159, -77% structured).
- **H2 (K=3 diff best)**: mbh_diff_3 is the best variant
  (consistent with round 158 finding).
- **H3 (stable)**: training is stable across all variants.

## Bench plan (30 cells)

5 conds × 3 datasets × 2 seeds × 30 epochs:
- Datasets: sin_irr, structured_irr, random_irr (D=2, T=32,
  missing_rate=0.3).
- Compare to round 159 eh_diff (-77% structured) and round 158
  mb_diff_3 (-65% structured).

## Success criteria

- **STRICTLY POSITIVE** if all 3 datasets improve vs CfC.
- **TARGET-DEPENDENT** if 1-2 datasets improve and 1 worsens.
- **NEGATIVE** if any dataset degrades ≥30% (catastrophic).

## Files

- `lnn/core/multi_beta_h_cfc.py` (~280 lines)
- `tests/test_multi_beta_h_cfc.py` (~30 tests)
- `scripts/bench_multi_beta_h_cfc.py` (30-cell bench)
- `docs/research/2026-06-15_multi_beta_h_cfc_report.md`
- `memory/lnn-round-160-multi-beta-h-cfc.md`
