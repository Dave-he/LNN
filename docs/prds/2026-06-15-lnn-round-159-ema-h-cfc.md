# PRD #10-121 — Round 159 EMA-H-CfC (Hidden State EMA Augmentation)

**Date**: 2026-06-15
**Round**: 159
**Audit context (91-158)**: 20 strictly positive + 15 target-dep +
28 negatives = 63 mechanism classes.

## Motivation

Rounds 155-158 (4 rounds, 6 strictly positive winners) all
augmented the **input x** with various signals:
- Round 155 (DELTA-CfC): h deltas — 15th, 16th positive
- Round 156 (EMA-X-CfC): scalar β=0.9 EMA — 17th positive
- Round 157 (LearnedBeta-CfC): per-feature learnable β — 18th positive
- Round 158 (MultiBeta-CfC): K=2/3 fixed β — 19th, 20th positive

The natural next step is to apply the EMA idea to a different
signal: the **hidden state h**. This tests whether the multi-scale
EMA pattern transfers to a different signal.

## Mechanism

For each step, augment the hidden state h with an EMA of h::

    ema_h_t = beta * ema_h_{t-1} + (1 - beta) * h_t
    aug_h_t = f_concat(h_t, ema_h_t)  # 4 variants

This is structurally different from rounds 155-158 (which augment
input). It tests: does multi-scale hidden state EMA help?

### Variants (mirror round 156)

1. **eh_concat**: aug_h = [h_t, ema_h_t] (2H input, +100% params)
2. **eh_gate**: aug_h = alpha * h_t + (1 - alpha) * ema_h_t, learned alpha
3. **eh_diff**: aug_h = [h_t, ema_h_t - h_t] (2H input, high-pass signal)
4. **eh_ema_only**: aug_h = ema_h_t only (H input, no h) — control

### Key design choices

1. **β = 0.9** (fixed hyperparameter, start simple)
2. **EMA of h, not x** — different signal from rounds 156-158
3. **Same 4 variants as round 156** — for direct comparability
4. **Augmentation is INTERIOR** — affects g_branch and h_branch
   within the cell, not the input projection

## Hypotheses

- **H1 (EMA of h is also useful)**: aug_h = [h, ema_h] helps
  CfC, similar to aug_x = [x, ema_x] in round 156.
- **H2 (diff still best)**: eh_diff is the best variant.
- **H3 (stable training)**: ema_h is stable.
- **H4 (interior vs input aug)**: interior aug (this round) is
  complementary to input aug (rounds 156-158).

## Bench plan (30 cells)

5 conds × 3 datasets × 2 seeds × 30 epochs:
- Datasets: sin_irr, structured_irr, random_irr (D=2, T=32,
  missing_rate=0.3, same as rounds 156-158).
- Compare directly to round 156 (input EMA, scalar β=0.9).

## Success criteria

- **STRICTLY POSITIVE** if all 3 datasets improve vs CfC.
- **TARGET-DEPENDENT** if 1-2 datasets improve and 1 worsens.
- **NEGATIVE** if any dataset degrades ≥30% (catastrophic).

## Files

- `lnn/core/ema_h_cfc.py` (~280 lines)
- `tests/test_ema_h_cfc.py` (~30 tests)
- `scripts/bench_ema_h_cfc.py` (30-cell bench)
- `docs/research/2026-06-15_ema_h_cfc_report.md`
- `memory/lnn-round-159-ema-h-cfc.md`
