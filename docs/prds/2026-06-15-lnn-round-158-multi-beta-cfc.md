# PRD #10-120 — Round 158 MultiBeta-CfC (Multi-Scale EMA Augmentation)

**Date**: 2026-06-15
**Round**: 158
**Audit context (91-157)**: 18 strictly positive + 13 target-dep +
28 negatives = 59 mechanism classes.

## Motivation

Rounds 155-157 showed input augmentation is a winning pattern:
- Round 155 (DELTA-CfC): h deltas — 15th, 16th strictly positive
- Round 156 (EMA-X-CfC): scalar β=0.9 EMA — 17th strictly positive
- Round 157 (LearnedBeta-CfC): per-feature learnable β — 18th positive

The progression has been on a SINGLE β value. The natural next
step is **MULTI-SCALE EMA** — use MULTIPLE β values in parallel
to capture temporal context at different time-scales.

## Mechanism

For each input x, compute K parallel EMAs with different β values::

    # Initialize: ema_k = x_0 for each k
    # At step t:
    ema_k,t[d] = β_k · ema_k,t-1[d] + (1 - β_k) · x_t[d]
    aug_x_t = f_concat(x_t, ema_1,t, ema_2,t, ..., ema_K,t)

This is structurally different from:
- Round 156 (EMA-X): K=1, scalar β=0.9.
- Round 157 (LearnedBeta): K=1, per-feature learnable β.
- Round 129 (Multi-timescale ELM): multi-timescale but with ELM,
  found NEGATIVE.
- Round 76 (n_tau): multi-timescale τ in CfC recurrence, 7th winner.

MultiBeta extends single-β EMA to multi-scale EMA, providing
temporal context at multiple time-scales simultaneously.

### Variants (4 conds)

K=2: β ∈ {0.7, 0.95} (short, long)
K=3: β ∈ {0.5, 0.9, 0.99} (short, medium, long)

1. **mb_diff_2**: aug_x = [x, ema_1-x, ema_2-x] (3D input, K=2 high-pass)
2. **mb_concat_2**: aug_x = [x, ema_1, ema_2] (3D input, K=2 low-pass)
3. **mb_diff_3**: aug_x = [x, ema_1-x, ema_2-x, ema_3-x] (4D input, K=3 high-pass)
4. **mb_concat_3**: aug_x = [x, ema_1, ema_2, ema_3] (4D input, K=3 low-pass)

### Key design choices

1. **Fixed β values** — no learned parameters, just fixed decay rates.
2. **Multiple β in parallel** — captures multiple time-scales.
3. **Same 2 modes (diff/concat) as round 156** — for direct comparability.
4. **K=2 and K=3 tested** — see if more scales help.

## Hypotheses

- **H1 (multi-scale > single-scale)**: K=3 is strictly better than
  K=1 (round 156's ema_diff).
- **H2 (diff still best)**: mb_diff_3 is the best variant
  (consistent with rounds 156, 157).
- **H3 (stable training)**: fixed β values ensure stability.
- **H4 (more scales help structured)**: structured -63% (round
  157's lb_diff) can be further improved by adding more β values.

## Bench plan (30 cells)

5 conds × 3 datasets × 2 seeds × 30 epochs:
- Datasets: sin_irr, structured_irr, random_irr (D=2, T=32,
  missing_rate=0.3, same as rounds 156, 157).
- Confrim benchmark: CfC baseline from round 156.
- Compare directly to round 156 (scalar β) and round 157
  (per-feature learnable β).

## Success criteria

- **STRICTLY POSITIVE** if all 3 datasets improve vs CfC.
- **TARGET-DEPENDENT** if 1-2 datasets improve and 1 worsens.
- **NEGATIVE** if any dataset degrades ≥30% (catastrophic).
- **OUTPERFORMS round 157** if structured improves beyond -63%.

## Files

- `lnn/core/multi_beta_cfc.py` (~280 lines)
- `tests/test_multi_beta_cfc.py` (~30 tests)
- `scripts/bench_multi_beta_cfc.py` (30-cell bench)
- `docs/research/2026-06-15_multi_beta_cfc_report.md`
- `memory/lnn-round-158-multi-beta-cfc.md`
