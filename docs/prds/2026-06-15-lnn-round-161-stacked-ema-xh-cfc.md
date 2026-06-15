# PRD #10-123 — Round 161 Stacked-EMA-XH-CfC (Input + Hidden State EMA)

**Date**: 2026-06-15
**Round**: 161
**Audit context (91-160)**: 22 strictly positive + 17 target-dep +
30 negatives = 69 mechanism classes.

## Motivation

Rounds 156-160 (5 rounds, 8 strictly positive winners) explored
EMA-style mechanisms on either input x or hidden state h. The
audit reveals:
- Round 156 (EMA-X, K=1, x): 17th positive (-42% structured)
- Round 157 (LearnedBeta, K=1, x): 18th positive (-63% structured)
- Round 158 (MultiBeta, K=2/K=3, x): 19th/20th positive (-60%/-65%)
- Round 159 (EMA-H, K=1, h): 21st positive (-77% structured BEST)
- Round 160 (MultiBeta-H, K=2/K=3, h): 22nd positive (-32% sin BEST)

The natural next step is the **CROSS-PRODUCT OF X AND H**:
apply BOTH input-side EMA AND hidden-state EMA simultaneously.
This tests if the two mechanisms are complementary.

**Key insight**: The two mechanisms operate on different signals
(input x = raw observation; h = recurrent state). Stacking them
gives the model access to:
- X-EMA: smoothed input (recent observation context)
- H-EMA: smoothed hidden state (recent recurrent context)

**The trade-off discovered**:
- H-side helps structured (best -77% ever) and now sin (best -32% ever)
- X-side helps structured (best -65% ever)
- Combining them may give BEST of BOTH: -32% sin AND -77% structured!

## Mechanism

For each step, augment BOTH input x and hidden state h with
EMAs::

    # Input-side EMAs (round 156/158):
    ema_x_k,t = beta_x_k * ema_x_k,t-1 + (1 - beta_x_k) * x_t
    aug_x_t = [x_t, ema_x_1,t, ..., ema_x_Kx,t]  # Kx EMAs

    # Hidden-state EMAs (round 159/160):
    ema_h_k,t = beta_h_k * ema_h_k,t-1 + (1 - beta_h_k) * h_t
    aug_h_t = [h_t, ema_h_1,t, ..., ema_h_Kh,t]  # Kh EMAs

    # Combined:
    z_t = cat(aug_x_t, aug_h_t)  # 4D-5D or higher input

This is the FULL STACK of all 5 EMA mechanisms from rounds 156-160.

### Variants (4 conds, mirror round 158/160)

1. **sx_xh_diff_1_1**: aug_x = [x, ema_x_h] (Kx=1, β=0.9),
   aug_h = [h, ema_h-h] (Kh=1, β=0.9) — single-scale both
2. **sx_xh_diff_3_2**: aug_x = [x, ema_x_1-x, ema_x_2-x, ema_x_3-x]
   (Kx=3, β ∈ {0.5, 0.9, 0.99}),
   aug_h = [h, ema_h_1-h, ema_h_2-h] (Kh=2, β ∈ {0.7, 0.95})
3. **sx_xh_concat_2_2**: aug_x = [x, ema_x_1, ema_x_2] (Kx=2, β ∈ {0.7, 0.95}),
   aug_h = [h, ema_h_1, ema_h_2] (Kh=2, β ∈ {0.7, 0.95})
4. **sx_xh_best**: use round 158's mb_diff_3 (Kx=3) +
   round 160's mbh_diff_2 (Kh=2) — the two best variants combined

## Hypotheses

- **H1 (X+H is complementary)**: combining x-side and h-side
  gives BEST of BOTH worlds (best sin AND best structured).
- **H2 (best_x + best_h)**: round 158 mb_diff_3 + round 160
  mbh_diff_2 should achieve -32% sin AND -77% structured.
- **H3 (training stable)**: combined EMA mechanisms don't
  destabilize training.

## Bench plan (30 cells)

5 conds × 3 datasets × 2 seeds × 30 epochs:
- Datasets: sin_irr, structured_irr, random_irr (D=2, T=32,
  missing_rate=0.3).
- Compare to round 159 eh_diff (-77% structured) and round 160
  mbh_diff_2 (-32% sin).

## Success criteria

- **STRICTLY POSITIVE** if all 3 datasets improve vs CfC.
- **TARGET-DEPENDENT** if 1-2 datasets improve and 1 worsens.
- **NEGATIVE** if any dataset degrades ≥30% (catastrophic).
- **BREAKTHROUGH** if a single cond achieves BOTH -32% sin AND
  -77% structured.

## Files

- `lnn/core/stacked_ema_xh_cfc.py` (~280 lines)
- `tests/test_stacked_ema_xh_cfc.py` (~30 tests)
- `scripts/bench_stacked_ema_xh_cfc.py` (30-cell bench)
- `docs/research/2026-06-15_stacked_ema_xh_cfc_report.md`
- `memory/lnn-round-161-stacked-ema-xh-cfc.md`
