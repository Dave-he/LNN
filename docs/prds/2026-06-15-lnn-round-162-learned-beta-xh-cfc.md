# PRD #10-124 — Round 162 LearnedBeta-XH-CfC (Per-Feature β on Stacked XH)

**Date**: 2026-06-15
**Round**: 162
**Audit context (91-161)**: 24 strictly positive + 17 target-dep +
33 negatives = 74 mechanism classes.

## Motivation

Rounds 156-161 (6 rounds, 10 strictly positive winners) explored
EMA-style mechanisms. The audit reveals:
- Round 156 (EMA-X, K=1, x, scalar β): 17th positive (-42% structured)
- Round 157 (LearnedBeta, K=1, x, per-feature β): 18th positive (-63% structured)
- Round 158 (MultiBeta, K=2/K=3, x, fixed β): 19th/20th positive (-60%/-65%)
- Round 159 (EMA-H, K=1, h, scalar β): 21st positive (-77% structured)
- Round 160 (MultiBeta-H, K=2/K=3, h, fixed β): 22nd positive (-32% sin)
- Round 161 (Stacked-XH, K=3 x + K=2 h, fixed β): 23rd/24th positive
  (-33% sin, **-86% structured**)

Round 161 was a **MAJOR BREAKTHROUGH**: combining x-side and
h-side achieves BOTH best sin AND best structured.

The natural next step is the **ULTIMATE CROSS-PRODUCT**: combine
all 6 winners with per-feature learned β. This is the most
powerful mechanism yet, integrating:
- Multi-scale input (round 158 K=3)
- Multi-scale hidden (round 160 K=2)
- Per-feature learned β (round 157)
- Stacked X+H (round 161)

## Mechanism

For each step, augment BOTH input x and hidden state h with
K parallel EMAs, but the β values are PER-FEATURE LEARNED::

    # Per-feature learned β (round 157):
    beta_x_k,d = sigmoid(beta_x_k_raw[d])  # shape [Kx, D]
    beta_h_k,d = sigmoid(beta_h_k_raw[d])  # shape [Kh, H]

    # Input-side EMAs (round 158):
    ema_x_k,t[d] = beta_x_k,d * ema_x_k,t-1[d] + (1 - beta_x_k,d) * x_t[d]
    aug_x_t = [x_t, ema_x_1,t - x_t, ..., ema_x_Kx,t - x_t]

    # Hidden-state EMAs (round 160):
    ema_h_k,t[d] = beta_h_k,d * ema_h_k,t-1[d] + (1 - beta_h_k,d) * h_t[d]
    aug_h_t = [h_t, ema_h_1,t - h_t, ..., ema_h_Kh,t - h_t]

    # Combined:
    z_t = cat(aug_x_t, aug_h_t)

This is the **FULL CROSS-PRODUCT** of rounds 156-161.

### Variants (4 conds)

1. **lb_xh_diff_1_1**: Kx=1, Kh=1, per-feature learned β
2. **lb_xh_diff_3_2**: Kx=3, Kh=2, per-feature learned β
3. **lb_xh_concat_2_2**: Kx=2, Kh=2, concat mode (control)
4. **lb_xh_best**: Kx=3, Kh=2, both diff, per-feature learned β

## Hypotheses

- **H1 (per-feature β on stacked is best)**: lb_xh_best
  outperforms round 161's sx_xh_best (fixed β).
- **H2 (BREAKTHROUGH+)**: lb_xh_best achieves NEW BESTS on
  BOTH sin and structured (>33% sin, >86% structured).
- **H3 (training stable)**: per-feature β on multi-scale is
  stable (no catastrophic divergences).

## Bench plan (30 cells)

5 conds × 3 datasets × 2 seeds × 30 epochs:
- Datasets: sin_irr, structured_irr, random_irr (D=2, T=32,
  missing_rate=0.3).
- Compare to round 161 sx_xh_best (sin -33%, structured -86%).

## Success criteria

- **STRICTLY POSITIVE** if all 3 datasets improve vs CfC.
- **BREAKTHROUGH+** if a cond beats round 161 on BOTH sin AND
  structured.
- **NEGATIVE** if any dataset degrades ≥30% (catastrophic).

## Files

- `lnn/core/learned_beta_xh_cfc.py` (~310 lines)
- `tests/test_learned_beta_xh_cfc.py` (~30 tests)
- `scripts/bench_learned_beta_xh_cfc.py` (30-cell bench)
- `docs/research/2026-06-15_learned_beta_xh_cfc_report.md`
- `memory/lnn-round-162-learned-beta-xh-cfc.md`
