# PRD #10-119 — Round 157 LearnedBeta-CfC (per-feature learnable β EMA)

**Date**: 2026-06-15
**Round**: 157
**Audit context (91-156)**: 17 strictly positive + 12 target-dep +
26 negatives = 55 mechanism classes.

## Motivation

Round 156 (EMA-X-CfC) showed that `ema_diff = ema - x` (high-pass
signal via EMA, β=0.9 fixed scalar) is a 17th STRICTLY POSITIVE
winner:
- sin -11%, structured -42%, random -1% (sin + structured improves).

The natural next step is to make β **per-feature and learnable**:
- Different features likely need different smoothing.
- A slow trend feature benefits from high β (long EMA window).
- A noisy feature benefits from low β (short EMA window).
- Per-feature β lets the model learn this automatically.

This is the inverse of FiLM 153: instead of learnable per-feature
γ, β for affine modulation, learnable per-feature β for EMA
smoothing.

## Mechanism

```
β ∈ R^D, per-feature learnable, initialized to 0.9
ema_t[d] = β[d] · ema_{t-1}[d] + (1 - β[d]) · x_t[d]
aug_x_t = f_concat(x_t, ema_t)  # 4 variants
```

### Variants (mirror round 156)

1. **lb_concat**: aug_x = [x, ema] (2D input, +34% params)
2. **lb_gate**: aug_x = α·x + (1-α)·ema, learned scalar α (D input)
3. **lb_diff**: aug_x = [x, ema - x] (2D input, +34% params)
4. **lb_ema_only**: aug_x = ema only (D input, no x) — control

### Key design choices

1. **β parameterized via sigmoid to keep β ∈ (0, 1)**:
   `β = sigmoid(β_raw)`, where β_raw is the unconstrained parameter.
   This ensures stability of the EMA recursion.
2. **Per-feature β (dim D)** — different features get different
   smoothing.
3. **Same 4 variants as round 156** — for direct comparability.
4. **Closed-form CfC solution unchanged**.

## Hypotheses

- **H1 (per-feature β > scalar β)**: per-feature β is more
  flexible and strictly better than scalar β=0.9.
- **H2 (diff still best)**: lb_diff is the best variant (same
  pattern as round 156).
- **H3 (stable training)**: per-feature β stays in (0, 1) due
  to sigmoid parameterization.
- **H4 (β adapts to feature)**: learned β values differ across
  features after training (audit via per-feature β distribution).

## Bench plan (30 cells)

5 conds × 3 datasets × 2 seeds × 30 epochs:
- Datasets: sin_irr, structured_irr, random_irr (D=2, T=32,
  missing_rate=0.3, same as round 156).
- Confrim benchmark: CfC baseline from round 156.
- Compare directly to round 156's ema_diff (β=0.9 scalar).

## Success criteria

- **STRICTLY POSITIVE** if all 3 datasets improve vs CfC.
- **TARGET-DEPENDENT** if 1-2 datasets improve and 1 worsens.
- **NEGATIVE** if any dataset degrades ≥30% (catastrophic).

## Files

- `lnn/core/learned_beta_cfc.py` (~280 lines)
- `tests/test_learned_beta_cfc.py` (~30 tests)
- `scripts/bench_learned_beta_cfc.py` (30-cell bench)
- `docs/research/2026-06-15_learned_beta_cfc_report.md`
- `memory/lnn-round-157-learned-beta-cfc.md`
