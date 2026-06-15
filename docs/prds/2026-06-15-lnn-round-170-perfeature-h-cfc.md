# PRD #10-132 — Round 170 PerFeatureH-CfC (Per-Feature β on H-Side)

**Date**: 2026-06-15
**Round**: 170
**Audit context (91-169)**: 42 strictly positive + 17 target-dep +
35 negatives = 94 mechanism classes.

## Motivation

Rounds 156-169 explored EMA-based augmentations with **scalar β**
on the h-side (all hidden units share the same time-scale). Round
169's SOTA `ld_constant_h3_finer` uses 3 scalar β values: {0.75,
0.85, 0.95}.

The question: does **per-feature β on h-side** (each hidden unit
gets its own β) provide additional benefit? Round 162 tested
this with Kh=2 and saw regression (-15% vs -33% round 161) — but
the model architecture has changed significantly since then.

**Hypothesis**: with Kh=3 and the right base β values, per-feature
β on h-side could provide a real breakthrough.

## Mechanism

Same as round 167's LayerDecayCfCStackedNetwork, but h-side β
values are **learned per-feature** (per hidden unit), not scalar::

    For each layer:
        # Per-feature learned β on h-side (NEW):
        beta_h_k,h = sigmoid(beta_h_raw[k, h])  # shape [Kh, H]
        # Per-feature learned β on x-side (round 163+):
        beta_x_k,d = sigmoid(beta_x_k_raw[k, d])  # shape [Kx, D]
        # Per-sample EMAs:
        ema_x_k,t[b,d] = beta_x_k,d * ema_x_k,t-1[b,d] + (1 - beta_x_k,d) * x_t[b,d]
        ema_h_k,t[b,h] = beta_h_k,h * ema_h_k,t-1[b,h] + (1 - beta_h_k,h) * h_t[b,h]
        z_t = cat(aug_x_t, aug_h_t)
        h_t = CfC(z_t)

### Variants (3-layer + Kx=5)

1. **pfh_h3_finer**: 3-layer, Kx=5, Kh=3, per-feature β on h-side,
   base β ∈ {0.75, 0.85, 0.95}
2. **pfh_h3_default**: 3-layer, Kx=5, Kh=3, per-feature β on h-side,
   base β ∈ {0.7, 0.85, 0.95}
3. **pfh_h3_k6**: 3-layer, Kx=6, Kh=3, per-feature β on h-side,
   base β ∈ {0.75, 0.85, 0.95}
4. **pfh_h4_wide**: 3-layer, Kx=5, Kh=4, per-feature β on h-side,
   base β ∈ {0.5, 0.7, 0.85, 0.99}
5. **pfh_h2_const**: 3-layer, Kx=5, Kh=2, per-feature β on h-side,
   base β ∈ {0.7, 0.95} (round 165 control with per-feature)

## Hypotheses

- **H1 (per-feature helps with Kh=3)**: pfh_h3_finer beats
  ld_constant_h3_finer (sin -72%) on at least one dimension.
- **H2 (per-feature helps structured)**: per-feature β
  recovers the structured regression seen in rounds 167-169.
- **H3 (per-feature with Kh=2 regresses again)**: pfh_h2_const
  confirms round 162's finding (regression).

## Bench plan (30 cells)

5 conds × 3 datasets × 2 seeds × 30 epochs (30 cells)

## Success criteria

- **STRICTLY POSITIVE** if a cond beats round 169's -72% sin.
- **DOUBLE POSITIVE** if a cond beats BOTH -72% sin AND round
  165's -91% structured (the elusive DOUBLE BEST).
- **NEGATIVE** if any dataset degrades ≥30%.

## Files

- `lnn/core/perfeature_h_cfc.py` (new core class)
- `tests/test_perfeature_h_cfc.py` (~10 tests)
- `scripts/bench_perfeature_h_cfc.py` (30-cell bench)
- `docs/research/2026-06-15_perfeature_h_cfc_report.md`
- `memory/lnn-round-170-perfeature-h-cfc.md`

## Why this is interesting

1. **Tests round 162's negative finding** — does Kh=3 fix the
   per-feature β regression?
2. **Orthogonal dimension** — per-feature β on h-side is NEW
   (not tested with Kh=3)
3. **Could unlock double-best** — per-feature might help BOTH
   sin and structured
4. **Cheap** — just adds [Kh, H] learnable params
