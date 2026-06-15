# PRD #10-131 — Round 169 LayerDecay-H4-CfC (Kh=4 with constant β)

**Date**: 2026-06-15
**Round**: 169
**Audit context (91-168)**: 41 strictly positive + 17 target-dep +
35 negatives = 93 mechanism classes.

## Motivation

Round 168 established that **Kh=3 with constant β** beats Kh=2
with constant β (sin -71% vs -63% round 165). The Kh dimension
(more hidden-side time-scales) is the winning axis.

Natural follow-up: does **Kh=4** help even more? And does
**Kx=6** (more input-side time-scales) help when combined with
the best Kh=3 config?

## Mechanism

Same as round 168's LayerDecay-CfC, but with **Kh=4** or
**Kx=6** instead of Kh=3, Kx=5::

    For layer l in 0..L-1:
        # Constant schedule (all layers get same betas_h):
        beta_h_k = betas_h[k]  # Kh values
        # Same as round 167:
        beta_x_k,d = sigmoid(beta_x_k_raw[d])  # shape [Kx, D]
        # Input-side EMAs (per-feature, per-sample):
        ema_x_k,t[b,d] = beta_x_k,d * ema_x_k,t-1[b,d] + (1 - beta_x_k,d) * x_t[b,d]
        # Hidden-state EMAs (Kh values per layer, Kh=4 here):
        ema_h_k,t[b,d] = beta_h_k * ema_h_k,t-1[b,d] + (1 - beta_h_k) * h_t[b,d]
        z_t = cat(aug_x_t, aug_h_t)
        h_t = CfC(z_t)

### Variants

1. **ld_constant_h4_default**: 3-layer, Kx=5, Kh=4, β ∈ {0.6, 0.75, 0.85, 0.95}
2. **ld_constant_h4_wide**: 3-layer, Kx=5, Kh=4, β ∈ {0.5, 0.7, 0.85, 0.99}
3. **ld_constant_h4_narrow**: 3-layer, Kx=5, Kh=4, β ∈ {0.8, 0.85, 0.9, 0.95}
4. **ld_constant_h3_k6**: 3-layer, Kx=6, Kh=3, β ∈ {0.7, 0.85, 0.95}
5. **ld_constant_h3_wider**: 3-layer, Kx=5, Kh=3, β ∈ {0.6, 0.8, 0.99}
6. **ld_constant_h3_finer**: 3-layer, Kx=5, Kh=3, β ∈ {0.75, 0.85, 0.95}
7. **ld_constant_h5**: 3-layer, Kx=5, Kh=5, β ∈ {0.5, 0.7, 0.85, 0.95, 0.99}

## Hypotheses

- **H1 (Kh=4 helps)**: ld_constant_h4_default beats round 168
  ld_constant_h3 (sin -71%).
- **H2 (Kx=6 helps)**: ld_constant_h3_k6 beats ld_constant_h3
  on at least one dimension.
- **H3 (wider β range)**: ld_constant_h4_wide beats
  ld_constant_h4_default on sin.
- **H4 (Kh=5 saturates)**: ld_constant_h5 does NOT improve
  over Kh=4 (saturation).

## Bench plan (42 cells)

7 conds × 3 datasets × 2 seeds × 30 epochs (42 cells)

## Success criteria

- **STRICTLY POSITIVE** if a cond beats round 168's -71% sin.
- **DOUBLE POSITIVE** if a cond beats BOTH -71% sin AND round
  165's -91% structured.
- **NEGATIVE** if any dataset degrades ≥30%.

## Files

- `lnn/core/layer_decay_h4_cfc.py` (re-export)
- `tests/test_layer_decay_h4_cfc.py` (~8 tests)
- `scripts/bench_layer_decay_h4_cfc.py` (42-cell bench)
- `docs/research/2026-06-15_layer_decay_h4_cfc_report.md`
- `memory/lnn-round-169-layer-decay-h4-cfc.md`

## Why this is interesting

1. **Tests Kh scaling** — Kh=2 → Kh=3 (round 168) → Kh=4 → Kh=5
2. **Tests Kx scaling** — Kx=5 → Kx=6 with best Kh
3. **Tests β range sensitivity** — narrow/wide/default
4. **Cheap** — just changes Kh, Kx, and betas_h
