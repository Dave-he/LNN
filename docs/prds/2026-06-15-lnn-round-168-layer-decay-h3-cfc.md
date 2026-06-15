# PRD #10-130 — Round 168 LayerDecay-H3-CfC (Kh=3 with REVERSE β)

**Date**: 2026-06-15
**Round**: 168
**Audit context (91-167)**: 40 strictly positive + 17 target-dep +
35 negatives = 92 mechanism classes.

## Motivation

Round 167 established that per-layer β REVERSE schedule (slow at
low layers, fast at high layers) achieves **sin -69% NEW BEST** —
breaking the round 165 barrier of -63%.

The REVERSE winner `ld_reverse_k5` used **Kh=2** (2 hidden-side
time-scales: 0.99 slow, 0.5 fast). Round 165 found that higher
**Kx (input-side time-scales)** helps — Kx=5 was best.

The natural test: does **higher Kh (hidden-side time-scales)**
under REVERSE schedule compound the benefits? Specifically:
- Kh=3 with REVERSE schedule ∈ [0.99, 0.75, 0.5]
- Kh=4 with REVERSE schedule ∈ [0.99, 0.83, 0.66, 0.5]
- Different schedules (linear, reverse, wider)

## Mechanism

Same as round 167's LayerDecayCfCStackedNetwork, but with
**Kh ∈ {3, 4}** instead of Kh=2::

    For layer l in 0..L-1:
        # Per-layer β schedule for h-side (Kh values):
        beta_h_l_k = schedule(l, beta_min, beta_max, K_h, mode)
        # Same as round 167:
        beta_x_k,d = sigmoid(beta_x_k_raw[d])  # shape [Kx, D]
        # Input-side EMAs (per-feature, per-sample):
        ema_x_k,t[b,d] = beta_x_k,d * ema_x_k,t-1[b,d] + (1 - beta_x_k,d) * x_t[b,d]
        # Per-layer hidden-state EMAs (Kh values):
        ema_h_k,t[b,d] = beta_h_l_k * ema_h_k,t-1[b,d] + (1 - beta_h_l_k) * h_t[b,d]
        z_t = cat(aug_x_t, aug_h_t)
        h_t = CfC(z_t)

### Variants

1. **ld_reverse_h3_k5**: 3-layer, Kx=5, Kh=3, REVERSE β ∈ [0.99, 0.75, 0.5]
2. **ld_reverse_h4_k5**: 3-layer, Kx=5, Kh=4, REVERSE β ∈ [0.99, 0.83, 0.66, 0.5]
3. **ld_reverse_h3_wider**: 3-layer, Kx=5, Kh=3, REVERSE β ∈ [0.999, 0.7, 0.3]
4. **ld_reverse_h3_k6**: 3-layer, Kx=6, Kh=3, REVERSE β ∈ [0.99, 0.75, 0.5]
5. **ld_reverse_h3_h2**: 3-layer, Kx=5, Kh=3, REVERSE β ∈ [0.95, 0.85, 0.7]
6. **ld_constant_h3**: 3-layer, Kx=5, Kh=3, constant β ∈ {0.7, 0.85, 0.95} (control)

## Hypotheses

- **H1 (Kh=3 helps)**: ld_reverse_h3_k5 beats ld_reverse_k5 (sin -69%)
  on sin.
- **H2 (wider range helps)**: ld_reverse_h3_wider beats ld_reverse_k5
  on sin.
- **H3 (Kh=3 helps structured)**: ld_reverse_h3_k5 recovers
  some of the structured regression from round 167 (which was -82%
  vs round 165 -91%).
- **H4 (constant Kh=3 baseline)**: ld_constant_h3 (control) is
  WORSE than ld_reverse_h3_k5 — confirming the REVERSE schedule
  helps even with Kh=3.

## Bench plan (36-48 cells)

6 conds × 3 datasets × 2 seeds × 30 epochs (36 cells)
+ 1 control × 3 datasets × 2 seeds × 30 epochs (6 cells)
Total: 42 cells

## Success criteria

- **STRICTLY POSITIVE** if a cond beats round 167's ld_reverse_k5
  on BOTH sin AND structured.
- **SIN IMPROVEMENT** if a cond beats sin -69% (round 167).
- **NEGATIVE** if any dataset degrades ≥30%.

## Files

- `lnn/core/layer_decay_h3_cfc.py` (re-export of round 167 with
  Kh=3/4 factory functions)
- `tests/test_layer_decay_h3_cfc.py` (~8 tests)
- `scripts/bench_layer_decay_h3_cfc.py` (42-cell bench)
- `docs/research/2026-06-15_layer_decay_h3_cfc_report.md`
- `memory/lnn-round-168-layer-decay-h3-cfc.md`

## Why this is interesting

1. **Compounds the round 167 win** — REVERSE schedule × higher Kh
   is an orthogonal combination
2. **Tests the schedule hypothesis** — does REVERSE work because
   of the β contrast or because of the absolute β values?
3. **Cheap** — just changes Kh from 2 to 3/4, no new architecture
4. **Addresses round 167 structured regression** — Kh=3 might
   recover the structured loss
