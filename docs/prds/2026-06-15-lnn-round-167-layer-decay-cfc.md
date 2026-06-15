# PRD #10-129 — Round 167 LayerDecay-CfC (Per-Layer β Schedule)

**Date**: 2026-06-15
**Round**: 167
**Audit context (91-166)**: 39 strictly positive + 17 target-dep +
35 negatives = 91 mechanism classes.

## Motivation

Rounds 156-166 (11 rounds, 23+ strictly positive winners)
explored EMA-based augmentations on the X (input) and H (hidden
state) sides. Round 165 (hb_xh_deep_h2_k5, 3-layer, Kx=5, Kh=2)
achieved the SOTA double-best: sin -63% AND structured -91%.

Round 166 showed that **depth saturates at 3 layers** — 4-layer
REGRESSES sin by 26pp (from -63% to -37%) while only marginally
improving structured (from -91% to -92%). This suggests **all
3 layers use the SAME time-scale** (β ∈ {0.7, 0.95}) which may
be suboptimal — different layers may need different time-scales.

**Hypothesis**: Layer 0 needs **fast β** (low-level, captures
short-term input fluctuations), deeper layers need **slow β**
(high-level, smooth abstractions). A linear β schedule across
layers could unlock depth benefits.

## Mechanism

LayerDecay-CfC: same as round 165 HybridBeta-XH, but each
layer `l` has its **own β for h-side EMA**::

    For layer l in 0..L-1:
        # Per-layer β schedule for h-side:
        beta_h_l = beta_h_min + l * (beta_h_max - beta_h_min) / (L-1)
        # Same as round 165:
        beta_x_k,d = sigmoid(beta_x_k_raw[d])  # shape [Kx, D]
        # Input-side EMAs (per-feature):
        ema_x_k,t[d] = beta_x_k,d * ema_x_k,t-1[d] + (1 - beta_x_k,d) * x_t[d]
        # Per-layer hidden-state EMAs:
        ema_h_k,t[d] = beta_h_l_k * ema_h_k,t-1[d] + (1 - beta_h_l_k) * h_t[d]
        z_t = cat(aug_x_t, aug_h_t)
        h_t = CfC(z_t)

Three schedules tested:
1. **constant**: all layers use β ∈ {0.7, 0.95} (round 165 baseline)
2. **linear**: β_l_k = β_min_k + l * (β_max_k - β_min_k) / (L-1)
   where β_min = 0.5, β_max = 0.99
3. **reverse_linear**: β_l_k = β_max_k - l * (β_max_k - β_min_k) / (L-1)
   where β_min = 0.5, β_max = 0.99

### Variants (3-layer + Kx=5, Kh=2)

1. **ld_constant**: round 165 baseline (control)
2. **ld_linear_k5**: 3-layer, Kx=5, Kh=2, linear β ∈ [0.5, 0.99]
3. **ld_reverse_k5**: 3-layer, Kx=5, Kh=2, reverse linear β ∈ [0.99, 0.5]
4. **ld_linear_slow**: 3-layer, Kx=5, Kh=2, linear β ∈ [0.7, 0.99]
5. **ld_linear_fast**: 3-layer, Kx=5, Kh=2, linear β ∈ [0.3, 0.9]
6. **ld_relu**: 3-layer, Kx=5, Kh=2, linear β ∈ [0.7, 0.95] (round 165 range)

## Hypotheses

- **H1 (linear β helps)**: ld_linear_k5 beats round 165 baseline
  on at least one dimension.
- **H2 (β schedule matters)**: linear vs reverse linear vs
  constant give different results (i.e., the schedule isn't
  trivial).
- **H3 (depth unlock)**: 4-layer with linear β unlocks depth
  benefits (tested as bonus).

## Bench plan (18-30 cells)

3-layer: 6 conds × 3 datasets × 2 seeds × 30 epochs (36 cells)
4-layer: 2 conds × 3 datasets × 2 seeds × 30 epochs (12 cells)
Total: 48 cells

## Success criteria

- **STRICTLY POSITIVE** if a 3-layer cond beats round 165 on
  BOTH sin AND structured.
- **DEPTH UNLOCK** if 4-layer with linear β beats round 166
  hb_xh_4layer_h2_3x (structured -92%).
- **NEGATIVE** if any dataset degrades ≥30%.

## Files

- `lnn/core/layer_decay_cfc.py` (new core class)
- `tests/test_layer_decay_cfc.py` (~10 tests)
- `scripts/bench_layer_decay_cfc.py` (48-cell bench)
- `docs/research/2026-06-15_layer_decay_cfc_report.md`
- `memory/lnn-round-167-layer-decay-cfc.md`

## Why this is interesting

1. **New orthogonal dimension**: per-layer β schedule has NOT
   been tested before in our 91-class audit
2. **Addresses round 166 finding**: depth saturation may be due
   to uniform β across layers
3. **Biology-inspired**: cortex hierarchy has different time-
   scales at different depths (fast V1, slow PFC)
4. **Cheap**: just (L × K_h) scalar params, no new arch
