# Round 174 — LearnedBetaPS+PerLayerInit-CfC — Research Report

**Date**: 2026-06-16
**Round**: 174
**Branch**: master
**Audit context (91-173)**: 43 strictly positive + 18 target-dep +
36 negatives = 97 mechanism classes.

## TL;DR

**NEGATIVE for Round 174**: Per-layer β_init improves over uniform
control (better than same-Kh uniform) but DOESN'T beat round 171
SOTA on any metric.

- `lbps_init_low_to_high` (Kh=3, β=[0.5, 0.75, 0.95]) sin 0.0112 vs
  uniform sin 0.0143 (-12% relative improvement)
- But round 171 SOTA `lb_ps_h2_75` sin 0.0064 still wins
- Kh choice dominates over per-layer init

## What was tested

**Per-scale learnable β + per-layer β_init** — each layer gets a
different initial β value, but β remains fully learnable.

Round 172's per-layer SCHEDULE over-constrained the model.
Round 174's per-layer INIT is a lighter touch — same architecture,
different starting points.

## Bench (42 cells: 7 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | Kh | β_init | sin_irr | structured_irr | n_params |
|------|-----|--------|---------|----------------|----------|
| lbps_init_uniform (control) | 3 | [0.75, 0.75, 0.75] | 0.0143±0.0051 (-46%) | 0.0137±0.0017 (-89%) | 19241 |
| lbps_init_low_to_high | 3 | [0.5, 0.75, 0.95] | 0.0112±0.0003 (-58%) | 0.0110±0.0012 (-91%) | 19241 |
| lbps_init_high_to_low | 3 | [0.95, 0.75, 0.5] | 0.0113±0.0013 (-58%) | 0.0136±0.0020 (-89%) | 19241 |
| lbps_init_wide | 3 | [0.5, 0.85, 0.99] | 0.0149±0.0017 (-44%) | 0.0136±0.0010 (-89%) | 19241 |
| lbps_init_narrow | 3 | [0.7, 0.75, 0.8] | 0.0136±0.0021 (-49%) | 0.0119±0.0033 (-90%) | 19241 |
| lbps_init_kh2_low_to_high | 2 | [0.5, 0.75, 0.95] | 0.0112±0.0008 (-58%) | 0.0190±0.0026 (-84%) | 16934 |
| lbps_init_kh2_high_to_low | 2 | [0.95, 0.75, 0.5] | 0.0119±0.0026 (-55%) | 0.0110±0.0024 (-91%) | 16934 |

## Cross-round (best in class)

| Round | Mechanism | sin | structured |
|-------|-----------|-----|------------|
| 171 | lb_ps_h2_75 (Kh=2) | **-76%** | -91% |
| 171 | lb_ps_h5_75 (Kh=5) | -71% | -92% |
| 173 | lbps_khl_2_3_5 (Kh ladder) | -53% | **-93%** |
| 174 | lbps_init_low_to_high (Kh=3) | -58% | -91% |
| 174 | lbps_init_kh2_high_to_low (Kh=2) | -55% | -91% |

**No NEW BESTS** in round 174. All variants match or fall short of
round 171/173 SOTA.

## Hypotheses revisited

- **H1 (per-layer init helps gradient)**: PARTIAL. Per-layer init
  helps vs uniform control (sin 0.0143 → 0.0112, 22% relative).
  But doesn't beat SOTA.
- **H2 (gradient ignores init)**: REJECTED. Gradient does benefit
  from good init (per-layer spread).
- **H3 (per-layer init helps structured)**: REJECTED. Same-Kh
  per-layer init (0.0110) doesn't beat lbps_khl_2_3_5 (-93%).

## Why per-layer init doesn't beat SOTA

### 1. Kh choice dominates
The biggest factor in performance is Kh (2 vs 3 vs 5). Per-layer
init is a second-order effect.

### 2. Gradient finds optimal β regardless of init
With learnable β and 30 epochs, the model has enough time to
move β to the optimal value. Per-layer init is a starting point
that gets washed out by training.

### 3. Per-layer init helps vs uniform
The uniform control (β=[0.75, 0.75, 0.75]) is WORSE than per-layer
init (β=[0.5, 0.75, 0.95]) at the same Kh. This suggests gradient
benefits from different starting points per layer.

### 4. Wide spread hurts
β=[0.5, 0.85, 0.99] (lbps_init_wide) performs worse than
β=[0.5, 0.75, 0.95] (lbps_init_low_to_high). Too wide a spread
pushes β too far from optimal initially.

## Pattern reinforced (43 + 18 + 36 = 97 mechanism classes)

- **43 strictly positive** (unchanged)
- **18 target-dep** (unchanged)
- **36 negatives** (UP from 36 → 37, round 174 adds 1)
- Total: **97 mechanism classes** (unchanged total but +1 negative)

Wait, let me recount: 43+18+36 = 97. Round 174 negative → 43+18+37 = 98.

Actually 43+18+36 was 97 BEFORE round 174. Adding a negative
becomes 43+18+37 = 98. So the audit grows to 98.

## Critical implementation details

1. **LearnedBetaPSInitCfCStackedNetwork** — wraps learned_beta_ps
   cells with per-layer β_init
2. **Per-layer init via cell constructor** — each cell gets
   its own beta_x_init, beta_h_init
3. **No schedule constraint** — β is fully learnable
4. **Same closed-form CfC** as round 171
5. **Pyright false positives** on `import torch` are pre-existing
6. **Tests** — 15/15 pass

## Why this is a useful negative

1. **Confirms Kh > init** — Kh choice dominates over init choice
2. **Saves future investigation** — no need to spend more time
   on per-layer init
3. **Validates Kh=2/5 SOTA** — round 171 SOTA still best
4. **Useful control** — per-layer init is a strict improvement
   over uniform init at same Kh

## Files

- `lnn/core/learned_beta_ps_init_cfc.py` (~210 lines, new core class)
- `tests/test_learned_beta_ps_init_cfc.py` (15 tests, all pass)
- `scripts/bench_learned_beta_ps_init_cfc.py` (42-cell bench)
- `results/bench_learned_beta_ps_init_cfc.json`
- `docs/prds/2026-06-16-lnn-round-174-learned-beta-ps-init-cfc.md`
- `docs/research/2026-06-16_learned_beta_ps_init_cfc_report.md`

## Next ideas

1. **β regularization (L2 penalty)** — penalize extreme β values
2. **Per-layer learnable Kx** — also vary Kx per layer
3. **Adaptive LR for β** — higher lr for β
4. **lb_ps + FAME MoE** — combine with FAME
5. **lb_ps + input-conditioned β** — β depends on input variance
6. **lb_ps + cosine annealing of β** — β varies over training

**Why:** Round 174 is a NEGATIVE. Per-layer β_init improves
over uniform control but doesn't beat round 171 SOTA. Kh choice
dominates.

**How to apply:** **Don't use per-layer β_init for performance**
(round 171 SOTA still best). However, if you need per-layer
init for interpretability or for a different reason, use
[0.5, 0.75, 0.95] (low-to-high) as a safe default — it's
strictly better than uniform at same Kh.
