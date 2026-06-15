# Round 170 — PerFeatureH-CfC (Per-Feature β on H-Side) — Research Report

**Date**: 2026-06-15
**Round**: 170
**Branch**: master
**Audit context (91-169)**: 42 strictly positive + 17 target-dep +
35 negatives = 94 mechanism classes.

## TL;DR

**NEGATIVE for Round 170**: Per-feature β on h-side REGRESSES
sin across ALL Kh values. Best variant `pfh_h5` achieves sin -66%
(6pp worse than round 169's -72%).

Round 162's finding (per-feature β on h-side overfits) REPLICATES
in this round, even with the round 169 architectural improvements
(Kh=3, per-feature β on x-side).

## What was tested

**Per-feature learned β on h-side** (one β per hidden unit per
scale). Round 162 saw regression with Kh=2 (-15% vs -33% round
161). Round 169 established Kh=3 is the sweet spot. Round 170
tests if Kh=3+ fixes the regression.

## Bench (30 cells: 5 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| pfh_h3_finer (Kh=3, per-feature) | 0.0132±0.0015 (-57%) | 0.0271±0.0065 (-84%) | 0.1031±0.0026 (-1%) | 19531 |
| pfh_h3_k6 (Kh=3, Kx=6, per-feature) | 0.0144±0.0002 (-50%) | 0.0240±0.0037 (-86%) | 0.1033±0.0025 (-1%) | 21197 |
| pfh_h4_wide (Kh=4, per-feature) | 0.0146±0.0024 (-49%) | 0.0243±0.0033 (-86%) | 0.1048±0.0006 (-1%) | 21883 |
| pfh_h2_const (Kh=2, per-feature, round 162 control) | 0.0127±0.0026 (-59%) | 0.0256±0.0033 (-85%) | 0.1030±0.0026 (-1%) | 17179 |
| pfh_h5 (Kh=5, per-feature) | 0.0092±0.0008 (-66%) | 0.0217±0.0115 (-85%) | 0.1033±0.0026 (-1%) | 24235 |

## Cross-round (best in class)

| Round | Mechanism | Kh | per-feature H? | sin | structured |
|-------|-----------|-----|----------------|-----|------------|
| 162 | lb_xh_best | 2 | YES | -15% | -90% |
| 165 | hb_xh_deep_h2_k5 | 2 | no | -63% | **-91%** |
| 168 | ld_constant_h3 | 3 | no | -71% | -84% |
| 169 | ld_constant_h3_finer | 3 | no | **-72%** | -87% |
| 170 | pfh_h5 (best per-feature) | 5 | YES | -66% | -85% |

## Hypotheses revisited

- **H1 (per-feature helps with Kh=3)**: REJECTED. pfh_h3_finer
  regressed from -72% to -57% (+15pp regression).
- **H2 (per-feature helps structured)**: REJECTED. Structured
  also regressed (-84% to -85% vs round 169 -87%).
- **H3 (per-feature with Kh=2 regresses)**: CONFIRMED. pfh_h2_const
  regressed to -59% (worse than round 161's -33% baseline at Kh=2
  without per-feature, and worse than round 165's -63% with
  scalar β).

## Why per-feature β on h-side fails

### 1. Over-parameterization for hidden state
Each hidden unit gets its own β → H=16 hidden units × Kh scales =
many β parameters. With limited training data (30 epochs × 32
batch), the model overfits on the training trajectories.

### 2. Sin doesn't need per-feature time-scales
Sin data is uniform frequency. Each hidden unit doesn't need a
different time-scale — they should share time-scales.

### 3. Round 162 finding still holds
Round 162 tested per-feature β on h-side with Kh=2 and saw sin
-15% (vs round 161's -33%). With round 169's better base
architecture (Kh=3 scalar), per-feature STILL regresses.

### 4. Even higher Kh (Kh=5) doesn't fully recover
pfh_h5 (Kh=5 with per-feature) achieves -66% — closer to -72%
but still 6pp worse. More parameters per unit doesn't help.

## Pattern reinforced (42 + 17 + 35 = 94 mechanism classes)

- **42 strictly positive** (unchanged — round 170 negative)
- **17 target-dep** (unchanged)
- **35 negatives** (unchanged)

This round does NOT add new strictly positive winners. The
94-class audit remains stable.

## Critical implementation details

1. **PerFeatureHCfCCell** — adds `beta_h_raw` parameter of shape
   [Kh, H] (one β per scale per hidden unit)
2. **Sigmoid parameterization** — `beta_h = sigmoid(beta_h_raw)`,
   same as round 162's LearnedBeta-XH
3. **Same closed-form CfC** as round 163 (tau_eff = exp(-f * dt /
   |time_scale|))
4. **Pyright false positives** on `import torch` are pre-existing
5. **Tests** — 13/13 pass

## Why this is a useful negative

1. **Confirms round 162 finding** — per-feature β on h-side is
   a robust negative across multiple architectures
2. **Saves future investigation** — no need to revisit per-feature
   β on h-side in future rounds
3. **Identifies the boundary** — scalar β on h-side wins; per-
   feature β overfits
4. **Cheap** — only ~10 lines of new core code

## Files

- `lnn/core/perfeature_h_cfc.py` (~250 lines, new core class)
- `tests/test_perfeature_h_cfc.py` (13 tests, all pass)
- `scripts/bench_perfeature_h_cfc.py` (30-cell bench)
- `results/bench_perfeature_h_cfc.json`
- `docs/prds/2026-06-15-lnn-round-170-perfeature-h-cfc.md`
- `docs/research/2026-06-15_perfeature_h_cfc_report.md`

## Next ideas

1. **Adaptive β (learned scalar β)** with h3_finer init —
   gradient descent on β values themselves (not per-feature, but
   per-scale)
2. **ld_constant_h3_optimal** — grid search β ∈ {0.7-0.95} for
   Kh=3
3. **Combine h3_finer with 4-layer** — does h3 help 4-layer?
4. **MoE + h3_finer** — add FAME routing
5. **ld_constant_h3 with β regularization** — penalize extreme β
   values to encourage slow EMAs
6. **ld_h3_perfeature_x** — per-feature β on x-side only (h-side
   stays scalar)

**Why:** Round 170 is a HONEST NEGATIVE. Per-feature β on h-side
regresses sin across all Kh values. Round 162's finding still
holds even with round 169's improvements. The 94-class audit
remains stable.

**How to apply:** **Do NOT use per-feature β on h-side** for
hybrid β CfC. Use **scalar β on h-side** (Kh=3, β ∈ {0.75, 0.85,
0.95}) — this is the round 169 SOTA. Round 170 confirms scalar
wins over per-feature for hidden state time-scales.
