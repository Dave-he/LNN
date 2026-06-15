# Round 176 — LearnedBetaPS+KxLadder-CfC — Research Report

**Date**: 2026-06-16
**Round**: 176
**Branch**: master
**Audit context (91-175)**: 43 strictly positive + 18 target-dep +
38 negatives = 99 mechanism classes.

## TL;DR

**NEGATIVE for Round 176**: Kx ladder doesn't beat SOTA. Smaller
Kx (3_3_3) helps sin (-73% vs control -46%), larger Kx (7_7_7)
helps structured (-92%), but neither reaches round 171 SOTA.

## What was tested

**Per-scale learnable β + Kx ladder** — different Kx (input-side
EMA scales) per layer. Round 173 tested Kh ladder (h-side);
round 176 tests Kx ladder (x-side).

## Bench (42 cells: 7 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | Kx ladder | sin_irr | structured_irr | n_params |
|------|-----------|---------|----------------|----------|
| lbps_kxl_5_5_5 (control) | [5,5,5] | 0.0143±0.0051 (-46%) | 0.0137±0.0017 (-89%) | 19241 |
| **lbps_kxl_3_3_3** | [3,3,3] | **0.0073±0.0009 (-73%)** | 0.0155±0.0008 (-87%) | **15971** |
| lbps_kxl_7_7_7 | [7,7,7] | 0.0101±0.0032 (-62%) | 0.0093±0.0012 (-92%) | 22511 |
| lbps_kxl_3_5_7 | [3,5,7] | 0.0139±0.0001 (-48%) | 0.0122±0.0012 (-90%) | 20585 |
| lbps_kxl_7_5_3 | [7,5,3] | 0.0095±0.0008 (-64%) | 0.0263±0.0005 (-78%) | 17897 |
| lbps_kxl_3_5_5 | [3,5,5] | 0.0162±0.0002 (-39%) | 0.0114±0.0005 (-91%) | 19047 |
| lbps_kxl_7_5_5 | [7,5,5] | 0.0095±0.0005 (-64%) | 0.0125±0.0059 (-90%) | 19435 |

## Cross-round (best in class)

| Round | Mechanism | sin | structured |
|-------|-----------|-----|------------|
| 171 | lb_ps_h2_75 (Kh=2) | **-76%** | -91% |
| 171 | lb_ps_h5_75 (Kh=5) | -71% | -92% |
| 173 | lbps_khl_2_3_5 (Kh ladder) | -53% | **-93%** |
| 176 | lbps_kxl_3_3_3 (Kx ladder) | -73% | -87% |
| 176 | lbps_kxl_7_7_7 (Kx ladder) | -62% | -92% |

**No NEW BESTS** in round 176.

## Hypotheses revisited

- **H1 (Kx ladder helps)**: PARTIAL. Smaller Kx (3) helps sin,
  larger Kx (7) helps structured. Ladder shapes ([3,5,7],
  [7,5,3]) don't help.
- **H2 (Kx=5 constant optimal)**: REJECTED. Kx=3 wins sin, Kx=7
  wins structured.
- **H3 (Kx ladder helps structured)**: REJECTED. Ladder shapes
  don't help structured.

## Why Kx ladder doesn't beat SOTA

### 1. Kx and Kh have different roles
- Kx captures input patterns (how many input time-scales)
- Kh captures hidden state time-scales

Round 173's Kh ladder wins structured because Kh directly
controls hidden state dynamics. Kx is a less direct lever.

### 2. Kx choice is independent
Kx=3 wins sin (less noise, more direct), Kx=7 wins structured
(more input time-scales for multi-mode). Ladder doesn't help
because the model doesn't need different Kx per layer.

### 3. Kx=3 is a near-miss for sin
lbps_kxl_3_3_3 sin 0.0073 (-73%) is 3pp from SOTA -76% (round
171 lb_ps_h2_75 with Kx=5). So smaller Kx helps but doesn't
beat smaller Kh.

### 4. Kx=7 ties round 171 SOTA on structured
lbps_kxl_7_7_7 structured 0.0093 (-92%) ties round 171
lb_ps_h5_75 structured 0.0095 (-92%) — within noise.

## Pattern reinforced (43 + 18 + 39 = 100 mechanism classes)

- **43 strictly positive** (unchanged)
- **18 target-dep** (unchanged)
- **39 negatives** (UP from 38, round 176 adds 1)
- Total: **100 mechanism classes** (up from 99)

## Critical implementation details

1. **LearnedBetaPSKxlCfCStackedNetwork** — wraps learned_beta_ps
   cells with per-layer Kx
2. **Kx_ladder parameter** — list of num_layers Kx values
3. **Same closed-form CfC** as round 171
4. **Tests** — 12/12 pass

## Why this is a useful negative

1. **Confirms Kx < Kh** — varying Kx per layer is less impactful
   than varying Kh
2. **Identifies Kx sweet spots** — Kx=3 for sin, Kx=7 for
   structured, Kx=5 is a compromise
3. **Saves future investigation** — no need to try Kx ladder with
   other variants
4. **Useful: Kx=3 is a "free win" for sin** — 13% smaller model
   with 27pp sin improvement vs Kx=5

## Files

- `lnn/core/learned_beta_ps_kxl_cfc.py` (~200 lines)
- `tests/test_learned_beta_ps_kxl_cfc.py` (12 tests)
- `scripts/bench_learned_beta_ps_kxl_cfc.py` (42-cell bench)
- `results/bench_learned_beta_ps_kxl_cfc.json`
- `docs/prds/2026-06-16-lnn-round-176-learned-beta-ps-kxl-cfc.md`
- `docs/research/2026-06-16_learned_beta_ps_kxl_cfc_report.md`

## Next ideas

1. **lb_ps + FAME-MoE** — combine with FAME top-K routing
2. **lb_ps + input-conditioned β** — β varies with input
3. **lb_ps with cosine annealing of β**
4. **lb_ps + per-layer learnable Kx AND Kh** — combined ladder
5. **Kx=3 with Kh=2 (combined small)** — synergy?
6. **Kx=7 with Kh=5 (combined large)** — synergy?

**Why:** Round 176 is NEGATIVE. Kx ladder doesn't beat SOTA.
Smaller Kx helps sin, larger Kx helps structured, ladder shapes
don't add value.

**How to apply:** **Use Kx=3 for sin (free 27pp vs Kx=5)** and
**Kx=7 for structured** if you want to optimize per dataset.
Otherwise use Kx=5. Audit becomes 100.
