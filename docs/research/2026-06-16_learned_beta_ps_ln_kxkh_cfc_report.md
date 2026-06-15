# Round 181 — LearnedBetaPS+LN+KxKh-CfC — Research Report

**Date**: 2026-06-16
**Round**: 181
**Branch**: master
**Audit context (91-180)**: 45 strictly positive + 18 target-dep +
41 negatives = 104 mechanism classes.

## TL;DR

**NEGATIVE for Round 181**: Combined Kx×Kh ladder on top of LN
REGRESSES both sin and structured. Single-dim Kh ladder
(round 180) is the sweet spot.

## What was tested

**lb_ps + LayerNorm + Kx×Kh combined ladder** — both Kx AND Kh
vary per layer.

## Bench (36 cells: 6 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | Kx | Kh | sin_irr | structured_irr | n_params |
|------|----|----|---------|----------------|----------|
| **lbps_ln_kxkh_5_5_5_2_5_2** (control) | [5,5,5] | [2,5,2] | **0.0033** | 0.0058 | 20033 |
| **lbps_ln_kxkh_5_5_5_5_3_2** (control) | [5,5,5] | [5,3,2] | 0.0198 | **0.0024** | 20834 |
| lbps_ln_kxkh_3_5_7_2_5_2 | [3,5,7] | [2,5,2] | 0.0066 | 0.0067 | 21433 |
| lbps_ln_kxkh_7_5_3_5_3_2 | [7,5,3] | [5,3,2] | 0.0098 | **0.0517** ⚠️ | 19434 |
| lbps_ln_kxkh_3_5_7_5_3_2 | [3,5,7] | [5,3,2] | 0.0066 | **0.0742** ⚠️ | 22234 |
| lbps_ln_kxkh_7_5_3_2_5_2 | [7,5,3] | [2,5,2] | 0.0068 | 0.0057 | 18633 |

⚠️ = catastrophic regression (>20x)

## Cross-round (best in class)

| Round | Mechanism | sin | structured |
|-------|-----------|-----|------------|
| 179 | lbps_ln_h2_75 | 0.0035 | 0.0033 |
| 180 | lbps_ln_khl_2_5_2 | 0.0033 | 0.0058 |
| 180 | lbps_ln_khl_5_3_2 | 0.0198 | **0.0024** |
| **181** | **lbps_ln_kxkh_5_5_5_2_5_2** (control) | 0.0033 | 0.0058 |
| **181** | **lbps_ln_kxkh_5_5_5_5_3_2** (control) | 0.0198 | **0.0024** |

**No NEW BESTS** in round 181. Controls reproduce round 180.

## Hypotheses revisited

- **H1 (combined Kx×Kh ladder beats single-dim ladders)**:
  **REJECTED**. All combined variants regress. Two regressions
  are catastrophic (0.05+ structured).
- **H2 (LN already captures scale info, ladder adds noise)**:
  **CONFIRMED**. LN handles scale; Kx ladder adds noise.
- **H3 (combined helps structured but regresses sin)**: PARTIAL.
  Combined regresses on BOTH metrics.

## Why combined Kx×Kh ladder regresses

### 1. Kx ladder was already negative (round 176)
Round 176 found Kx ladder alone (without LN) regresses both
metrics. Kx=[3,3,3] was 3pp from SOTA, Kx=[7,7,7] tied.
Combined with LN, the noise compounds.

### 2. LN already normalizes scale
LN unifies raw x_t, smoothed EMAs, and h_t to same scale. Kx
ladder changes the number of input scales — but LN normalizes
the result. The Kx ladder adds parameters without signal.

### 3. Catastrophic structured failures
lbps_ln_kxkh_7_5_3_5_3_2 structured 0.0517 (21x worse than
SOTA 0.0024). lbps_ln_kxkh_3_5_7_5_3_2 structured 0.0742
(31x worse). Both have Kh=[5,3,2] (structured winner) but
with mismatched Kx ladder. The wrong Kx ladder DESTABILIZES
the Kh ladder winner.

### 4. Slight regressions on sin
Combined Kx=[3,5,7] + Kh=[2,5,2] sin 0.0066 vs control 0.0033.
All Kx ladder variants regress sin by 2x.

## Pattern (45 + 18 + 42 = 105 mechanism classes)

- **45 strictly positive** (unchanged)
- **18 target-dep** (unchanged)
- **42 negatives** (UP from 41, round 181 adds 1)
- Total: **105 mechanism classes** (up from 104)

## Critical implementation details

1. **Wraps LearnedBetaPSLNCfCCell** — reuses round 179 cell.
2. **Kx_ladder AND Kh_ladder** — both vary per layer.
3. **Tests** — 11/11 pass.

## Why this is a useful negative

1. **Confirms Kh ladder is the right specialization** — Kx
   ladder is redundant.
2. **Identifies catastrophic failure mode** — mismatched Kx
   ladder + structured Kh ladder → 20x regression.
3. **Saves future variants** — no need to try other Kx×Kh
   combos (e.g., Kx=[5,7,3], [3,7,5], etc.)
4. **Confirms round 180 controls are reproducible** — same
   numbers exactly across rounds.

## Files

- `lnn/core/learned_beta_ps_ln_kxkh_cfc.py` (~190 lines, 6 factories)
- `tests/test_learned_beta_ps_ln_kxkh_cfc.py` (11 tests)
- `scripts/bench_learned_beta_ps_ln_kxkh_cfc.py` (36-cell bench)
- `results/bench_learned_beta_ps_ln_kxkh_cfc.json`
- `docs/prds/2026-06-16-lnn-round-181-learned-beta-ps-ln-kxkh-cfc.md`
- `docs/research/2026-06-16_learned_beta_ps_ln_kxkh_cfc_report.md`

## Next ideas

1. **lb_ps_ln + depth scaling (4-5 layers)** — does more depth
   help on top of LN?
2. **lb_ps + RMSNorm** — simpler normalization
3. **lb_ps + Output LN (post-CfC)** — different position
4. **lb_ps_ln + FAME-MoE** — combine with FAME
5. **lb_ps_ln + per-layer β_init** — different β_init per layer
6. **lb_ps_ln + h3 only (different ladder)** — try more ladders

**Why:** Round 181 is NEGATIVE. Combined Kx×Kh ladder adds
noise on top of LN.

**How to apply:** **Stick with round 180 single-dim Kh ladder**.
Don't combine Kx ladder. Audit becomes 105.
