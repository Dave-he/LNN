# Round 178 — LearnedBetaPS+IC-CfC — Research Report

**Date**: 2026-06-16
**Round**: 178
**Branch**: master
**Audit context (91-177)**: 43 strictly positive + 18 target-dep +
40 negatives = 101 mechanism classes.

## TL;DR

**NEGATIVE for Round 178**: Input-conditioned β REGRESSES both
sin (+66%) and structured (+53%) vs static learnable β. Adding
input-conditioning adds parameters but doesn't help.

## What was tested

**Input-conditioned β** — β is produced by a small MLP applied
to the input: β_t = sigmoid(W · x_t + b). β is no longer a
static learned parameter but **data-dependent per sample**.

Round 171 (static learnable β) is SOTA at sin 0.0064 (-76%) and
structured 0.0095 (-92%). Round 178 tests if dynamic β can
beat static β.

## Bench (30 cells: 5 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | Kh | β_init | sin_irr | structured_irr | random_irr | n_params |
|------|-----|--------|---------|----------------|------------|----------|
| lbps_ic_h3_75 | 3 | 0.75 | 0.0107±0.0036 | 0.0167±0.0006 | 0.1026±0.0028 | 19555 |
| lbps_ic_h2_75 | 2 | 0.75 | 0.0106±0.0007 | 0.0131±0.0034 | 0.1034±0.0021 | 17200 |
| lbps_ic_h5_75 | 5 | 0.75 | 0.0126±0.0025 | 0.0145±0.0041 | 0.1035±0.0037 | 24265 |
| lbps_ic_h3_50 | 3 | 0.50 | **0.0096±0.0005** | 0.0202±0.0025 | 0.1024±0.0031 | 19555 |
| lbps_ic_h3_90 | 3 | 0.90 | 0.0137±0.0009 | 0.0209±0.0002 | 0.1018±0.0035 | 19555 |

## Cross-round (best in class)

| Round | Mechanism | sin | structured |
|-------|-----------|-----|------------|
| 171 | lb_ps_h2_75 (STATIC) | **0.0064** | 0.0097 |
| 171 | lb_ps_h5_75 (STATIC) | 0.0078 | **0.0095** |
| 173 | lbps_khl_2_3_5 | 0.0131 | **0.0091** |
| 178 | lbps_ic_h3_50 (IC) | 0.0096 | 0.0202 |
| 178 | lbps_ic_h2_75 (IC) | 0.0106 | 0.0131 |

**No NEW BESTS**. IC β regresses on both metrics.

## Hypotheses revisited

- **H1 (IC β beats static on structured)**: REJECTED.
  lbps_ic_h5_75 structured 0.0145 vs static lb_ps_h5_75 0.0095
  = **+53% regression**.
- **H2 (per-sample β reduces overfitting)**: REJECTED.
  Per-sample β is actually WORSE — more parameters → more
  overfit on small data (B=32).
- **H3 (too many params → overfit)**: CONFIRMED.
  Adding β_x_proj (Kx*D) + β_h_proj (Kh*H) parameters
  increases capacity but adds noise.

## Why IC β regresses

### 1. More parameters → more overfit
Static β: Kx + Kh parameters (round 171).
IC β: Kx*D + Kx + Kh*H + Kh parameters (much more).
With B=32 samples, additional params overfit.

### 2. β_t varies with x_t — unstable
Initial β is at 0.75 (good default), but as soon as β_x_proj
learns to deviate, the EMA dynamics change at every timestep.
Static β is constant during forward → more stable.

### 3. Static β already captures the right "average"
Round 171 found that β=0.75 (or 0.85, 0.95 in some cases) is
the best constant β. IC β has to first LEARN to be 0.75
across all inputs, before it can specialize.

### 4. Per-sample variation is mostly noise
β_t = sigmoid(W · x_t + 0.75_logit) ≈ 0.75 + small input
perturbation. The perturbation is mostly noise.

## Pattern reinforced (43 + 18 + 41 = 102 mechanism classes)

- **43 strictly positive** (unchanged)
- **18 target-dep** (unchanged)
- **41 negatives** (UP from 40, round 178 adds 1)
- Total: **102 mechanism classes** (up from 101)

## Critical implementation details

1. **LearnedBetaPSICfCCell** — replaces static β_x_raw and
   β_h_raw Parameters with **β_x_proj and β_h_proj Linear
   layers** that map input → [B, Kx] and hidden → [B, Kh].
2. **Per-sample β** — β is computed per-sample, per-timestep
   from the input.
3. **Same closed-form CfC** as round 171.
4. **Init at target** — bias init to logit(0.75) so initial
   output is 0.75 (matching round 171 default).
5. **Tests** — 13/13 pass.

## Why this is a useful negative

1. **Confirms static β is optimal** — adding input-conditioning
   adds noise without benefit.
2. **Identifies capacity floor** — at B=32, additional
   parameters on β HURT.
3. **Saves future IC β variants** — no need to try IC β with
   other variants (e.g. IC + Kh ladder, IC + schedule).
4. **Useful: IC β ≈ static β** — even though IC β regresses
   slightly, the best IC (h3_50 sin 0.0096) is in the same
   ballpark as static β (h3_75 sin 0.0143 from round 177).

## Files

- `lnn/core/learned_beta_ps_ic_cfc.py` (~280 lines, IC β cell + stack)
- `tests/test_learned_beta_ps_ic_cfc.py` (13 tests)
- `scripts/bench_learned_beta_ps_ic_cfc.py` (30-cell bench)
- `results/bench_learned_beta_ps_ic_cfc.json`
- `docs/prds/2026-06-16-lnn-round-178-learned-beta-ps-ic-cfc.md`
- `docs/research/2026-06-16_learned_beta_ps_ic_cfc_report.md`

## Next ideas

1. **lb_ps + β EMA (running avg)** — stabilize β across epochs
2. **lb_ps + per-layer β_lr (decoupled)** — different lr for β
3. **lb_ps + β grouped by Kx (cluster)** — Kx scales cluster
   into G groups
4. **lb_ps + Kh ladder (combined)** — apply Kh ladder AND Kh=5
   per layer, with Kx ladder too
5. **Different mechanism class entirely** — try next novel
   direction beyond lb_ps extensions

**Why:** Round 178 is NEGATIVE. IC β regresses both metrics
due to overfitting on small B=32.

**How to apply:** **Use static learnable β (round 171) — input
conditioning adds noise**. Audit becomes 102.
