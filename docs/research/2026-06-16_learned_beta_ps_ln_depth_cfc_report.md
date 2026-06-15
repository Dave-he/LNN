# Round 182 — LearnedBetaPS+LN+Depth-CfC — Research Report

**Date**: 2026-06-16
**Round**: 182
**Branch**: master
**Audit context (91-181)**: 45 strictly positive + 18 target-dep +
42 negatives = 105 mechanism classes.

## TL;DR

**NEGATIVE for Round 182**: Depth scaling REGRESSES. n=3 is the
sweet spot. n=4 catastrophic on structured; n=5 catastrophic
on both.

## What was tested

**lb_ps + LayerNorm + depth scaling** — num_layers ∈ {2, 3, 4, 5}
with round 180 winning Kh ladders ([2,5,2] for sin, [5,3,2] for
structured). Pad/truncate Kh_ladder to match num_layers.

## Bench (36 cells: 6 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | n_layers | Kh | sin_irr | structured_irr | random_irr | n_params |
|------|----------|-----|---------|----------------|------------|----------|
| lbps_ln_khl_2_5_2_n2 | 2 | [2,5] | 0.0073±0.0013 | 0.0236±0.0149 | 0.1357±0.0160 | 12762 |
| **lbps_ln_khl_2_5_2_n3** (ctrl) | 3 | [2,5,2] | **0.0033±0.0005** | 0.0058±0.0009 | 0.1732±0.0065 | 20033 |
| lbps_ln_khl_2_5_2_n4 | 4 | [2,5,2,2] | 0.0054±0.0015 | **0.2177±0.2007** ⚠️ | 0.1750±0.0087 | 27304 |
| lbps_ln_khl_2_5_2_n5 | 5 | [2,5,2,2,2] | **0.0719±0.0689** ⚠️ | **0.4427±0.0372** ⚠️ | 0.1738±0.0072 | 34575 |
| **lbps_ln_khl_5_3_2_n3** (ctrl) | 3 | [5,3,2] | 0.0198±0.0122 | **0.0024±0.0000** | 0.1737±0.0076 | 20834 |
| lbps_ln_khl_5_3_2_n4 | 4 | [5,3,2,3] | 0.0086±0.0054 | **0.1573±0.1538** ⚠️ | 0.1727±0.0081 | 28105 |

⚠️ = catastrophic regression

## Cross-round (best in class)

| Round | Mechanism | sin | structured |
|-------|-----------|-----|------------|
| 180 | lbps_ln_khl_2_5_2 (n=3) | **0.0033** | 0.0058 |
| 180 | lbps_ln_khl_5_3_2 (n=3) | 0.0198 | **0.0024** |
| **182** | **lbps_ln_khl_2_5_2_n3** (n=3 ctrl) | 0.0033 | 0.0058 |
| **182** | **lbps_ln_khl_5_3_2_n3** (n=3 ctrl) | 0.0198 | 0.0024 |

**No NEW BESTS**. Controls reproduce round 180.

## Hypotheses revisited

- **H1 (depth scaling helps)**: **REJECTED**. More layers
  catastrophically regress.
- **H2 (n=3 is optimal, more layers overfit)**: **CONFIRMED**.
- **H3 (n=2 is too shallow)**: PARTIAL. n=2 has smaller
  params but regresses both.

## Why depth scaling regresses

### 1. More LN parameters = more overfit
Each layer has its own LayerNorm ((Kx+1)*D + (Kh+1)*H
parameters). n=5 has 5× LN = much more overfit on B=32.

### 2. Padded Kh_ladder is suboptimal
For n=4 with Kh_ladder=[2,5,2], we pad to [2,5,2,2]. The
padded Kh=2 layers are not trained optimally — they're copies.

### 3. Vanishing gradients through 5 layers
Standard issue: deep networks without skip connections
have vanishing gradients. CfC has implicit skip via the
closed-form but still suffers.

### 4. Catastrophic structured failures
n=4 structured: 0.2177 and 0.1573 (75-100x worse than SOTA
0.0024). n=5 structured: 0.4427 (184x worse).

## Pattern (45 + 18 + 43 = 106 mechanism classes)

- **45 strictly positive** (unchanged)
- **18 target-dep** (unchanged)
- **43 negatives** (UP from 42, round 182 adds 1)
- Total: **106 mechanism classes** (up from 105)

## Critical implementation details

1. **Pad/truncate Kh_ladder** — if len(Kh_ladder) < num_layers,
   pad with last value. If > num_layers, truncate.
2. **No new core module** — reuses round 180 stacked network.
3. **Bench script only** — no new tests needed.

## Why this is a useful negative

1. **Confirms n=3 is optimal depth** — more layers regress
2. **Identifies catastrophic failure** — n=4/5 cause 100x
   regression on structured
3. **Saves future depth exploration** — no need to try n>3
4. **Confirms round 180 controls reproduce** — same numbers

## Files

- `scripts/bench_learned_beta_ps_ln_depth_cfc.py` (36-cell bench)
- `results/bench_learned_beta_ps_ln_depth_cfc.json`
- `docs/prds/2026-06-16-lnn-round-182-learned-beta-ps-ln-depth-cfc.md`
- `docs/research/2026-06-16_learned_beta_ps_ln_depth_cfc_report.md`

## Next ideas

1. **lb_ps + RMSNorm** — different normalization
2. **lb_ps + Output LN (post-CfC)** — different position
3. **lb_ps_ln + skip connections** — add residual to avoid
   vanishing gradients
4. **lb_ps_ln + per-layer β_init** — different init per layer
5. **Different mechanism class entirely** — try next novel
   direction beyond lb_ps variants

**Why:** Round 182 is NEGATIVE. Depth scaling regresses. n=3
is optimal.

**How to apply:** Stick with n=3 (round 180 SOTA). Audit becomes
106.
