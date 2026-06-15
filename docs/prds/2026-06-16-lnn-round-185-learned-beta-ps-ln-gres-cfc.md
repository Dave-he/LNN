# PRD #10-147 — Round 185 — LearnedBetaPS+LN+GatedResidual-CfC

**Date**: 2026-06-16
**Round**: 185
**Branch**: master
**Audit context (91-184)**: 45 strictly positive + 18 target-dep
+ 45 negatives = 108 mechanism classes.

## Background

SOTA is round 180:
- `lbps_ln_khl_2_5_2`: sin 0.0033
- `lbps_ln_khl_5_3_2`: structured 0.0024

4 NEGATIVEs in a row (181, 182, 183, 184). Round 184 (raw
residual skip) catastrophically regressed because it
**replaced** the CfC step entirely, losing:
- Time-scale control
- Bounded output (h_branch is tanh-bounded)
- Non-linear interpolation between old and new

## Goal

Try **gated residual ON TOP OF** the CfC step (not replacing
it). This preserves the closed-form CfC interpolation while
adding a learnable correction.

## Mechanism

**Variant 1 (scalar gate, simpler)**:
```python
# CfC step (unchanged):
h_cfc = tau_eff * g + (1.0 - tau_eff) * h_branch
# NEW (round 185): gated residual ON TOP
alpha = sigmoid(self.alpha_raw)  # learnable scalar per cell
h_residual = self.residual_proj(z_norm)  # Linear(aug_total, H)
h_new = h_cfc + alpha * h_residual
```

Init `alpha_raw = log(0.1 / 0.9)` so sigmoid ≈ 0.1 at start
(behaves like round 180 with small perturbation).

## Hypotheses

- **H1 (positive)**: gated residual adds well-conditioned
  correction while preserving CfC stability
- **H2 (negative)**: residual is redundant with CfC's
  tau_eff
- **H3 (mixed)**: helps structured (preserves slow
  components) but hurts sin (over-emphasis on h_t history)

## Configurations (3 conds)

1. `lbps_lngr_h3_75`: Kh=3, residual_init=0.1
2. `lbps_lngr_h2_75`: Kh=2, residual_init=0.1
3. `lbps_lngr_h5_75`: Kh=5, residual_init=0.1

All with Kx=5, β=0.75, num_layers=3, alpha_init=0.1.

## Datasets

- sin_irr (D=2, T=32, missing_rate=0.3)
- structured_irr (D=2, T=32, missing_rate=0.3)
- random_irr (D=2, T=32, missing_rate=0.3)

## Bench

18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs.

## Success criteria

- **STRICTLY POSITIVE**: at least one cond beats round 180
  SOTA on at least one dataset.
- **NEGATIVE**: no cond beats round 180.
- **TARGET-DEP**: helps one dataset, hurts another.

## Files

- `lnn/core/learned_beta_ps_ln_gres_cfc.py` (~250 lines)
- `tests/test_learned_beta_ps_ln_gres_cfc.py` (12+ tests)
- `scripts/bench_learned_beta_ps_ln_gres_cfc.py` (18-cell bench)
- `docs/research/2026-06-16_learned_beta_ps_ln_gres_cfc_report.md`

**Why:** Round 184 showed that REPLACING the CfC step
destroys stability. Gated residual ON TOP of CfC
preserves stability.

**How to apply:** If SP, add to SOTA. If NEGATIVE, log
the negative result. If TD, document dataset dependence.
