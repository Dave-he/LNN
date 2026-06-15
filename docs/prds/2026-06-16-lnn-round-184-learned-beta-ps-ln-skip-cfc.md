# PRD #10-146 — Round 184 — LearnedBetaPS+LN+Skip-CfC

**Date**: 2026-06-16
**Round**: 184
**Branch**: master
**Audit context (91-183)**: 45 strictly positive + 18 target-dep
+ 44 negatives = 107 mechanism classes.

## Background

SOTA is round 180:
- `lbps_ln_khl_2_5_2`: sin 0.0033
- `lbps_ln_khl_5_3_2`: structured 0.0024

3 NEGATIVEs in a row (181, 182, 183). Round 183 confirmed that
**Input LN (round 179) is the right position for LayerNorm**
— post-LN loses magnitude info.

## Goal

Try **skip connection (residual)** on top of the SOTA Input LN
base. The residual can:
1. Preserve gradient flow across timesteps
2. Compensate for LN normalizing scale (h_t magnitude preserved)
3. Add correction signal without rewriting h_t entirely

## Hypotheses

- **H1 (positive)**: residual + LN improves both metrics
  - LN normalizes z (no scale issue) → residual adds
    well-conditioned correction
  - h_t = h_t + Residual(LN(z)) preserves h_t magnitude
- **H2 (negative)**: residual dilutes LN effect
  - h_t = h_t + small_delta → LN's normalization is
    dominated by h_t history
  - No help, possibly hurt
- **H3 (mixed)**: helps structured, hurts sin
  - Structured has long-term patterns → residual preserves
    slow components
  - Sin has phase-sensitive dynamics → residual over-
    emphasizes h_t history

## Mechanism

**Skip (residual) variant of round 180's lbps_ln**:

```python
# Round 180 (baseline):
h_new = tau_eff * g + (1.0 - tau_eff) * h_branch

# Round 184 (skip):
h_cfc = tau_eff * g + (1.0 - tau_eff) * h_branch  # CfC step
h_residual = self.residual_proj(self.layer_norm(z))  # LN(z) → residual
h_new = h_t + h_residual  # residual update (à la ResNet)
```

OR with gate (à la Highway/Transformer FFN):
```python
h_new = h_t + alpha * self.residual_proj(self.layer_norm(z))
```

where `alpha` is a learnable scalar (init to small value, e.g. 0.1)
or 1.0.

## Configurations to test (3 conds)

1. `lbps_lns_h3_75`: Kh=3, residual_proj=Linear(hidden→hidden)
2. `lbps_lns_h2_75`: Kh=2, residual_proj=Linear(hidden→hidden)
3. `lbps_lns_h5_75`: Kh=5, residual_proj=Linear(hidden→hidden)

All with Kx=5, β=0.75, num_layers=3.

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

- `lnn/core/learned_beta_ps_ln_skip_cfc.py` (~250 lines)
- `tests/test_learned_beta_ps_ln_skip_cfc.py` (12+ tests)
- `scripts/bench_learned_beta_ps_ln_skip_cfc.py` (18-cell bench)
- `docs/research/2026-06-16_learned_beta_ps_ln_skip_cfc_report.md`

**Why:** Skip connections are a known stability mechanism
not yet tried on top of round 180's SOTA Input LN base.

**How to apply:** If SP, add to the SOTA. If NEGATIVE, log
the negative result. If TD, document dataset dependence.
