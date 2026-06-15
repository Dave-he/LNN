# PRD #10-148 — Round 186 — LearnedBetaPS+LN+FFT-CfC

**Date**: 2026-06-16
**Round**: 186
**Branch**: master
**Audit context (91-185)**: 45 strictly positive + 18 target-dep
+ 46 negatives = 109 mechanism classes.

## Background

5 NEGATIVEs in a row (rounds 181-185), all trying to extend
lb_ps SOTA on the same axis (add a mechanism on top of
round 180's `lbps_ln_khl_*`). The lb_ps variant space is
exhausted. **Pivot to a different mechanism class**.

## Goal

Add **FFT magnitude as additional input features**. This is
fundamentally different from lb_ps variants — it introduces
**frequency-domain information** that the EMA-based
features miss.

## Mechanism

**FFT input features** (per-feature, per-timestep):
```
x_clean = nan_to_num(x, nan=0)        # [B, T, D]
x_fft = abs(rfft(x_clean, dim=1))     # [B, T//2+1, D]
x_fft_pad = pad(x_fft, [0, T//2])     # [B, T, D] (zero-pad T dim)
x_aug = cat([x, x_fft_pad], dim=-1)   # [B, T, 2D]
# Pass to lb_ps_ln pipeline with input_size=2D
```

## Hypotheses

- **H1 (positive)**: FFT magnitude captures frequency
  patterns that EMAs miss → improvement
- **H2 (negative)**: FFT adds noise (magnitude can be
  dominated by noise/random fluctuations) → no help
- **H3 (mixed)**: helps periodic (sin) but not non-periodic
  (random)

## Configurations (3 conds)

1. `lbps_lnfft_h3_75`: Kh=3
2. `lbps_lnfft_h2_75`: Kh=2
3. `lbps_lnfft_h5_75`: Kh=5

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

## Why this direction is novel for the audit

- All previous rounds (171-185) operate in the time domain
  (EMA, CfC, LN, skip, residual)
- FFT adds a **frequency-domain view** that the time-domain
  models cannot easily replicate
- Frequency features are known to help periodic data (sin)

## Files

- `lnn/core/learned_beta_ps_ln_fft_cfc.py` (~250 lines)
- `tests/test_learned_beta_ps_ln_fft_cfc.py` (12+ tests)
- `scripts/bench_learned_beta_ps_ln_fft_cfc.py` (18-cell bench)
- `docs/research/2026-06-16_learned_beta_ps_ln_fft_cfc_report.md`

**Why:** Pivot from lb_ps variants (5 NEGATIVEs in a row)
to a fundamentally different mechanism class (frequency
domain).

**How to apply:** If SP, add to SOTA. If NEGATIVE, log
the negative. If TD, document dataset dependence.
