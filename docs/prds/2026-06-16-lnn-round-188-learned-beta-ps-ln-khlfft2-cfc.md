# PRD #10-150 — Round 188 — LearnedBetaPS+LN+Khl+FFT2-CfC

**Date**: 2026-06-16
**Round**: 188
**Branch**: master
**Audit context (91-187)**: 46 strictly positive + 19 target-dep
+ 46 negatives = 111 mechanism classes.

## Background

Round 187 (FFT + Kh ladder hybrid) was STRICTLY POSITIVE 46th:
- lbps_lnkhlfft_5_3_2 sin **0.0026** (21% better than SOTA
  0.0033) — new SOTA
- structured 0.0059 (2.5x worse than SOTA 0.0024) — TD

The hybrid helps sin (FFT captures dominant frequency) but
hurts structured (FFT destroys regime info).

## Goal

**Add phase features** to the FFT input encoder. Phase
captures the TIMING of dominant frequency, which is exactly
what changes at regime boundaries. This may fix the
structured regression.

## Mechanism

**FFT2 (magnitude + phase) input encoder**:
```python
x_clean = nan_to_num(x, nan=0)        # [B, T, D]
x_fft = rfft(x_clean, dim=1)          # [B, T//2+1, D] complex
x_fft_mag = abs(x_fft)                # [B, T//2+1, D]
x_fft_phase = angle(x_fft)            # [B, T//2+1, D]
# Pad to T
x_fft_mag_pad = pad(x_fft_mag)        # [B, T, D]
x_fft_phase_pad = pad(x_fft_phase)    # [B, T, D]
# Concat: original + mag + phase
x_aug = cat([x, x_fft_mag_pad, x_fft_phase_pad], dim=-1)  # [B, T, 3D]
```

## Hypotheses

- **H1 (positive)**: phase captures regime timing → fixes
  structured regression while preserving sin benefit
- **H2 (negative)**: phase adds noise (high variance) →
  no help or hurts
- **H3 (mixed)**: helps structured but doesn't help sin as
  much

## Configurations (2 conds)

1. `lbps_lnkhlfft2_2_5_2`: Kh=[2,5,2] + FFT2
2. `lbps_lnkhlfft2_5_3_2`: Kh=[5,3,2] + FFT2 (round 187
   winner)

Both with Kx=5, β=0.75, num_layers=3.

## Datasets

- sin_irr (D=2, T=32, missing_rate=0.3)
- structured_irr (D=2, T=32, missing_rate=0.3)
- random_irr (D=2, T=32, missing_rate=0.3)

## Bench

12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs.

## Success criteria

- **STRICTLY POSITIVE**: at least one cond beats SOTA on
  BOTH sin AND structured.
- **TARGET-DEP**: helps one, hurts the other.
- **NEGATIVE**: no cond beats SOTA on either.

## Why this direction is promising

1. **Phase captures regime timing** — different phases in
   different halves of structured data
2. **Sin has fixed phase** — phase feature won't disrupt
   sin
3. **Direct extension of round 187** (winner) — just add
   one more FFT component

## Files

- `lnn/core/learned_beta_ps_ln_khlfft2_cfc.py` (~150 lines)
- `tests/test_learned_beta_ps_ln_khlfft2_cfc.py` (12+ tests)
- `scripts/bench_learned_beta_ps_ln_khlfft2_cfc.py`
  (12-cell bench)
- `docs/research/2026-06-16_learned_beta_ps_ln_khlfft2_cfc_report.md`

**Why:** Add phase to FFT features. May fix structured
regression from round 187 while preserving sin benefit.

**How to apply:** If SP, replace SOTA. If NEGATIVE, log.
If TD, document.
