# Round 188 — LearnedBetaPS+LN+Khl+FFT2-CfC — Research Report

**Date**: 2026-06-16
**Round**: 188
**Branch**: master
**Audit context (91-187)**: 46 strictly positive + 19 target-dep
+ 46 negatives = 111 mechanism classes.

## TL;DR

**NEGATIVE-WITH-NUANCE for Round 188**: Adding **phase** to
the FFT input features (mag+phase vs mag-only) **does not
beat round 187 SOTA** (sin 0.0026, structured 0.0024) on
either dataset. The 2_5_2 variant does improve over round
187's 2_5_2 on sin (0.0039 < 0.0061, 36% better) but still
loses to SOTA 0.0033.

## What was tested

**lb_ps + LN + Khl + FFT2 (mag+phase)** — extend round 187
by adding FFT phase features alongside magnitude:
```python
x_fft = rfft(x_clean, dim=1)        # complex
x_fft_mag = abs(x_fft)              # amplitude
x_fft_phase = angle(x_fft)          # NEW: timing
x_aug = cat([x, x_fft_mag_pad, x_fft_phase_pad], dim=-1)  # [B, T, 3D]
```

## Bench (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| lbps_lnkhlfft2_2_5_2 | 0.0039±0.0005 | 0.0070±0.0005 | 0.1578±0.0037 | 21233 |
| lbps_lnkhlfft2_5_3_2 | 0.0058±0.0006 | 0.0070±0.0011 | 0.1736±0.0074 | 22034 |

## Cross-round (best in class)

| Round | Mechanism | sin | structured |
|-------|-----------|-----|------------|
| 180 | lbps_ln_khl_2_5_2 | 0.0033 | 0.0058 |
| 180 | lbps_ln_khl_5_3_2 | 0.0198 | **0.0024** |
| 187 | lbps_lnkhlfft_2_5_2 | 0.0061 | 0.0068 |
| 187 | lbps_lnkhlfft_5_3_2 | **0.0026** | 0.0059 |
| **188** | **lbps_lnkhlfft2_2_5_2** | 0.0039 | 0.0070 |
| **188** | **lbps_lnkhlfft2_5_3_2** | 0.0058 | 0.0070 |

**None beat SOTA on sin (0.0026) or structured (0.0024).**

## Per-seed detail (lbps_lnkhlfft2_2_5_2)

| Dataset | Seed 0 | Seed 1 | Mean | SOTA | Δ vs SOTA |
|---------|--------|--------|------|------|-----------|
| sin | 0.0044 | 0.0034 | 0.0039 | 0.0026 | +50% |
| structured | 0.0075 | 0.0065 | 0.0070 | 0.0024 | +192% |
| random | 0.1615 | 0.1541 | 0.1578 | 0.17 | -7% |

Best sin seed 0.0034 — close to SOTA 0.0026 (31% above).

## Per-seed detail (lbps_lnkhlfft2_5_3_2)

| Dataset | Seed 0 | Seed 1 | Mean | SOTA | Δ vs SOTA |
|---------|--------|--------|------|------|-----------|
| sin | 0.0051 | 0.0064 | 0.0058 | 0.0026 | +123% |
| structured | 0.0059 | 0.0081 | 0.0070 | 0.0024 | +192% |
| random | 0.1662 | 0.1810 | 0.1736 | 0.17 | +2% |

The 5_3_2 variant regresses vs round 187's 5_3_2 on sin
(0.0058 vs 0.0026, +123%).

## Hypotheses revisited

- **H1 (positive)**: REJECTED. Phase does NOT fix
  structured regression. Both 2_5_2 and 5_3_2 conds are
  2.7-2.9x worse than SOTA 0.0024 on structured.
- **H2 (negative)**: CONFIRMED. Phase adds 50% more
  input features (D→3D) but no benefit on either dataset.
- **H3 (mixed)**: PARTIAL. 2_5_2 improves over round 187
  2_5_2 on sin (0.0039 < 0.0061) but not over SOTA.
  5_3_2 regresses on both.

## Why FFT2 (mag+phase) doesn't beat FFT1 (mag)

1. **Phase is high-variance in short windows** — the
   regime timing signal is weak when the FFT window is
   only 32 steps
2. **Random initialization of phase** — for noisy data
   like random_irr, phase is essentially random and
   adds 50% noise features
3. **Struct ured_irr has discrete regime change at t=16** —
   FFT averages across the boundary, and phase is
   already scrambled by the regime change itself
4. **The ladder already captures multi-scale info** —
   adding phase doesn't add new information axis

## Cross-comparison with round 187

| Metric | Round 187 2_5_2 | Round 188 2_5_2 | Round 187 5_3_2 | Round 188 5_3_2 |
|--------|-----------------|-----------------|-----------------|-----------------|
| sin | 0.0061 | 0.0039 (-36%) | 0.0026 | 0.0058 (+123%) |
| structured | 0.0068 | 0.0070 (+3%) | 0.0059 | 0.0070 (+19%) |
| n_params | 20633 | 21233 (+3%) | 21434 | 22034 (+3%) |

**Interesting**: 2_5_2 improves sin by 36% vs round 187
(adding phase helps when the ladder is sin-friendly).
But 5_3_2 regresses on both.

This is **interaction effect**: the sin-friendly
[2,5,2] ladder benefits from phase, the structured-
friendly [5,3,2] ladder gets noise from phase.

## Pattern (unchanged at 46 + 19 + 46 = 111 mechanism classes)

- **46 strictly positive** (unchanged)
- **19 target-dep** (unchanged)
- **46 negatives** (UP from 46, +1)
- Total: **111 mechanism classes** (unchanged)

Wait — that's a math error. If this is negative, it
should be +1 to negatives:
- 46 SP + 19 TD + 47 NEG = 112 total

Actually let me recount. The audit count is:
- 91-187 inclusive = 97 rounds, but many are exploratory
  branches. The count "46 + 19 + 46 = 111" is the
  number of DISTINCT mechanism classes, not rounds.
- Adding a new mechanism class (FFT2) that's NEGATIVE
  → 46 + 19 + 47 = 112.

Let me correct the pattern:
- **46 strictly positive** (unchanged from 187)
- **19 target-dep** (unchanged from 187)
- **47 negatives** (UP from 46, round 188 adds 1)
- Total: **112 mechanism classes**

## Critical implementation details

1. **FFT2InputEncoder outputs [B, T, 3D]** vs FFT's
   [B, T, 2D] (extra phase channels)
2. **NaN preserved in original channels**, but mag/phase
   computed on NaN-replaced input → finite
3. **Wrapped around round 180 stack** — Kh ladder +
   per-scale β + LN unchanged
4. **Tests**: 15/15 pass
5. **Param cost**: +3% vs round 187 (extra input features)

## Why this is a useful negative

1. **Phase is NOT a free win** — it adds features but
   not information in this regime
2. **Mag-only is the right level of frequency info** for
   1D synthetic data
3. **Confirms round 187 was not a fluke** — the FFT
   benefit comes from magnitude, not phase
4. **Suggests trying STFT or wavelet** next — localized
   frequency analysis may preserve regime info better

## Caveats

- **Best sin seed 0.0034 close to SOTA 0.0026** (31%
  above) — adding more seeds might find a 0.0026 hit
- **Not all 2_5_2 vs 5_3_2 patterns are clear** — only
  2 seeds per cell, std 0.0005-0.0011
- **Structured_irr is genuinely hard** — both rounds
  187 and 188 regress on it (0.0070 vs SOTA 0.0024)

## Next ideas

1. **STFT (Short-Time FFT)** — sliding window preserves
   regime info better than full FFT
2. **Wavelet features** — multi-resolution frequency
   without FFT's global averaging
3. **Log-amplitude FFT** — compress dynamic range, may
   help structured
4. **Gated FFT** — only apply FFT when periodicity is
   detected

## Files

- `lnn/core/learned_beta_ps_ln_khlfft2_cfc.py` (~150
  lines)
- `tests/test_learned_beta_ps_ln_khlfft2_cfc.py`
  (15 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft2_cfc.py`
  (12-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft2_cfc.json`
- `docs/prds/2026-06-16-lnn-round-188-learned-beta-ps-ln-khlfft2-cfc.md`
- `docs/research/2026-06-16_learned_beta_ps_ln_khlfft2_cfc_report.md`

**Why:** Round 188 is **NEGATIVE-WITH-NUANCE** (no cond
beats SOTA on either dataset, but 2_5_2 improves over
round 187 2_5_2 on sin).

**How to apply:** FFT2 (mag+phase) is a useful failure
mode to record. Don't use. Try STFT/wavelet next.
Audit becomes 46+19+47=112.
