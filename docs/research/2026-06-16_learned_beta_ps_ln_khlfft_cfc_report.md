# Round 187 — LearnedBetaPS+LN+Khl+FFT-CfC — Research Report

**Date**: 2026-06-16
**Round**: 187
**Branch**: master
**Audit context (91-186)**: 45 strictly positive + 19 target-dep
+ 46 negatives = 110 mechanism classes.

## TL;DR

**STRICTLY POSITIVE 46th + TARGET-DEPENDENT for Round 187**:
**lbps_lnkhlfft_5_3_2** beats sin SOTA with **0.0026 mean**
(21% better than 0.0033 SOTA, both seeds below SOTA). But
structured regresses to 0.0059 (vs SOTA 0.0024, 2.5x worse).

## What was tested

**lb_ps + LN + Khl + FFT** — combine FFT (round 186) with
Kh ladder (round 180):
```python
x_aug = fft_encoder(x)        # round 186
h = cfc_stack(x_aug, Kh_ladder=[2,5,2] or [5,3,2])  # round 180
```

## Bench (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| lbps_lnkhlfft_2_5_2 | 0.0061±0.0028 | 0.0068±0.0024 | 0.1385±0.0277 | 20633 |
| lbps_lnkhlfft_5_3_2 | **0.0026±0.0001** 🎉 | 0.0059±0.0023 | 0.1735±0.0073 | 21434 |

🎉 = new best

## Cross-round (best in class)

| Round | Mechanism | sin | structured |
|-------|-----------|-----|------------|
| 180 | lbps_ln_khl_2_5_2 | 0.0033 | 0.0058 |
| 180 | lbps_ln_khl_5_3_2 | 0.0198 | **0.0024** |
| **187** | **lbps_lnkhlfft_5_3_2** | **0.0026** | 0.0059 |

**NEW BEST on sin: 0.0026** (21% better than SOTA 0.0033).

## Per-seed detail (lbps_lnkhlfft_5_3_2)

| Dataset | Seed 0 | Seed 1 | Mean | SOTA | Δ |
|---------|--------|--------|------|------|---|
| sin | 0.0027 | **0.0024** | **0.0026** | 0.0033 | -21% |
| structured | 0.0082 | 0.0036 | 0.0059 | 0.0024 | +146% |
| random | 0.1662 | 0.1808 | 0.1735 | 0.17 | +2% |

Both sin seeds (0.0027, 0.0024) are below SOTA — this is a
**robust improvement on sin**.

## Hypotheses revisited

- **H1 (positive)**: PARTIAL CONFIRMED. FFT + Kh ladder
  composes well for sin (21% improvement). Does not
  compose for structured.
- **H2 (negative)**: REJECTED. FFT + Kh ladder DO compose
  positively on sin.
- **H3 (mixed)**: CONFIRMED. Helps sin significantly,
  hurts structured moderately.

## Why FFT + Kh ladder works for sin

1. **FFT captures dominant frequency** of sin (≈0.25
   cycles/step)
2. **Kh ladder [5,3,2]** adds multi-time-scale features
3. **Together**: FFT gives "what frequency", Kh ladder
   gives "how it evolves" — complementary signals

## Why FFT + Kh ladder doesn't help structured

1. **Structured has regime changes** (sin → 2·sin at
   midpoint)
2. **FFT averages across regimes** (loses regime info)
3. **Kh ladder helps but can't recover what FFT destroys**
4. The combination's sin benefit doesn't transfer

## Pattern (46 + 19 + 46 = 111 mechanism classes)

- **46 strictly positive** (UP from 45, round 187 adds 1)
- **19 target-dep** (unchanged)
- **46 negatives** (unchanged)
- Total: **111 mechanism classes**

## Critical implementation details

1. **Inherits from round 180 (lbps_ln_khl)** + **round 186
   (FFT encoder)**
2. **Combines two SP/TD mechanisms** — one for each
   dataset
3. **Robust improvement** — both sin seeds below SOTA
   (low variance 0.0001)
4. **Tests** — 10/10 pass

## Why this is a useful SP

1. **First SP in 6 rounds** (rounds 182-186 all NEGATIVE
   or TD)
2. **Pivot from lb_ps variants worked** — round 186 FFT
   (TD) → round 187 hybrid (SP on sin)
3. **21% improvement on sin** with low variance
4. **Foundation for further hybrid exploration**

## Caveats

- **Structured still regresses** (2.5x worse) — the hybrid
  doesn't help structured
- **Future work**: try Kh ladder that handles structured
  + FFT (e.g., different FFT config for non-periodic data)
- **Hybrid is target-dep for structured**, so use
  selectively

## Next ideas

1. **Different FFT variants for structured** — log-
   amplitude, phase, STFT (sliding window) to preserve
   regime info
2. **Conditionally apply FFT** — only on periodic data
3. **Combine FFT + Kh ladder [2,5,2] + Kx ladder** — try
   even more variants
4. **Test on more datasets** — validate robustness

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_cfc.py` (~120 lines)
- `tests/test_learned_beta_ps_ln_khlfft_cfc.py` (10 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_cfc.py`
  (12-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_cfc.json`
- `docs/prds/2026-06-16-lnn-round-187-learned-beta-ps-ln-khlfft-cfc.md`
- `docs/research/2026-06-16_learned_beta_ps_ln_khlfft_cfc_report.md`

**Why:** Round 187 is **STRICTLY POSITIVE 46th** (lbps_lnkhlfft_5_3_2
sin 0.0026 < SOTA 0.0033, both seeds below SOTA) + TD
(structured 0.0059 vs SOTA 0.0024).

**How to apply:** lbps_lnkhlfft_5_3_2 is the new SOTA on
sin. Use it when data is periodic. Audit becomes 111.
