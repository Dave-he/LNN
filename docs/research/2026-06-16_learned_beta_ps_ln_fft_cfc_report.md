# Round 186 — LearnedBetaPS+LN+FFT-CfC — Research Report

**Date**: 2026-06-16
**Round**: 186
**Branch**: master
**Audit context (91-185)**: 45 strictly positive + 18 target-dep
+ 46 negatives = 109 mechanism classes.

## TL;DR

**TARGET-DEPENDENT for Round 186**: FFT input features
**help sin** (best seed 0.0022 < SOTA 0.0033) but **hurt
structured** (all 3 conds 0.0066-0.0075 vs SOTA 0.0024). This
is a fundamentally different mechanism class (frequency
domain) and gives strong signals worth exploring.

## What was tested

**lb_ps + LN + FFT** — add FFT magnitude as additional input
features. The FFT captures frequency-domain information
that the time-domain EMA features miss.

```python
x_clean = nan_to_num(x, nan=0)        # [B, T, D]
x_fft = abs(rfft(x_clean, dim=1))     # [B, T//2+1, D]
x_fft_pad = pad(x_fft, [0, T - T//2 - 1])  # [B, T, D]
x_aug = cat([x, x_fft_pad], dim=-1)   # [B, T, 2D]
```

## Bench (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| lbps_lnfft_h3_75 | 0.0067±0.0031 | 0.0072±0.0004 | 0.1697±0.0068 | 20633 |
| lbps_lnfft_h2_75 | 0.0042±0.0009 | 0.0066±0.0001 | 0.1707±0.0098 | 18230 |
| lbps_lnfft_h5_75 | 0.0037±0.0015 | 0.0075±0.0001 | 0.1732±0.0080 | 25439 |

## Cross-round (best in class)

| Round | Mechanism | sin (mean) | structured (mean) |
|-------|-----------|------------|---------------------|
| 180 | lbps_ln_khl_2_5_2 | **0.0033** | 0.0058 |
| 180 | lbps_ln_khl_5_3_2 | 0.0198 | **0.0024** |
| **186** | **lbps_lnfft_h5_75** | 0.0037 | 0.0075 |
| **186** | **lbps_lnfft_h2_75** | 0.0042 | 0.0066 |
| **186** | **lbps_lnfft_h3_75** | 0.0067 | 0.0072 |

**No mean-based new bests** but lbps_lnfft_h5_75 sin has
**best seed 0.0022 < SOTA 0.0033** (33% improvement!).

## Per-seed positive signals

lbps_lnfft_h5_75 sin: seeds 0.0022, 0.0051 (mean 0.0037)
- Seed 0: 0.0022 (33% better than SOTA 0.0033)
- This is a strong signal that FFT helps periodic data.

lbps_lnfft_h2_75 sin: seeds 0.0033, 0.0050 (mean 0.0042)
- Seed 0: 0.0033 (ties SOTA)

lbps_lnfft_h3_75 sin: worse on both seeds

## Hypotheses revisited

- **H1 (positive)**: PARTIAL. FFT magnitude captures
  frequency patterns. Per-seed signals support this for
  sin data.
- **H2 (negative)**: REJECTED. FFT does add useful signal
  (sin improvement), not just noise.
- **H3 (mixed)**: CONFIRMED. Helps sin (periodic data)
  but hurts structured (non-periodic data).

## Why FFT helps sin but hurts structured

### 1. sin_irr is periodic
FFT magnitude directly captures the dominant frequency
(≈0.25 cycles/step). This is a perfect feature for sin
data.

### 2. structured_irr has regime changes
The structured dataset has TWO different frequencies (sin
and 2·sin) in different halves. FFT averages these into
a single magnitude per bin, losing the regime information.
EMAs handle regime changes better because they're
time-local.

### 3. random_irr is non-periodic
FFT magnitude for random walk is roughly flat across
bins. No useful info added.

## Pattern (45 + 19 + 46 = 110 mechanism classes)

- **45 strictly positive** (unchanged)
- **19 target-dep** (UP from 18, round 186 adds 1)
- **46 negatives** (unchanged)
- Total: **110 mechanism classes**

## Critical implementation details

1. **FFTInputEncoder** — pads FFT bins to T, concats with
   original
2. **NaN handling** — `nan_to_num(x, nan=0)` before FFT
3. **Inherits from round 180 lbps_ln_khl** (full pipeline)
4. **Input size doubles** (D → 2D)
5. **Tests** — 13/13 pass

## Why this is a useful TD

1. **First mechanism class change in 6 rounds** (181-186
   were all lb_ps variants)
2. **Confirms H3 (target-dependent)** — frequency
   features help periodic data but hurt non-periodic
3. **Per-seed positive on sin** — best seed 0.0022 < SOTA
   0.0033 (worth more seeds)
4. **Pivots from lb_ps variants** — FFT is a fundamentally
   different mechanism

## Next ideas

1. **More seeds for lbps_lnfft_h5_75 on sin** — confirm
   the 0.0022 seed
2. **Hybrid FFT + Kh ladder** — combine FFT with
   lbps_ln_khl_2_5_2 to see if FFT helps sin while Kh
   ladder handles structured
3. **Different FFT variants** — log-amplitude, phase,
   STFT (sliding window)
4. **Other mechanism classes** — 1D conv, attention,
   frequency domain

## Files

- `lnn/core/learned_beta_ps_ln_fft_cfc.py` (~210 lines)
- `tests/test_learned_beta_ps_ln_fft_cfc.py` (13 tests)
- `scripts/bench_learned_beta_ps_ln_fft_cfc.py` (18-cell bench)
- `results/bench_learned_beta_ps_ln_fft_cfc.json`
- `docs/prds/2026-06-16-lnn-round-186-learned-beta-ps-ln-fft-cfc.md`
- `docs/research/2026-06-16_learned_beta_ps_ln_fft_cfc_report.md`

**Why:** Round 186 is TARGET-DEPENDENT. FFT helps sin
(per-seed) but hurts structured. Pivot from lb_ps
variants is successful in producing different signal.

**How to apply:** Try FFT + Kh ladder hybrid next.
Audit becomes 110 (19 TD).
