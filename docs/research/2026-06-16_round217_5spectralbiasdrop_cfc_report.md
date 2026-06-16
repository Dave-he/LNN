# Round 217 — 5-Scale Spectral + Bias + Dropout — Research Report

**Date**: 2026-06-16
**Round**: 217
**Branch**: master
**Audit context (91-216)**: 54 strictly positive + 27 target-dep
+ 58 negatives = 139 mechanism classes.

## TL;DR

**NEGATIVE 59th for Round 217** 🎯: 5-scale spectral (with 1-freq
sixteenth scale) regresses on all 3 datasets vs r216's 4-scale.

- sin: -29.3% (vs cf), +10.8% (vs r216)
- struct: +167.7% (vs cf), +361.1% (vs r216) — **REGRESSED**
- random: -2.6% (vs cf), +0.9% (vs r216)
- mean: -9.5% (vs cf), +9.1% (vs r216)

**4-SCALE IS THE SWEET SPOT** for the spectral axis.

## What was tested

**5-scale spectral + per-frequency bias + dropout p=0.2.**
Pushes the scale count from 4 (r216) to 5 by adding the
sixteenth scale (1 frequency for hidden=16).

## Bench (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs, hidden=16)

| Cond | sin_irr | structured_irr | random_irr | mean |
|------|---------|----------------|------------|------|
| cf | 0.0597 | 0.0031 | 0.0885 | 0.0504 |
| 4spectralbiasdrop (r216) | 0.0381 | 0.0018 | 0.0854 | 0.0418 |
| **5spectralbiasdrop (r217)** | **0.0422** | **0.0083** | **0.0862** | **0.0456** |

## Per-dataset analysis

### sin_irr
- cf: 0.0575 / 0.0618 (mean 0.0597)
- r216: 0.0366 / 0.0395 (mean 0.0381, -36.2%)
- **r217: 0.0380 / 0.0463 (mean 0.0422, -29.3% vs cf, +10.8% vs r216)** ✗

### structured_irr
- cf: 0.0031 / 0.0031 (mean 0.0031)
- r216: 0.0010 / 0.0025 (mean 0.0018, -41.9%)
- **r217: 0.0131 / 0.0034 (mean 0.0083, +167.7% vs cf, +361.1% vs r216)** ✗

### random_irr
- cf: 0.0955 / 0.0815 (mean 0.0885)
- r216: 0.0909 / 0.0798 (mean 0.0854, -3.5%)
- **r217: 0.0931 / 0.0792 (mean 0.0862, -2.6% vs cf, +0.9% vs r216)** ✗

## Why 5-scale hurts

For hidden=16, the 5th scale (sixteenth) has only **1 frequency**.
This is too few to be useful — the mask linear (1→1) doesn't
provide meaningful frequency selection. It adds noise without
contributing to the gating signal.

The 4-scale sweet spot:
- Scale 1 (full, 9 freqs) — most info
- Scale 2 (half, 5 freqs) — broad
- Scale 3 (quarter, 3 freqs) — medium
- Scale 4 (eighth, 2 freqs) — narrow but still useful

5th scale at 1 freq is **sub-threshold** — the model can't
distinguish this from bias noise.

## Pattern (54 + 27 + 58 = 139 → **54 + 27 + 59 = 140**)

- 54 strictly positive (unchanged)
- 27 target-dep (unchanged)
- **59 negatives (UP from 58, +1)** 🎯
- Total: **140 mechanism classes**

## Why this is a useful negative

1. **Confirms 4-scale sweet spot** — don't push past
2. **Sub-threshold mechanism** — 1-freq mask adds noise
3. **Clear verdict** — vs r216 is 0/3 wins

## Caveats

- 2 seeds, 30 epochs
- Hidden=16, lr=1e-2, batch_size=16
- 5-scale ~15% slower than 4-scale

## Next ideas

1. **Adaptive scale weights** (not simple avg) — round 211 was 3-scale adaptive
2. **Spectral L2 regularization** — penalize mask norm
3. **Per-scale bias** (not per-frequency) — simpler parameterization
4. **PhysioNet test** — real-world data

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_5spectralbiasdrop_cfc.py` (~300 lines)
- `tests/test_learned_beta_ps_ln_khlfft_5spectralbiasdrop_cfc.py` (10 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_5spectralbiasdrop_cfc.py` (18-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_5spectralbiasdrop_cfc.json`

**Why:** Round 217 is **NEGATIVE 59th** — 5-scale regresses on
all 3 datasets vs r216's 4-scale. The 4-scale sweet spot is
confirmed.

**How to apply:** Don't push past 4 scales for hidden=16.
Use 4-scale + bias + dropout (r216) as the canonical
spectral variant.
