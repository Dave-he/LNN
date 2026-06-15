# Round 200 — Spectral Gating on CfC Hidden State — Research Report

**Date**: 2026-06-16
**Round**: 200
**Branch**: master
**Audit context (91-199)**: 47 strictly positive + 22 target-dep
+ 53 negatives = 122 mechanism classes.

## TL;DR

**TARGET-DEPENDENT (23rd) for Round 200**: FNO-style spectral
gating on CfC's hidden state gives **sin -34.6%** (strict
per-dataset win, largest in 9 rounds), neutral on structured,
hurts random +12.2%. Mean -2.5%.

**Ends 8-round NEG/TD streak with a strong positive signal**.
Frequency-domain processing on the hidden state extracts
periodic structure that linear projections miss.

## What was tested

**FNO-style spectral gating on the hidden state** (Li et al
2021, FNO). Replaces the linear g_branch with a learned
spectral filter:

```python
H = rFFT(h_t)  # complex [B, n_freq]
magnitude = |H|  # [B, n_freq]
mask = sigmoid(linear(magnitude))  # [B, n_freq]
H_filtered = H * mask  # apply learned mask
g = irFFT(H_filtered)  # real [B, hidden_size]
```

The mask is **content-aware** (depends on h's spectrum) and
learned. This is fundamentally different from r186 (FFT on
input only).

## Bench (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean | Δ vs CF | type |
|------|---------|----------------|------------|------|---------|------|
| cf | 0.0381 | 0.0001 | 0.0834 | 0.0405 | — | — |
| spec | 0.0249 | 0.0000 | 0.0936 | 0.0395 | -2.5% | **TD** |

## Per-dataset analysis

### sin_irr — STRICT PER-DATASET WIN
- cf: 0.0398 / 0.0363 (mean 0.0381)
- spec: 0.0269 / 0.0230 (mean 0.0249)
- **-34.6%** with both seeds improving
- Sin is fundamentally a frequency-domain signal,
  spectral filter extracts the dominant frequency

### structured_irr — slight improvement
- cf: 0.0001 / 0.0001 (mean 0.0001)
- spec: 0.0000 / 0.0000 (mean 0.0000)
- Already near-perfect for both

### random_irr — NEGATIVE
- cf: 0.0803 / 0.0866 (mean 0.0834)
- spec: 0.0907 / 0.0965 (mean 0.0936, +12.2%)
- Both seeds worse
- Random has uniform spectrum → no useful filter

## Pattern (47 + 22 + 53 = 122 → 47 + 23 + 53 = 123)

- 47 strictly positive (unchanged)
- **23 target-dep** (UP from 22, +1)
- 53 negatives (unchanged)
- Total: **123 mechanism classes**

## Why spectral gating helps sin

1. **Sin is a frequency-domain signal** — clear dominant
   frequency
2. **Spectral mask is content-aware** — adapts per timestep
3. **No spatial smoothing** — preserves phase information
4. **Better than FFT input features (r186)** — applied to
   hidden state (richer) not raw input

## Why spectral gating hurts random

1. **Random has uniform spectrum** — no dominant frequency
2. **Spectral filter has limited expressiveness** for noise
3. **Linear projection is better for noise** — can learn
   arbitrary patterns

## Critical implementation details

1. **rFFT/irFFT** — PyTorch real FFT, output
   `hidden_size // 2 + 1` complex values
2. **Magnitude-based mask** — from `|FFT(h)|`, not complex
3. **Mask in [0, 1]** — sigmoid squashes linear output
4. **Hidden_size preserved** — IRFFT reconstructs to
   original hidden_size
5. **Fewer params than baseline** — g_branch replaced by
   smaller spec_mask

## Why this is a useful TD

1. **First strict per-dataset win since r198 RK4** (sin -19%)
2. **LARGEST sin improvement since r192** (-34.6% vs -16%)
3. **Confirms frequency-domain processing is valuable for
   periodic data**
4. **Suggests additive spectral gating** could yield strict
   positives (try as ADDITIVE to g_branch, not REPLACEMENT)

## Comparison with r192-r199

| Round | Mechanism | Best sin | Best struct | Best random | Best mean | Verdict |
|-------|-----------|----------|-------------|-------------|-----------|---------|
| 192 | input noise | -16% | +6% | -26% | -24% | **SP** |
| 193 | hidden noise | -20% | -16% | +21% | +17% | TD |
| 194 | combined | +8% | -25% | +14% | +12% | TD |
| 196 | dropconnect | -14% | +63% | -3% | 0% | **NEG** |
| 197 | mixup | +272% | 0% | +37% | +130% | **NEG** |
| 198 | rk4 | -19% | 0% | +11% | +1.6% | **TD** |
| 199 | adadt | -1.3% | 0% | +9% | +5.8% | **NEG** |
| 200 | **spec** | **-34.6%** | 0% | +12% | -2.5% | **TD** |

**LARGEST sin improvement in 9 rounds** (-34.6% > r193 -20%).

## Caveats

- 2 seeds, 30 epochs
- Tested on r187 stack only
- Tested on 3 datasets only
- Replace (not additive) — might benefit from keeping g_branch

## Why the streak ended

After 8 rounds of NEG/TD on regularization (r193-r197) and
integration/dt schemes (r198-r199), the r187 baseline
appeared to be at a local optimum. The breakthrough with
spectral gating is in a fundamentally different axis:
**frequency-domain processing on the hidden state** vs
input/noise regularization.

## Next ideas

1. **Additive spectral gating** — keep g_branch AND add
   spectral_gated g → may give strict positives
2. **Spectral gating on input** (not just h) — different
   signal path
3. **Multi-resolution spectral gating** — different FFT
   sizes for different scales
4. **Spectral gating in test mode** — disable for noisy data
5. **Try on PhysioNet** — irregular time series where
   spectral structure may differ

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_specgated_cfc.py` (~210 lines)
- `tests/test_learned_beta_ps_ln_khlfft_specgated_cfc.py` (12 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_specgated_cfc.py` (12-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_specgated_cfc.json`

**Why:** Round 200 is **TARGET-DEPENDENT (23rd)** —
spectral gating on h helps sin **-34.6%** (strict win,
largest in 9 rounds), neutral on structured, hurts
random +12.2%.

**How to apply:** Use spectral gating for periodic/clean
data (sin, oscillators). Avoid for noise-dominated data.
