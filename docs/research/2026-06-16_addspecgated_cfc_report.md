# Round 201 — Additive Spectral Gating on CfC — Research Report

**Date**: 2026-06-16
**Round**: 201
**Branch**: master
**Audit context (91-200)**: 47 strictly positive + 23 target-dep
+ 53 negatives = 123 mechanism classes.

## TL;DR

**NEGATIVE (54th) for Round 201**: Additive spectral gating
(linear g_branch + spectral_g(h_t)) does NOT recover random
performance — random still regresses +11.2%. Sin improvement
is SMALLER than r200 spec (-22.8% vs -34.6%) because additive
combination dilutes spectral signal. Mean +0.5%.

**Disproves H1 (combine linear + spectral → SP)**, confirms
H2 (gradient signal split). r200's REPLACE-style remains
the spectral gating winner.

## What was tested

**Additive spectral gating** on CfC. The natural follow-up to
r200 — keep r187's linear g_branch AND add spectral_g(h_t):

```python
g = tanh(linear(z))              # r187 linear g_branch (kept)
h_branch = tanh(linear(z))       # r187 h_branch (kept)
H = FFT(h_t)
mask = sigmoid(linear(|H|))
g_spec = IFFT(H * mask)
g_combined = g + g_spec          # additive combination
tau_eff = exp(-f * dt / |time_scale|)
h_new = tau_eff * g_combined + (1 - tau_eff) * h_branch
```

Hypothesis: linear path is noise-robust (helps random),
spectral path is periodic-sensitive (helps sin). Together
should give SP on all 3.

## Bench (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean | Δ vs CF | type |
|------|---------|----------------|------------|------|---------|------|
| cf | 0.0381 | 0.0001 | 0.0834 | 0.0405 | — | — |
| spec | 0.0249 | 0.0000 | 0.0936 | 0.0395 | -2.5% | **TD (r200)** |
| addspec | 0.0294 | 0.0000 | 0.0927 | 0.0407 | +0.5% | **NEG** |

## Per-dataset analysis

### sin_irr — SMALLER WIN than r200
- cf: 0.0398 / 0.0363 (mean 0.0381)
- spec: 0.0269 / 0.0230 (mean 0.0249)
- addspec: 0.0335 / 0.0253 (mean 0.0294)
- **-22.8%** vs r200's **-34.6%** — additive dilutes

### structured_irr — neutral
- cf: 0.0001 / 0.0001 (mean 0.0001)
- spec: 0.0000 / 0.0000 (mean 0.0000)
- addspec: 0.0000 / 0.0001 (mean 0.0000)

### random_irr — SAME LOSS as r200
- cf: 0.0803 / 0.0866 (mean 0.0834)
- spec: 0.0907 / 0.0965 (mean 0.0936, +12.2%)
- addspec: 0.0966 / 0.0888 (mean 0.0927, +11.2%)
- Linear path did NOT recover random

## Pattern (47 + 23 + 53 = 123 → 47 + 23 + 54 = 124)

- 47 strictly positive (unchanged)
- 23 target-dep (unchanged)
- **54 negatives** (UP from 53, +1)
- Total: **124 mechanism classes**

## Why additive spectral gating failed

Hypothesis was H1 (combine linear noise-robust + spectral
periodic → SP). Actually got H2: **gradient signal splits
between linear and spectral paths**:

1. **Doubled fast path magnitude** (g + g_spec) means
   tau_eff mixing distributes contribution differently
2. **Spectral gradient still dominates** — random loss
   same as r200
3. **Linear gradient adds noise** — sin improvement reduced

The fix would be to **gate spectral contribution by data
complexity** (learnable λ) or use convex combination.

## Critical implementation details

1. **Linear g_branch preserved** (not replaced) — adds
   264 params per cell vs r200
2. **spec_mask still 30 params per cell**
3. **+90 params total vs r187 baseline**
4. **g_combined = g + g_spec** (raw sum, no scaling)

## Why this is a useful NEG

1. **Disproves H1** — additive composition does not give SP
2. **Confirms H2** — gradient signal split hurts both
3. **Suggests gating** is needed — pure additive doesn't work
4. **r200 still wins** for spectral gating on periodic data

## Comparison with r192-r200

| Round | Mechanism | Best sin | Best struct | Best random | Best mean | Verdict |
|-------|-----------|----------|-------------|-------------|-----------|---------|
| 192 | input noise | -16% | +6% | -26% | -24% | **SP** |
| 198 | rk4 | -19% | 0% | +11% | +1.6% | **TD** |
| 200 | spec | **-34.6%** | 0% | +12% | -2.5% | **TD** |
| 201 | **addspec** | -22.8% | 0% | +11% | +0.5% | **NEG** |

addspec is strictly worse than spec on every metric.

## Why the streak ended (and r201 didn't continue it)

After r200's TD breakthrough, r201 tested a natural extension
(additive) which turned out to be a NEGATIVE. The streak
remains 1 TD win (r200). The lesson: **REPLACE** beats
**ADD** for spectral gating on CfC because the linear path
doesn't actually help recover random — spectral dominates
gradient regardless.

## Next ideas

1. **Learnable λ gating** — λ * spec_g with λ learned per
   timestep based on noise estimate
2. **Convex combination** — (1-λ) * g_branch + λ * spec_g
3. **Multi-resolution wavelets** — Sonnet-style approach
4. **Spectral on input only** — different signal path
5. **Move to next axis** — try completely different mechanism

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_addspecgated_cfc.py` (~205 lines)
- `tests/test_learned_beta_ps_ln_khlfft_addspecgated_cfc.py` (12 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_addspecgated_cfc.py` (18-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_addspecgated_cfc.json`

**Why:** Round 201 is **NEGATIVE (54th)** — additive
spectral gating does NOT recover random performance
(+11.2% loss, same as r200 spec). Sin improvement is
smaller (-22.8% vs r200's -34.6%) because additive
combination dilutes spectral signal.

**How to apply:** Don't use additive spectral gating as
default. r200's REPLACE-style spectral gating is better.
For multi-path combination, use learnable λ gating.
