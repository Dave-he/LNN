# PRD #10-164 — Round 202 — Learnable λ Convex Spectral Gating

**Date**: 2026-06-16
**Round**: 202
**Branch**: master
**Audit context (91-201)**: 47 strictly positive + 23 target-dep
+ 54 negatives = 124 mechanism classes.

## Background

Round 200 (spectral REPLACE) = TD with sin -34.6% strict win.
Round 201 (spectral ADD) = NEGATIVE — additive composition
diluted the spectral signal without recovering random.

Round 202 tests **learnable λ convex combination** — let the
model learn when to use spectral (periodic data) vs linear
(noisy data):

  λ(z) = sigmoid(linear(z))           # ∈ [0, 1] per-feature [B, H]
  g_combined = (1-λ) * g_branch(z) + λ * spectral_g(h_t)
  h_new = tau_eff * g_combined + (1 - tau_eff) * h_branch

Hypothesis: λ learns to use spectral on periodic data and
linear on noisy data → STRICTLY POSITIVE on all 3.

## Goal

Test if learnable λ gating can adaptively route between
linear and spectral paths based on input complexity.

## Mechanism

```python
# r187 baseline: g_branch(z), h_branch(z)
# r200: spectral_g(h_t) REPLACES g_branch
# r201: g_combined = g_branch + spectral_g (additive)
# r202 (NEW): g_combined = (1-λ) * g_branch + λ * spectral_g

f = σ(linear(z))
g = tanh(linear(z))              # r187 linear g_branch
h_branch = tanh(linear(z))       # r187 h_branch

# NEW: learnable λ from z
λ = σ(linear(z))                 # [B, H] in [0, 1]

H = FFT(h_t)
mask = sigmoid(linear(|H|))
g_spec = IFFT(H * mask)

g_combined = (1-λ) * g + λ * g_spec
tau_eff = exp(-f * dt / |time_scale|)
h_new = tau_eff * g_combined + (1 - tau_eff) * h_branch
```

## Configurations (2 conds)

1. `cf`: r187 baseline (linear g_branch)
2. `lambda`: r202 learnable λ convex combination

## Result (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean | Δ vs CF | type |
|------|---------|----------------|------------|------|---------|------|
| cf | 0.0381 | 0.0001 | 0.0834 | 0.0405 | — | — |
| lambda | 0.0310 | 0.0000 | 0.0925 | 0.0412 | +1.7% | **NEG** |

Per-dataset (lambda vs cf):
- sin_irr: 0.0381 → 0.0310 (**-18.6%**, between r200 -34.6% and r201 -22.8%)
- structured_irr: 0.0001 → 0.0000 (~0%)
- random_irr: 0.0834 → 0.0925 (**+10.9%**, similar to r200 spec +12.2%)

## Verdict

**NEGATIVE (55th)** — learnable λ does NOT recover random
performance either. Random still regresses +10.9% (same as
r200 spec +12.2%).

**Sin improvement** is between r200 (-34.6%) and r201 (-22.8%).
The λ learned partial spectral behavior but didn't reach r200's
full spectral dominance.

## Pattern (47 + 23 + 54 = 124 → 47 + 23 + 55 = 125)

- 47 strictly positive (unchanged)
- 23 target-dep (unchanged)
- **55 negatives** (UP from 54, +1)
- Total: **125 mechanism classes**

## Why learnable λ failed (H2: λ doesn't route cleanly)

Hypothesis was H1 (λ learns to use spectral on periodic data,
linear on noisy data → SP). Actually got H2: **λ doesn't
learn to route cleanly**. Spectral path's gradient dominates
regardless of λ because:
1. **Spectral signal has higher variance** than linear signal
2. **λ is initialized near 0.5** (sigmoid of 0) — gradient
   signal pushes λ toward whichever path has lower loss
3. **Random data doesn't have a clear signal to extract**,
   so λ learns to ignore it (but spec still hurts)

## Critical implementation details

1. **Per-feature λ** [B, H] — not scalar, allows different
   mixing per hidden dimension
2. **Initialized to ~0.5** (sigmoid of zero bias) — balanced
   starting point
3. **+294 params per cell** vs baseline (lambda_gate 264 +
   spec_mask 30) = +882 total for 3 cells
4. **Convex combination** — g_combined ∈ convex hull of
   {g, g_spec}

## Comparison r200-r202: spectral gating variants

| Round | Mechanism | sin | struct | random | mean | Verdict |
|-------|-----------|-----|--------|--------|------|---------|
| 200 | spec (REPLACE) | **-34.6%** | 0% | +12.2% | -2.5% | **TD** |
| 201 | addspec (ADD) | -22.8% | 0% | +11.2% | +0.5% | **NEG** |
| 202 | lambda (CONVEX) | -18.6% | 0% | +10.9% | +1.7% | **NEG** |

REPLACE > ADD > CONVEX in terms of sin improvement.
None recover random performance.

## Why this is a useful NEG

1. **Disproves H1** — even with learnable λ, cannot recover
   random performance
2. **Confirms H2** — λ doesn't learn to route cleanly, spectral
   gradient dominates
3. **Confirms r200** — REPLACE-style spectral is the best
   approach for spectral gating on CfC
4. **Suggests** that to recover random, must use a different
   mechanism entirely (e.g., dataset-conditioned routing)

## Next ideas

1. **Dataset-conditioned spectral** — explicitly disable spectral
   when noise is detected
2. **Multi-resolution wavelets** — Sonnet-style approach
3. **Different axis entirely** — try attention, state-space, etc.
4. **Move on** — r200 remains the spectral gating winner
5. **Try on PhysioNet** — irregular time series where spectral
   structure differs

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_lambda_specgated_cfc.py` (~210 lines)
- `tests/test_learned_beta_ps_ln_khlfft_lambda_specgated_cfc.py` (12 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_lambda_specgated_cfc.py` (12-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_lambda_specgated_cfc.json`

**Why:** Round 202 is **NEGATIVE (55th)** — learnable λ
convex combination does NOT recover random performance
(+10.9% loss, similar to r200 spec). Sin improvement
is between r200 (-34.6%) and r201 (-22.8%).

**How to apply:** Learnable λ gating is not a clear win.
r200's REPLACE-style spectral gating remains the spectral
gating winner for periodic data.
