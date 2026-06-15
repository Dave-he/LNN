# Round 199 — Content-Aware Adaptive dt for CfC — Research Report

**Date**: 2026-06-16
**Round**: 199
**Branch**: master
**Audit context (91-198)**: 47 strictly positive + 22 target-dep
+ 52 negatives = 121 mechanism classes.

## TL;DR

**NEGATIVE for Round 199**: Content-aware adaptive dt is
**+5.8% mean worse**. Per-feature dt is over-parameterized
and redundant with CfC's existing time_scale parameter.
Sin/structured neutral, random +9.1% worse.

**8-round streak of NEG/TD** (r193-r199). After r192 input
noise (the only SP in 8 rounds), all subsequent mechanisms
have failed to consistently improve on the r187 baseline.

## What was tested

**Content-aware adaptive dt** — let the model learn dt at
each timestep based on input content:

```python
dt_t = sigmoid(linear(z)) * dt_max  # [B, H] per-feature dt
tau_eff = exp(-f * dt_t / |time_scale|)
h_t = tau_eff * g + (1 - tau_eff) * h_branch
```

The dt is **per-feature** (one value per hidden dim), giving
the model fine-grained control over which features decay fast
vs slow.

Inspired by Hyena Edge (Liquid AI, April 2025) which uses
long convolutions as a replacement for attention, suggesting
that content-modulated dynamics can be a powerful inductive
bias.

## Bench (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean | Δ vs CF | type |
|------|---------|----------------|------------|------|---------|------|
| cf | 0.0381 | 0.0001 | 0.0834 | 0.0405 | — | — |
| adadt | 0.0376 | 0.0001 | 0.0910 | 0.0429 | +5.8% | **NEG** |

## Per-dataset analysis

### sin_irr — neutral (mixed)
- cf: 0.0398 / 0.0363 (mean 0.0381)
- adadt: 0.0415 / 0.0336 (mean 0.0376, -1.3%)
- Seed 0 worse, seed 1 better → noisy signal

### structured_irr — neutral
- cf: 0.0001 / 0.0001 (mean 0.0001)
- adadt: 0.0001 / 0.0000 (mean 0.0001)
- Already near-perfect for both

### random_irr — NEGATIVE
- cf: 0.0803 / 0.0866 (mean 0.0834)
- adadt: 0.0933 / 0.0887 (mean 0.0910, +9.1%)
- Both seeds worse

## Pattern (47 + 22 + 52 = 121 → 47 + 22 + 53 = 122)

- 47 strictly positive (unchanged)
- 22 target-dep (unchanged)
- **53 negatives** (UP from 52, +1)
- Total: **122 mechanism classes**

## Why AdaDt doesn't help

1. **Per-feature dt is over-parameterized** — H values per
   timestep is a lot of new flexibility, but the model
   doesn't have a clear training signal for each
2. **Random data has no clean dt** — for noise, every dt
   is equally wrong, so the model can't learn a useful
   per-feature pattern
3. **Sin data already learned** — the closed-form is
   exact for sin, and adaptive dt doesn't add info
4. **Sigmoid * dt_max bounds the range** — even at max
   (sigmoid=1, dt_max=2), the effect is limited

## Why this is a useful negative

1. **Content-aware dt is a plausible but ineffective
   mechanism** — confirms that simply adding more
   flexibility doesn't help
2. **Per-feature dt is the wrong granularity** — a
   scalar dt per timestep (or per layer) would be more
   constrained
3. **CfC's time_scale parameter is already a per-feature
   time constant** — adding a per-feature dt is
   redundant

## Critical implementation details

1. **Sigmoid * dt_max** — bounds dt in [0, dt_max=2.0]
2. **Per-feature dt** — output of `dt_predictor` is [B, H]
3. **dt_predictor shares LayerNorm output z** — same
   input as f_gate, g_branch, h_branch
4. **External dt still supported** — for ablation
5. **dt_max=0** — gives dt=0, h_new=h_branch (limit case)

## Comparison with r192-r198

| Round | Mechanism | Best sin | Best struct | Best random | Best mean | Verdict |
|-------|-----------|----------|-------------|-------------|-----------|---------|
| 192 | input noise | -16% | +6% | -26% | -24% | **SP** |
| 193 | hidden noise | -20% | -16% | +21% | +17% | TD |
| 194 | combined | +8% | -25% | +14% | +12% | TD |
| 196 | dropconnect | -14% | +63% | -3% | 0% | **NEG** |
| 197 | mixup | +272% | 0% | +37% | +130% | **NEG** |
| 198 | rk4 | -19% | 0% | +11% | +1.6% | **TD** |
| 199 | **adadt** | -1.3% | 0% | +9% | +5.8% | **NEG** |

8-round streak of NEG/TD (r193-r199). Only r192 input noise
is SP in this window.

## Why the streak?

After 3 noise/regularization rounds (r192-r196) and 2 paradigm
pivots (RK4 r198, AdaDt r199), the r187 baseline is hard to
beat. The mechanisms tested either:
- Add flexibility that's redundant with existing parameters
  (AdaDt vs time_scale)
- Add cost that doesn't translate to accuracy (RK4 on noisy data)
- Add data-specific assumptions that don't generalize (Mixup)

## Caveats

- 2 seeds, 30 epochs
- Tested on r187 stack only
- Tested on 3 datasets only
- dt_max=2.0 only (could try 0.5, 1.0, 4.0)

## Next ideas

1. **Scalar dt per timestep** — single value instead of
   per-feature
2. **Layer-wise dt** — one dt per layer, not per-feature
3. **Per-input dt** — condition dt on the input only
4. **Different dt for x and h** — separate time scales
5. **Pivot to a different paradigm** — done with
   regularization/integration/dt space

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_adadt_cfc.py` (~210 lines)
- `tests/test_learned_beta_ps_ln_khlfft_adadt_cfc.py` (12 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_adadt_cfc.py` (12-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_adadt_cfc.json`

**Why:** Round 199 is **NEGATIVE (53rd)** — content-aware
adaptive dt is +5.8% mean worse. Per-feature dt is
over-parameterized and doesn't help on these datasets.

**How to apply:** Don't use per-feature adaptive dt. CfC's
time_scale parameter is already a per-feature time constant,
making per-feature dt redundant.
