# PRD #10-163 — Round 201 — Additive Spectral Gating on CfC

**Date**: 2026-06-16
**Round**: 201
**Branch**: master
**Audit context (91-200)**: 47 strictly positive + 23 target-dep
+ 53 negatives = 123 mechanism classes.

## Background

Round 200's spectral gating (replacing linear g_branch) was
TARGET-DEPENDENT: sin -34.6% (strict win), struct 0%, random
+12.2%. The hypothesis for r201 was that ADDING spectral to
g_branch (not REPLACING it) would preserve the linear noise-
robust path AND add spectral periodic sensitivity, potentially
giving STRICTLY POSITIVE on all 3 datasets.

## Goal

Test if `g_combined = g_branch(z) + spectral_g(h_t)` recovers
random performance while keeping sin improvement.

## Mechanism

```python
# r187 (baseline): g_branch(z), h_branch(z), tau_eff * g + (1-tau_eff) * h_branch
# r200: spectral_g(h_t) replaces g_branch
# r201 (NEW): g_combined = g_branch(z) + spectral_g(h_t)
g = tanh(linear(z))                     # r187 linear g_branch
h_branch = tanh(linear(z))              # r187 h_branch
H = FFT(h_t)
mask = sigmoid(linear(|H|))
g_spec = IFFT(H * mask)
g_combined = g + g_spec                 # additive
tau_eff = exp(-f * dt / |time_scale|)
h_new = tau_eff * g_combined + (1 - tau_eff) * h_branch
```

The linear path provides noise robustness (helps random);
spectral path adds periodic content (helps sin).

## Configurations (3 conds)

1. `cf`: r187 baseline (linear g_branch)
2. `spec`: r200 spectral gating (REPLACES g_branch)
3. `addspec`: r201 additive spectral gating (linear + spectral)

## Result (18 cells: 3 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean | Δ vs CF | type |
|------|---------|----------------|------------|------|---------|------|
| cf | 0.0381 | 0.0001 | 0.0834 | 0.0405 | — | — |
| spec | 0.0249 | 0.0000 | 0.0936 | 0.0395 | -2.5% | **TD (r200)** |
| addspec | 0.0294 | 0.0000 | 0.0927 | 0.0407 | +0.5% | **NEG** |

Per-dataset (addspec vs cf):
- sin_irr: 0.0381 → 0.0294 (**-22.8%**, smaller than r200 spec -34.6%)
- structured_irr: 0.0001 → 0.0000 (~0%, slight improvement)
- random_irr: 0.0834 → 0.0927 (**+11.2%**, similar to r200 spec +12.2%)

## Verdict

**NEGATIVE (54th)** — additive spectral gating does NOT
recover random performance. Random still regresses +11.2%
(essentially same as r200 spec +12.2%).

**Surprising**: sin improvement is SMALLER than r200 spec
(-22.8% vs -34.6%). The linear g_branch partially dilutes
the spectral signal.

## Per-dataset analysis

### sin_irr — SMALLER WIN than r200 spec
- cf: 0.0398 / 0.0363 (mean 0.0381)
- spec: 0.0269 / 0.0230 (mean 0.0249) — r200 winner
- addspec: 0.0335 / 0.0253 (mean 0.0294)
- **-22.8%** (vs r200's **-34.6%**)
- Additive dilution: g_combined = g + g_spec doubles fast
  path magnitude, causing partial cancellation in tau_eff
  mixing

### structured_irr — neutral
- cf: 0.0001 / 0.0001 (mean 0.0001)
- spec: 0.0000 / 0.0000 (mean 0.0000)
- addspec: 0.0000 / 0.0001 (mean 0.0000)
- Already near-perfect for all

### random_irr — SAME LOSS as r200
- cf: 0.0803 / 0.0866 (mean 0.0834)
- spec: 0.0907 / 0.0965 (mean 0.0936, +12.2%)
- addspec: 0.0966 / 0.0888 (mean 0.0927, +11.2%)
- Linear path didn't help — spectral still dominates
- Both seeds worse

## Pattern (47 + 23 + 53 = 123 → 47 + 23 + 54 = 124)

- 47 strictly positive (unchanged)
- 23 target-dep (unchanged)
- **54 negatives** (UP from 53, +1)
- Total: **124 mechanism classes**

## Why additive spectral gating failed (H2: gradient signal split)

Hypothesis was H1 (combine linear noise-robust + spectral
periodic → SP). Actually got H2: **gradient signal splits
between linear and spectral paths, both contribute noise
to the fast path**:

1. **Linear g_branch has its own gradient signal**
   (noise-robust), but spectral_g ALSO adds gradient
   signal. Both update tau_eff indirectly via h_t.
2. **Doubled fast path magnitude** (g + g_spec) means
   tau_eff mixing distributes contribution differently —
   sin improvement reduced (less spectral dominance).
3. **Random unchanged** — spectral path still dominates
   the gradient because it sees more variance.

The fix would be to **gate spectral contribution by data
complexity** (e.g., learn to disable spectral on noisy data).
Or use a **convex combination** with learnable λ:

```
g_combined = (1-λ) * g_branch + λ * spectral_g
```

## Critical implementation details

1. **Linear g_branch preserved** (not replaced) — adds
   264 params per cell vs r200
2. **spec_mask still 30 params per cell**
3. **+90 params total vs r187 baseline** (3 cells × 30
   spec_mask = +90)
4. **g_combined = g + g_spec** (raw sum, no scaling)

## Why this is a useful NEG

1. **Disproves H1** — additive linear + spectral does
   NOT compose to SP
2. **Confirms H2** — gradient signal split hurts both
   sin (less improvement) and random (no recovery)
3. **Suggests gating** is needed — pure additive
   composition doesn't work

## Comparison with r192-r200

| Round | Mechanism | Best sin | Best struct | Best random | Best mean | Verdict |
|-------|-----------|----------|-------------|-------------|-----------|---------|
| 192 | input noise | -16% | +6% | -26% | -24% | **SP** |
| 196 | dropconnect | -14% | +63% | -3% | 0% | **NEG** |
| 197 | mixup | +272% | 0% | +37% | +130% | **NEG** |
| 198 | rk4 | -19% | 0% | +11% | +1.6% | **TD** |
| 199 | adadt | -1.3% | 0% | +9% | +5.8% | **NEG** |
| 200 | spec | **-34.6%** | 0% | +12% | -2.5% | **TD** |
| 201 | **addspec** | -22.8% | 0% | +11% | +0.5% | **NEG** |

addspec gets LESS sin improvement than spec and NO random
recovery. Replacing (r200) beats adding (r201).

## Next ideas

1. **Learnable λ gating** — λ * spec_g with λ learned per
   timestep based on noise estimate
2. **Convex combination** — (1-λ) * g_branch + λ * spec_g
3. **Multi-resolution** — wavelet decomposition then
   spectral on each scale
4. **Spectral on input only** — different signal path
5. **Move on** — r200 stands as the spectral gating winner

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
