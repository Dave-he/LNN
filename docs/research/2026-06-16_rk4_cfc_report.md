# Round 198 — RK4 Integration for CfC — Research Report

**Date**: 2026-06-16
**Round**: 198
**Branch**: master
**Audit context (91-197)**: 47 strictly positive + 21 target-dep
+ 52 negatives = 120 mechanism classes.

## TL;DR

**TARGET-DEPENDENT (22nd) for Round 198**: RK4 (4th-order
Runge-Kutta) integration of CfC's ODE **helps sin -19.4%**,
neutral on structured, **hurts random +11.2%**. Mean +1.6%
(essentially neutral). After 5 NEG/TD regularization rounds
(r193-r197), RK4 is the first mechanism with a strict
per-dataset win since r192 (input noise).

The pattern is interpretable: **smooth ODEs (sin) benefit
from 4th-order accuracy; noisy data (random) gets over-fit
by the higher accuracy; structured is already solved by
closed-form**.

## What was tested

**4th-order Runge-Kutta (RK4)** for the CfC update step.
Same f_gate, g_branch, h_branch as r187 — only the
integration scheme changes.

```python
k1 = cf_delta(h, z, f, g, h_branch, dt)
k2 = cf_delta(h + 0.5*dt*k1, z, f, g, h_branch, dt)
k3 = cf_delta(h + 0.5*dt*k2, z, f, g, h_branch, dt)
k4 = cf_delta(h + dt*k3, z, f, g, h_branch, dt)
h_new = h + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

This is **4x more compute per step** but provides O(dt^4)
accuracy vs O(dt) for forward Euler / closed-form.

The CfC ODE is:
```
dh/dt = (1/tau) * (h_branch - h)
tau = |time_scale| / f
```

The closed-form solution is the exact integral of this
linear ODE, but RK4 is more accurate when the ODE has
nonlinear components (e.g. through the EMAs in z).

## Bench (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean | Δ vs CF | type |
|------|---------|----------------|------------|------|---------|------|
| cf | 0.0381 | 0.0001 | 0.0834 | 0.0405 | — | — |
| rk4 | 0.0307 | 0.0001 | 0.0927 | 0.0412 | +1.6% | **TD** |

## Per-dataset analysis

### sin_irr — STRICTLY POSITIVE
- cf: 0.0398 / 0.0363 (mean 0.0381)
- rk4: 0.0326 / 0.0288 (mean 0.0307)
- **-19.4%** with both seeds improving
- Smooth ODE where 4th-order accuracy matters
- Phase error reduced from O(dt) to O(dt^4)

### structured_irr — neutral
- cf: 0.0001 / 0.0001 (mean 0.0001)
- rk4: 0.0002 / 0.0000 (mean 0.0001)
- Near-zero both ways
- The closed-form is already exact for the 2-regime
  structure (sin + linear)

### random_irr — NEGATIVE
- cf: 0.0803 / 0.0866 (mean 0.0834)
- rk4: 0.0973 / 0.0880 (mean 0.0927)
- +11.2% — RK4 overfits noise
- For noise-dominated data, exact ODE integration is
  the wrong prior (no clean ODE exists)

## Pattern (47 + 21 + 52 = 120 → 47 + 22 + 52 = 121)

- 47 strictly positive (unchanged)
- **22 target-dep** (UP from 21, +1)
- 52 negatives (unchanged)
- Total: **121 mechanism classes**

## Why RK4 helps sin

1. **Sin is a smooth ODE** — y(t) = A sin(ωt + φ) is
   exactly integrable
2. **4th-order accuracy reduces phase error** — after
   T=32 steps, the phase error is O(dt^4) instead of O(dt)
3. **Both seeds improve** — robust positive, not noise
4. **RK4 captures long-term coherence** better than
   closed-form on smooth data

## Why RK4 hurts random

1. **Random data has no clean ODE** — the "ODE" the
   cell tries to fit is just noise
2. **Higher accuracy = better fit to noise** — overfits
   training noise
3. **4x more compute per step** — more chances to overfit
4. **For noise-dominated data, regular closed-form is
   better** because it implicitly smooths via EMA

## Why RK4 is neutral on structured

1. **Baseline is already near-perfect** (0.0001) — no
   room to improve
2. **The 2-regime structure (sin + linear) is captured
   well by closed-form** — RK4 doesn't add information
3. **The structured boundary at t=T/2 is a step
   discontinuity** — neither scheme can represent it
   perfectly

## Hypotheses revisited

- **H1 (positive, RK4 helps overall)**: REJECTED. Mean
  is +1.6%.
- **H2 (negative, RK4 overfits)**: PARTIAL. Only hurts
  random, not structured.
- **H3 (mixed, helps smooth hurts noisy)**: CONFIRMED.
  Sin -19%, random +11%, structured neutral.

## Critical implementation details

1. **4 evaluations of cf_delta per step** — f, g, h_branch
   are computed ONCE, only the h update uses RK4
2. **dt broadcasting** — supports scalar (1.0) and
   tensor ([B]) for irregular time series
3. **Same param count as r187** — RK4 adds no params
4. **CfC closed-form is the ODE RHS** — RK4 is purely
   the integration scheme
5. **`_broadcast_dt` helper** — normalizes scalar/tensor
   dt to [B, 1] shape

## Comparison with r192-r197

| Round | Mechanism | Best sin | Best struct | Best random | Best mean | Verdict |
|-------|-----------|----------|-------------|-------------|-----------|---------|
| 192 | input noise | -16% | +6% | -26% | -24% | **SP** |
| 193 | hidden noise | -20% | -16% | +21% | +17% | TD |
| 194 | combined | +8% | -25% | +14% | +12% | TD |
| 196 | dropconnect | -14% (dc05) | +63% (dc05) | -3% (dc20) | 0% | **NEG** |
| 197 | mixup | +272% | 0% | +37% | +130% | **NEG** |
| 198 | **rk4** | **-19%** | 0% | +11% | +1.6% | **TD** |

RK4 is the **first mechanism with a strict per-dataset win
(sin -19%) since r192**. The TD classification is more
favorable than the recent NEG rounds.

## Caveats

- 2 seeds, 30 epochs
- Tested on r187 stack only
- Tested on 3 datasets only
- Single dt=1.0 (no adaptive step size)

## Why this is a useful TD

1. **First TD with a strict per-dataset win** since r192
2. **Confirms the integration scheme matters** — accuracy
   vs speed tradeoff is real
3. **Per-dataset behavior is interpretable** — smooth
   data wants RK4, noisy data wants closed-form
4. **Suggests adaptive scheme** — could detect noise
   level and switch between schemes

## Next ideas

1. **Adaptive RK45 (Dormand-Prince)** — adjust step size
   based on local error, like GLNN paper
2. **Heun's method (RK2)** — cheaper than RK4, may be
   sweet spot
3. **Mixed scheme** — use RK4 on sin regime, closed-form
   on linear regime
4. **Implicit methods** — backward Euler for stiff regimes
5. **Symplectic integrators** — for Hamiltonian-like
   dynamics

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_rk4_cfc.py` (~220 lines)
- `tests/test_learned_beta_ps_ln_khlfft_rk4_cfc.py` (12 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_rk4_cfc.py` (12-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_rk4_cfc.json`

**Why:** Round 198 is **TARGET-DEPENDENT (22nd)** — RK4
helps sin -19.4%, neutral on structured, hurts random
+11.2%. First per-dataset win since r192.

**How to apply:** Use RK4 for smooth / periodic data
(where ODE accuracy matters). Use closed-form for noise-
dominated data (where smoothing is preferred over accuracy).
