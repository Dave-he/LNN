---
title: "PRD #10-115 — Predictability-Gated Liquid τ (fixes r277 noise regression)"
round: 278
date: 2026-07-03
author: "Claude (r278 /loop 1h session #2)"
status: "complete"
parent: "r277 liquid τ (target-dependent positive) + 2026 nonstationarity audit"
---

# PRD #10-115 — Predictability-Gated Liquid τ

## Motivation

Round 277 introduced input-dependent (liquid) τ on the STE-CfC and
found a **target-dependent positive**:

| dataset    | liquid τ vs static |
|------------|-------------------:|
| toy_sin    | **-59.2%** (win)   |
| structured | **-12.3%** (win)   |
| random     | **+106.2%** (hurts)|

The failure mode is unambiguous: on **pure i.i.d. noise** the τ gate
chases unpredictable jitter and over-adapts (tau_tstd was highest on
random, 0.29), destabilising the recurrence.

The 2026 literature frames exactly this tension:
- *Liquid NN for Natural-Gas Spot Price* (arXiv:2604.24788) — dynamics
  must "limit responsiveness when market regimes shift rapidly";
  nonstationary prediction needs responsiveness to *structured* regime
  change but not to noise.
- *Urban-flood CfC* (Liu et al., Water Research 2026) & *SCTP-Net*
  (2026) — discrete networks fail on nonstationarity, but naive
  adaptivity is unstable.

The r277 report already named the fix: **gate the liquid strength on a
signal-predictability estimate** (analogous to the r99 reliability gate).

## Mechanism (parameter-free predictability gate)

```
vol_t  = EMA_γ( mean_c |x_t - x_{t-1}| )          # causal input volatility
g_t    = exp( -beta · vol_t )   ∈ (0, 1]          # predictable→1, noisy→0
τ_i(t) = tau_min + (tau_max - tau_min) *
         sigmoid( tau_bias_i + g_t · s · (W_τ·[x_t, h])_i )
```

- `g_t` scales the **liquid contribution only**. Smooth/structured
  input (low vol) ⇒ g_t≈1 ⇒ full liquid (recovers r277). Noisy input
  (high vol) ⇒ g_t≈0 ⇒ τ collapses to static bias (recovers r267).
- **The gate has NO learnable parameters** ⇒ it *cannot* learn to chase
  noise. This structurally forbids the r277 failure mode.
- Strict superset: beta=0 ⇒ g_t≡1 ⇒ exactly r277; g_t→0 ⇒ exactly r267.

## Hypotheses

- **H1**: gated liquid recovers r277's wins on toy_sin/structured
  (g_t≈1 there). [predicted CONFIRM]
- **H2** (headline): gated liquid FIXES r277's random regression
  (g_t≈0 ⇒ stable static τ). [predicted CONFIRM]
- **H3**: gate value high on toy_sin/structured, low on random
  (mechanism check). [smoke test already shows gate 0.79 smooth vs
  0.06 random]
- **H4**: beta=0 exactly reproduces r277. [CONFIRM by construction +
  unit test]

## Bench Config

- 3 modes (static_tau, liquid_tau, gated_liquid) × 3 datasets × 3 seeds
  = 27 cells
- 100 epochs, lr=1e-2, batch=16, hidden=192, T=64, density=0.3,
  ste_temperature=1.0, entropy_lambda=0.1
- gated_liquid: pred_gate_beta=4.0, ema_gamma=0.5
- Diagnostics: gate_mean/min/max, tau_temporal_std

## Files

- `lnn/core/pred_gated_liquid_tau_cfc.py` (NEW)
- `tests/test_pred_gated_liquid_tau_cfc.py` (18 tests, NEW)
- `scripts/bench_pred_gated_liquid_tau.py` (NEW)
- `analysis/pred_gated_liquid_tau_bench.json` (NEW)

## Pattern Audit

Second architectural change to the STE line (after r277). If H2 holds,
this converts r277 from *target-dependent* to a *strictly safe*
mechanism (never worse than the better of static/liquid). Classification
pending bench.
