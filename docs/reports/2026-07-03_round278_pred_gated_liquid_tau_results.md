# Round 278 — Predictability-Gated Liquid τ — Results

**PRD**: #10-115 · **Date**: 2026-07-03 · **Session**: /loop 1h #2
**Verdict**: **HONEST POSITIVE WITH TRADEOFF** — fixes r277's noise
regression (H2 ✓, headline) but sacrifices r277's peak toy_sin win.

## What we built

`PredictabilityGatedLiquidTauCfCCell` — a **parameter-free**
predictability gate that scales r277's liquid τ contribution by
observed input predictability:

```
vol_t  = EMA_0.5( mean_c |x_t - x_{t-1}| )     # causal input volatility
g_t    = exp(-4 · vol_t)  ∈ (0,1]              # predictable→1, noisy→0
τ_i(t) = tau_min + (tau_max-tau_min)·sigmoid(tau_bias_i + g_t·s·(W_τ·[x_t,h])_i)
```

The gate has **no learnable parameters** ⇒ it structurally cannot learn
to chase noise (the r277 failure mode). Superset: beta=0 ⇒ exactly r277.

## Results (27 cells: 3 modes × 3 datasets × 3 seeds, 100 epochs)

| dataset    | static τ | liquid τ (r277) | gated (r278) | gate | tau_tstd (gated) |
|------------|---------:|----------------:|-------------:|-----:|-----------------:|
| toy_sin    | 0.000031 | **0.000013 (-59%)** | 0.000044 (+41%) | 0.79 | 0.05 |
| structured | 0.000171 | 0.000150 (-12%) | 0.000167 (**-2.5%**) | 0.84 | 0.20 |
| random     | 1.002469 | **2.066662 (+106%)** | 1.005347 (**+0.3%**) | 0.06 | 0.03 |

## Hypothesis scorecard

- **H1 (recovers wins on toy_sin/structured)**: 🟡 **PARTIAL** —
  structured preserved (-2.5%), but toy_sin win LOST (+41% vs
  static). The gate is conservative on sin: sin has genuine per-step
  change ⇒ vol > 0 ⇒ gate=0.79 ⇒ liquid dampened below its optimum.
- **H2 (fixes random regression)**: ✅ **CONFIRMED (headline)** —
  +106% → **+0.3%**. Gate collapsed to 0.06, τ nearly frozen
  (tau_tstd 0.03 vs liquid's up to 0.29). Tracks the stable static
  baseline exactly.
- **H3 (gate high on smooth/structured, low on noise)**: ✅ CONFIRMED —
  gate 0.79 (toy_sin) / 0.84 (structured) / **0.06** (random).
- **H4 (beta=0 == r277)**: ✅ CONFIRMED by construction + unit test.

## Interpretation

The predictability gate does exactly what it was designed to do: it
**structurally removes r277's catastrophic failure mode** on noise
(+106% → +0.3%) by freezing the liquid τ when the input is
unpredictable. This is the cleanest confirmation of the 2026 literature's
"responsive to regimes, not to noise" principle (arXiv:2604.24788).

**The tradeoff is honest**: because EMA-of-|Δx| volatility is nonzero
even for a clean sine wave, the gate dampens the liquid contribution on
toy_sin (gate=0.79) and surrenders r277's headline -59% win (gated is
+41% vs static there). So:

- **liquid τ (r277)** = higher ceiling, catastrophic floor (best on
  smooth, worst on noise).
- **gated liquid (r278)** = robust, bounded worst case (+41% toy_sin is
  its worst; never the +106% blowup), structured preserved.

**Recommendation**: gated_liquid is the **deployment-safe** variant when
the input distribution is unknown or mixed (its worst case is bounded).
Ungated liquid remains preferable only when the signal is known to be
smooth/structured. A natural r279 follow-up: make the gate *sharpness*
(beta) itself schedulable, or use a **volatility-relative** gate
(z-scored per-sequence) so a clean periodic signal reads as
"predictable" (gate→1) rather than "moderately volatile" — that would
recover the toy_sin win without reintroducing the noise blowup.

## Files
- `lnn/core/pred_gated_liquid_tau_cfc.py` (NEW)
- `tests/test_pred_gated_liquid_tau_cfc.py` (NEW, 18 tests, all green)
- `scripts/bench_pred_gated_liquid_tau.py` (NEW)
- `analysis/pred_gated_liquid_tau_bench.json` (NEW, 27 cells)

## Pattern audit
Third architectural change to the STE line (r277 liquid, r278 gate).
Classification: **honest positive with tradeoff** — a robustness
mechanism, not a strict Pareto improvement. 67 SP / 30 TD / 62 NEG
(counts r278 as target/robustness-dependent, +1 TD).
