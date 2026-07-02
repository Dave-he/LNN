---
title: "PRD #10-114 — Liquid (Input-Dependent) τ on STE-CfC"
round: 277
date: 2026-07-03
author: "Claude (r277 /loop 1h session)"
status: "complete"
parent: "r276 batch-size sweep (completing) + 2026 LTC literature audit"
---

# PRD #10-114 — Liquid (Input-Dependent) τ on STE-CfC

## Motivation

The (τ, λ, hidden, T, d_in, density, batch) hyperparameter sweep of
the STE sparsity line (r267-r276) is essentially complete. This
round changes direction based on a **2026 literature audit** rather
than continuing the sweep.

Every 2026 LTC/CfC paper surfaced in this session emphasises the
**input-dependent time constant** as the core "liquid" advantage on
nonstationary sequences:

- *Efficient Semantic Segmentation via LTC with Adaptive Dynamics*
  (A. Al, 2026) — isolates CfC adaptive dynamics from generic gating.
- *LTC for Water-Level Forecasting in Urban Drainage* (Buczyński,
  2026) — a gating mechanism computes adaptive continuous-time dynamics.
- *Liquid NN for Natural-Gas Spot Price* (arXiv:2604.24788, 2026) —
  dynamic internal state updates for nonstationary price behaviour.

**Gap in our stack**: our base `NeuronWiseCfCCell` uses a
**static, learned per-neuron τ** (`tau_per_neuron` is a plain
`nn.Parameter`, fixed after training):

```python
raw = sigmoid(self.tau_per_neuron)             # (d_h,), independent of x_t
tau = tau_min + (tau_max - tau_min) * raw
```

This is a *half-liquid* base. The entire STE sparsity line (263-276)
was built on it and never restored the defining input-dependent τ.

## Mechanism

Upgrade τ from a static parameter to an input-dependent gate:

```
τ_i(t) = tau_min + (tau_max - tau_min) *
         sigmoid( tau_per_neuron_i + s · (W_τ · [x_t, h_{t-1}])_i )
```

- `tau_per_neuron` — inherited static per-neuron bias (kept).
- `W_τ` — NEW gate `Linear(d_in + d_h → d_h)`, **zero-initialised**.
- `s` = `liquid_tau_strength` — scale on the gate.

Zero-init `W_τ` ⇒ at initialisation τ_i(t) equals the static
per-neuron τ **exactly** ⇒ strict superset of r267. Any liquid
behaviour is *learned*, not imposed.

The STE hard/soft neighborhood mask and the soft-mask entropy penalty
are inherited unchanged from `STEWithEntropy`. Only the per-timestep
τ computation in the forward loop changes.

## Hypotheses

**H1**: liquid τ beats static τ on ≥1 dataset (esp. **structured** —
the analogue of the papers' nonstationary claim).
[predicted: PLAUSIBLE on structured — has regime/segment structure]

**H2**: liquid τ does NOT hurt **toy_sin** (smooth single-freq needs
no adaptation).
[predicted: TIE or mild regression — consistent with our
"fancy mechanisms are a tax on smooth data" pattern]

**H3**: zero-init gate ⇒ training-start equivalence to static τ
(no added instability).
[predicted: CONFIRM by construction — unit-tested]

**H4**: the learned τ actually flows: temporal std of τ across
timesteps > 0 after training.
[predicted: CONFIRM — smoke test already shows tau_tstd 0.05-0.09]

## Bench Config

- 2 modes (static_tau, liquid_tau) × 3 datasets × 3 seeds = 18 cells
- 100 epochs, lr=1e-2, batch=16 (r267-r275 production)
- input_size=1, hidden=192, T=64, density=0.3, ste_temperature=1.0,
  entropy_lambda=0.1
- Diagnostics: tau_temporal_std, tau_dynamic_{mean,min,max}, n_params

## Expected Outcomes

- **Target-dependent positive**: structured improves, toy_sin ties →
  confirms the 2026 papers' nonstationary claim reproduces in our
  toy regime.
- **Honest negative**: all tie → static per-neuron τ already
  sufficient in this 1D toy regime (regime shifts too weak). Both
  outcomes are informative; both get recorded.

## Files

- `lnn/core/liquid_tau_ste_cfc.py` — LiquidTauSTECfCCell (NEW)
- `tests/test_liquid_tau_ste_cfc.py` — 19 unit tests (NEW)
- `scripts/bench_liquid_tau_ste.py` — bench (NEW)
- `analysis/liquid_tau_ste_bench.json` — results (NEW)

## Pattern Audit

This is the **first architectural (not hyperparameter) change** to
the STE line since r265. It re-introduces the liquid property that
the sweep line dropped. Classification pending bench:
- If structured wins & toy_sin ties → +1 target-dependent positive.
- If all tie → +1 honest negative.
