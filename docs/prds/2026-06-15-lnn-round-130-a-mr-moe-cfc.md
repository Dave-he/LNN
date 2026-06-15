# PRD #10-92 — Round 130 MR-MoE CfC (arXiv:2606.12240)

**Date**: 2026-06-15
**Round**: 130
**Status**: Implemented, benched
**Verdict**: TBD (write after bench)

## Goal

Test whether the **Multi-Rate Mixture-of-Experts (MR-MoE)** framework
from arXiv:2606.12240 (Zong, Boker, Eldardiry 2026, Virginia Tech,
NeurIPS 2026 submission) helps our recurrent CfC cell. The paper
proposes K=3 LNN experts each operating at a distinct time constant
(τ₁ ≪ τ₂ ≪ τ₃), plus a **dual attention** module that combines
**feature-level attention** (suppresses noisy input variables) with
**temporal attention** (focuses on informative historical states).

## Reference

- **MR-MoE**: arXiv:2606.12240 (Zong, Boker, Eldardiry, 10 June 2026).
  Multi-Rate Mixture of Experts for Accelerating Liquid Neural Network
  Training. Multivariate time-series prediction on sepsis-like data;
  reports consistent AUROC/AUPRC improvements vs LSTM / monolithic
  LNN / standard MoE.

## Hypothesis

Combining multi-rate expert timescales with dual attention will give
the cell a **3-axis inductive bias** (per-expert τ + feature gate +
temporal gate) that should win on **structured_irr** (regime switch),
where one expert can lock onto the slow drift and another on the fast
transient.

## Design

- K=3 CfCCell experts, each with `n_tau=1` and a distinct
  `tau_init` (0.1, 1.0, 10.0) — fast / medium / slow.
- **Feature-level attention**: per-step input gate `α_t ∈ [0,1]^D` from
  a small MLP over `[x_t; h_prev]`. Applied as `x_t' = α_t ⊙ x_t`.
- **Temporal attention**: softmax over a window of H_prev (last few
  steps) to focus on the most informative historical state.
- **Router**: standard FAME top-K softmax over `[x_t'; h_prev]`
  (reuses `ForecastabilityRouter` from round 78).
- Output: `y_t = Σ_k g_k · expert_k(x_t', h_prev)`.

## Files

- `lnn/core/mr_moe_cfc.py` (NEW, ~250 lines) — MRMoECfCCell + MRMoECfCNetwork
- `tests/test_mr_moe_cfc.py` (NEW, 12+ tests)
- `scripts/bench_mr_moe_cfc.py` (NEW, 12-16 cells)

## Bench (12 cells, 30 epochs, 2 seeds)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | TBD | TBD | TBD | 2545 |
| mr_moe_k3_uniform (all τ=1.0) | TBD | TBD | TBD | TBD |
| mr_moe_k3_multirate (τ={0.1,1,10}) | TBD | TBD | TBD | TBD |
| mr_moe_k3_dualattn (τ={0.1,1,10} + feat+temporal) | TBD | TBD | TBD | TBD |

## Connection to 91-129 audit

- Reuses `CfCCell` (no new neuron family — round 76-78 already
  validated `n_tau`).
- Reuses `ForecastabilityRouter` (round 78 FAME).
- New axis: **dual attention** = `feature_gate + temporal_gate`.
- Compared against: round 78 FAME baseline, round 113 DeepSeek
  shared, round 116 Sigmoid (the 3 most relevant baselines).

## Verdict (filled after bench)

TBD
