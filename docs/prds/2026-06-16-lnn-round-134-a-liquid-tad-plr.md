# PRD #10-96 — Round 134 LiquidTAD-style PLR (arXiv:2604.18274)

**Date**: 2026-06-16
**Round**: 134
**Status**: Implemented, benched
**Verdict**: TBD (write after bench)

## Goal

Bring the **Parallel Liquid-inspired Relaxation (PLR)** operator from
LiquidTAD (arXiv:2604.18274, Sun, Zheng, Xia, Wu, Bao, Zhang, 20 April
2026) into our recurrent CfC toolkit as a **sequence encoder** that
captures the "exponential relaxation prior" of liquid neural dynamics
**without** sequential ODE solving. The paper's central claim: the
exponential decay prior (the α-weighted EMA from LNN closed-form) can
be reformulated as a fully vectorized, non-recursive operator using
standard neural ops, giving linear-time temporal modeling and
hardware-agnostic deployment.

The paper applies PLR inside a feature pyramid for Temporal Action
Detection (THUMOS-14, ActivityNet-1.3) and reports **69.46 % mAP with
only 10.82 M params / 27.17 G FLOPs**, reducing parameters >60 % vs
ActionFormer. We adapt the operator for **1-D sequence modeling**
(regime-switch + multi-frequency synthetic data) and test whether the
"liquid relaxation without ODE solver" idea generalises to time-series
benchmarks.

## Reference

- **LiquidTAD**: arXiv:2604.18274v2 (Sun et al., 20 April 2026).
  *LiquidTAD: Efficient Temporal Action Detection via Parallel
  Liquid-Inspired Temporal Relaxation*. Introduces the PLR operator
  and Hierarchical Decay-Rate Sharing (HDRS) across FPN levels.
  Validated on THUMOS-14 / ActivityNet-1.3.

## Hypothesis

PLR's vectorized EMA — `h_t = α · h_{t-1} + (1-α) · f(x_t)` rewritten
as `h_t = (1-α) · Σ_{k≤t} α^{t-k} f(x_k)` — is mathematically identical
to a single-step ODE-1 relaxation but admits a **closed-form,
parallel form via cumulative weighted sum**. This should:

1. Match or beat vanilla CfC on **multi-frequency** data because the
   EMA-weights all past observations equally in decay — the
   "leaky integrator" regulariser that LNNs are built on.
2. Be **strictly cheaper** than CfC at long horizons (no ODE
   coefficients to compute per step; one matmul + one cumsum-style
   weighted sum).
3. Combine cleanly with a CfC "head" — feed PLR features into a CfC
   for nonlinear gating — to give a **two-axis** design (linear
   relaxation prior + nonlinear gating).

## Design

- **PLR cell**: vectorised exponential relaxation over the time axis.
  - `α ∈ (0,1)` is a learnable scalar (or per-channel vector).
  - `h = (1 - α) · reverse_cumsum(reverse(α^{0..T-1}) · f(x))`
    (equivalently: weighted cumulative sum with `α^{t-k}` kernel).
  - Closed-form, no recurrence, fully parallel over T.
  - Default `α_init = exp(-1/τ)` with τ=1.0 (continuous time
    constant typical of CfC).
- **PLR encoder (1-D sequence)**: stack N PLR cells with channel-wise
  α sharing (analogous to HDRS), feed into a CfC head.
- **Hierarchical Decay-Rate Sharing (HDRS)**: when used in a multi-scale
  feature pyramid, share α across pyramid levels (paper §3.2). For
  1-D we expose this as `share_alpha_across_layers=True` for ablation.
- **Drop-in for CfC**: a `PLRCfCCell` that concatenates a PLR feature
  stream (linear relaxation prior) with a CfC hidden stream
  (nonlinear gating) before producing output.

## Files

- `lnn/core/liquid_tad.py` (NEW, ~220 lines) — PLRCell, PLREncoder,
  PLRCfCCell, PLRConfig.
- `tests/test_liquid_tad.py` (NEW, 12+ tests).
- `scripts/bench_liquid_tad.py` (NEW, 12-16 cells) — vs CfC baseline on
  structured_irr / multi_sin / mackey_glass.

## Expected outcome

- **POSITIVE or NEUTRAL** on multi-frequency / regime-switch tasks.
- **STRICTLY POSITIVE** on parameter count and FLOPs vs CfC (fewer
  ODE coefficients, single matmul + weighted cumsum).
- **Honest negative** flags if PLR loses on tasks where nonlinear
  gating (CfC) is critical — the relaxation prior is linear and
  cannot capture XOR-like regime boundaries on its own.
