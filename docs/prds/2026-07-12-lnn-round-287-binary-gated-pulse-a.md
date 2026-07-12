---
title: "PRD #10-128 — Binary-Gated Pulse (r287)"
round: 287
date: 2026-07-12
author: "Claude (r287 /loop session)"
status: "selected"
parent: "r286 sqrt-gate pulse (HONEST NEGATIVE); r285 linear-gate; r284 pulse; r280 blend"
paper: "arXiv:2603.00153 (Sharma 2026-03) — extension with additive threshold gate"
variant: "A"
---

> **Selected** (round 287, 2026-07-12): r284/r285/r286 collectively show
> that the **pulse + multiplicative-gate family is fundamentally
> misconfigured** — the trade-off curve between structured gap-robustness
> and random noise safety is roughly linear in gate shape, and no
> multiplicative variant (none / linear / sqrt) achieves strict-positive
> on both axes. The next-direction hypothesis: **abandon multiplicative
> gating entirely** and use an **additive / threshold gate** —
> `pulse = (g_t > τ) · A · sin(...)`. The pulse is *full strength* or
> *exactly zero*; there is no per-step attenuation that the optimizer
> must compensate for with A. Hypothesis: on structured (g ≈ 0.8 > τ) the
> pulse activates fully and recovers r284's gap-robustness; on noise
> (g ≈ 0.1 < τ) the pulse is exactly zero so no noise chasing.

# PRD #10-128 — Binary-Gated Pulse

## 目标
Test whether an additive threshold gate `pulse = (g_t > τ) · A · sin(...)`
decouples the two axes that the multiplicative family cannot:
- **Structured axis**: when `g_t > τ`, the pulse fires at *full* A (not
  attenuated), recovering the r284 gap-robustness.
- **Noise axis**: when `g_t ≤ τ`, the pulse is *exactly zero*, killing
  the noise amplification that plagued r284/r285/r286.

The threshold τ is the only new hyperparameter, swept across
{0.3, 0.5, 0.7} in the bench.

## 用户故事
- As a gate-line maintainer, I find the first pulse variant that is
  safe by default if `τ` is chosen well (likely τ ≈ 0.5 since
  structured has g ≈ 0.8 and noise has g ≈ 0.1).
- As a researcher, I confirm that the **multiplicative-gate failure mode
  is fundamentally about amplitude scaling**, not about the existence of
  a gate — and that threshold gates sidestep it.

## 引擎层职责 (canonical)
- `lnn/core/binary_gated_pulse_cfc.py` (NEW, ~150 LOC): `BinaryGatedPulseCfCCell`
  subclass of `PredictabilityGatedPulseCfCCell`. `_pulse_term` returns
  `pulse = (gate > threshold).float() * raw_pulse`. No new params
  beyond `threshold` (default 0.5).
- Strict superset:
  - `threshold=0` ≡ unconditional pulse ≡ r284 (PulseGatedLiquidTauCfCCell)
  - `threshold=2.0` (impossible to satisfy) ≡ zero pulse ≡ r280 blend

## 游戏层职责
- `scripts/bench_binary_gated_pulse.py` (NEW, ~280 LOC): modes =
  {static_tau, blend_gated, pulse_sin [r284], binary_pulse τ=0.3
  [r287a], binary_pulse τ=0.5 [r287b], binary_pulse τ=0.7 [r287c]};
  3 datasets × 2 seeds × 50 epochs.
- `analysis/binary_gated_pulse_bench.json` (NEW, 36 cells).
- `docs/research/2026-07-12_round287_binary_gated_pulse_report.md`.

## 验收标准 (H1-H6)
- H1 (structured gap_ratio ≤ r284 = 61) at τ ∈ {0.3, 0.5}.
- H2 (random Δ% ≤ +5% vs blend) at τ ∈ {0.3, 0.5}.
- H3 (random pulse_amp ≤ 0.20) — A-chase killed because the optimizer
  gets *zero* gradient on noise steps (no pulse → no A gradient).
- H4 (H1 ∧ H2 ∧ H3) → strict-positive default — **first in the line**.
- H5 (threshold=0 ≡ r284) — unit test.
- H6 (threshold=2.0 ≡ r280) — unit test (or threshold=10).

## 实现难度
**M** (2-3h). ~150 LOC cell + ~12 unit tests + ~280 LOC bench.

## 风险
- If H1 ✗ even at τ=0.3: the threshold gate still loses gap-robustness
  because (g > 0.3) is satisfied on structured but the resulting pulse
  is interrupted when g dips momentarily below 0.3 → HONEST NEGATIVE.
- If H1 ✓ but H2 ✗: τ=0.5 is too low; try τ=0.7 or adaptive τ from
  the empirical g_t distribution.
- If both ✗: pulse + any-gate is wrong; back off to r284 as default
  and consider non-pulse alternatives (r100 SNNL, r99 segment
  reliability gate).