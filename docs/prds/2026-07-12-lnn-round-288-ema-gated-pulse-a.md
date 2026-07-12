---
title: "PRD #10-129 — EMA-Smoothed Binary-Gated Pulse (r288)"
round: 288
date: 2026-07-12
author: "Claude (r288 /loop session)"
status: "selected"
parent: "r287 binary-gated pulse (HONEST NEGATIVE); r286 sqrt; r285 linear; r284 pulse"
paper: "arXiv:2603.00153 (Sharma 2026-03) — extension with EMA-smoothed gate"
variant: "A"
---

> **Selected** (round 288, 2026-07-12): four rounds of pulse variants
> (r284-r287) revealed an anti-correlated trade-off curve: stronger
> gating → better noise safety but worse gap-robustness. The
> *fundamental* failure: ANY gate interrupts the pulse when `g_t`
> momentarily dips during a gap, breaking the "continuous endogenous
> rhythm" claim. This round attacks that root cause by **smoothing the
> gate itself** with an EMA: `g_eff_t = α · g_t + (1-α) · g_eff_{t-1}`.
> On structured + gap: when input drops out, `g_t` spikes DOWN
> momentarily, but the EMA-smoothed `g_eff` stays high → binary mask
> stays on → pulse fires continuously through the gap. On noise:
> `g_t` is *consistently* low → EMA-smoothed `g_eff` collapses →
> mask stays off → no noise chasing. This decouples the two axes.

# PRD #10-129 — EMA-Smoothed Binary-Gated Pulse

## 目标
Test whether EMA-smoothing the per-step gate before binary thresholding
recovers r284's structured gap-robustness (H1) while still suppressing
noise chasing (H2/H3).

## 用户故事
- As a gate-line maintainer, I find the first pulse variant that is
  strict-positive default in the r284-r287 line.
- As a researcher, I confirm that the multiplicative-gate failure is
  specifically about *per-step* gate fluctuations during gaps, not
  about gating per se.

## 引擎层职责 (canonical)
- `lnn/core/ema_gated_pulse_cfc.py` (NEW, ~80 LOC delta on r287):
  `EmaGatedPulseCfCCell` subclass of `BinaryGatedPulseCfCCell`. Adds
  EMA state `g_ema` updated per step:
    `g_ema_t = α · g_t + (1-α) · g_ema_{t-1}`
  Threshold uses `g_ema_t` not `g_t` directly. Two new params:
  `ema_alpha: float = 0.3` (smoothing strength), `g_ema_init: float = 1.0`
  (start high so first-step mask is on; alternative: 0.5).
- Strict superset:
  - `ema_alpha=1.0` (no smoothing) ≡ r287
  - `threshold=0` ≡ r284
  - `threshold=10` ≡ r280

## 游戏层职责
- `scripts/bench_ema_gated_pulse.py` (NEW, ~310 LOC): modes =
  {static_tau, blend_gated, pulse_sin [r284], ema_tau α=0.3 τ=0.5,
  ema_tau α=0.5 τ=0.5, ema_tau α=0.7 τ=0.5, ema_tau α=0.3 τ=0.3};
  3 datasets × 2 seeds × 50 epochs.
- `analysis/ema_gated_pulse_bench.json` (NEW, 42 cells).
- `docs/research/2026-07-12_round288_ema_gated_pulse_report.md`.

## 验收标准 (H1-H6)
- H1 (structured gap_ratio ≤ r284 = 61) — the headline test.
- H2 (random Δ% ≤ +5% vs blend) at τ ∈ {0.3, 0.5}.
- H3 (random pulse_amp ≤ 0.20).
- H4 (H1 ∧ H2 ∧ H3) → strict-positive default — the first in the line.
- H5 (ema_alpha=1.0 ≡ r287) — unit test.
- H6 (threshold=0 ≡ r284) — unit test.

## 实现难度
**M** (2-3h). ~80 LOC cell delta + ~10 unit tests + ~280 LOC bench.

## 风险
- If H1 ✗ even at α=0.3: the EMA decays too fast during long gaps
  → try α=0.1 (heavier smoothing) or hard-init `g_ema_init=1.0` (so
  the pulse fires until g_t consistently says otherwise).
- If H1 ✓ but H2 ✗: the EMA carries over too much history → τ=0.7
  (stricter threshold).
- If both ✗: the gate-smoothing strategy is wrong; consider EMA on the
  *pulse amplitude* itself instead of the gate (e.g. `A_eff = α·A_target + (1-α)·A_eff_prev`).
- If all four rounds r285-r288 fail: abandon the pulse line entirely.