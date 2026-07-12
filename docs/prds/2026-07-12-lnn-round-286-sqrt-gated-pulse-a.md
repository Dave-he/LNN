---
title: "PRD #10-127 — sqrt-Gated Pulse Amplitude (r286)"
round: 286
date: 2026-07-12
author: "Claude (r286 /loop session)"
status: "selected"
parent: "r285 predictability-gated pulse (HONEST NEGATIVE-WITH-NUANCE); r284 pulse; r280 blend gate"
paper: "arXiv:2603.00153 (Sharma 2026-03) — extension with shape-preserving gate"
variant: "A"
---

> **Selected** (round 286, 2026-07-12): r285 multiplied the r284
> oscillatory pulse by the r280 gate `g_t`, which preserved noise
> suppression at the cost of (a) destroying the structured gap-robustness
> (gap_ratio 61→394) and (b) amplifying the parameter chase on `A`
> (0.40→0.71). The failure mode is **shape**: a linear multiplicative
> gate is too aggressive on high-g steps (structured) where the pulse
> needs to carry state through gaps. This round tries a **shape-
> preserving gate** — `sqrt(g_t)` — which keeps more amplitude on
> structured (sqrt(0.8)=0.89 vs 0.80) while still attenuating noise
> (sqrt(0.1)=0.32 vs 0.10). It also changes the gradient on `A` (now
> scaled by `1/sqrt(g_t)` not `1/g_t`), which may slow A-chase on noise.

# PRD #10-127 — sqrt-Gated Pulse Amplitude

## 目标
Test whether a **shape-preserving** gate `pulse = sqrt(g_t) · A · sin(...)`
keeps the r284 structured gap-robustness (H1) while still suppressing
the noise amplitude growth (H2/H3). If both hold, this is the **first
strict-positive pulse variant** in the r284/r285 line.

## 用户故事
- As a gate-line maintainer, I find the first pulse variant that is
  safe by default (no per-dataset toggle).
- As a researcher, I confirm the **shape hypothesis**: linear gates
  destroy structured signal; shape-preserving gates preserve it.
- As a downstream user, I get a single cell that works on structured
  AND noisy data without manual `pulse_strength` tuning.

## 引擎层职责 (canonical)
- `lnn/core/sqrt_gated_pulse_cfc.py` (NEW, ~50 LOC delta on r284):
  `SqrtGatedPulseCfCCell` subclass of `PulseGatedLiquidTauCfCCell`.
  Same `_pulse_term(t, T, h, noise_drive, gate=None)` extension as r285
  but applies `gate = gate.clamp_min(0).sqrt()` (i.e. sqrt(g_t)).
  New flag `gate_pulse_shape: str = 'sqrt'` controls shape:
  `'sqrt'` (default r286), `'linear'` (≡ r285 for backwards-compat),
  `'none'` (≡ r284).
- Strict superset guarantees:
  - `gate_pulse_shape='none'` ≡ r284 (PulseGatedLiquidTauCfCCell)
  - `gate_pulse_shape='linear'` ≡ r285 (PredictabilityGatedPulseCfCCell)
  - `pulse_strength=0` ≡ r280 blend cell

## 游戏层职责
- `scripts/bench_sqrt_gated_pulse.py` (NEW, ~300 LOC):
  4 modes (static_tau, blend_gated, pulse_sin [r284],
  sqrt_pulse [r286]) × 3 datasets × 2 seeds × 50 epochs.
  Clean + gap p=0.3 evaluation; report clean MSE, gap_ratio,
  pulse_amp final.
- `analysis/sqrt_gated_pulse_bench.json` (NEW, 24 cells).
- `docs/research/2026-07-12_round286_sqrt_gated_pulse_report.md`.

## 验收标准 (H1-H6)
- H1 (structured gap_ratio ≤ r284): structured gap_ratio ≤ 61.
- H2 (random Δ% ≤ +5% vs blend): safety restored.
- H3 (random pulse_amp ≤ 0.20): A-chase slowed vs r285's 0.71.
- H4 (H1 AND H2 AND H3 ALL pass): **strict-positive default** (the
  first in the r284/r285/r286 line).
- H5 (gate_pulse_shape='none' ≡ r284): unit test.
- H6 (gate_pulse_shape='linear' ≡ r285): unit test.

## 实现难度
**M** (2-3h). ~50 LOC cell delta on r285 (just change the gate
function from `gate` to `gate.sqrt()`) + ~10 unit tests + ~150 LOC
bench delta.

## 风险
- If H1 ✗: HONEST NEGATIVE — sqrt-gate still too aggressive; try
  cube-root or `g_t^{0.1}` (gentler shape) in r287.
- If H1 ✓ but H2 ✗: HONEST TARGET-DEPENDENT — sqrt gives partial
  improvement; combine with noise-mode-only gate.
- If both ✗: back off to r284 as the recommended default; the
  multiplicative-gate family is wrong for this pulse architecture.