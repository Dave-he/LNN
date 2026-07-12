---
title: "PRD #10-126 — Predictability-Gated Pulse Amplitude (r285)"
round: 285
date: 2026-07-12
author: "Claude (r285 /loop session)"
status: "selected"
parent: "r284 pulse-augmented gated liquid τ; r280 blend gate"
paper: "arXiv:2603.00153 (Sharma 2026-03) — extension with r280 gate"
variant: "A"
---

> **Selected** (round 285, 2026-07-12): r284 added an endogenous
> oscillatory pulse to the gated liquid-τ cell. It bought real
> gap-robustness on structured data (6×) but **broke the gate line's
> parameter-free ⇒ noise-safe invariant** — the learned amplitude grew
> 4× on random noise (+44.6% MSE). The r284 report itself recommended
> gating the pulse amplitude by the r280 predictability score `g_t`.
> This round does exactly that: `pulse = g_t · A · sin(...)` — zero new
> parameters, zero new loss, zero new schedule — and asks whether that
> gates-pulse cell (a) keeps the structured gap-robustness AND (b)
> restores noise safety. If both hold, r284's +1 TD upgrades to +1 SP.

# PRD #10-126 — Predictability-Gated Pulse Amplitude

## 目标
Test whether multiplying the r284 oscillatory pulse by the r280
predictability gate `g_t` (per-step scalar ∈ (0,1]) preserves the
structured-data gap-robustness of the pulse while suppressing its
amplitude growth on noise. The gating is parameter-free and loss-free.

## 用户故事
- As a gate-line maintainer, I promote the pulse from
  target-dependent-with-caveat to strict-positive default if both
  gap-robustness (H1) and noise safety (H2) hold.
- As a researcher, I get a clean example of *combining internal gate
  innovations with external paper claims* by reusing one scalar that
  already exists in the cell.
- As a downstream user with mixed noise/gappy data, I get a single
  knob (`pulse_strength`) that automatically backs off when input is
  erratic.

## 引擎层职责 (canonical)
- `lnn/core/predictability_gated_pulse_cfc.py` (NEW, ~50 LOC delta on
  r284): `PredictabilityGatedPulseCfCCell` subclass of
  `PulseGatedLiquidTauCfCCell`. Extends `_pulse_term` to accept an
  optional `gate` argument; `forward` passes the per-step
  r280 blend gate `gate = max(g_vel, g_acc)` (shape (B,1)). New flag
  `gate_pulse: bool = True` controls whether to apply the multiplication
  (default True = r285; False = r284 superset).
- Strict superset guarantees:
  - `gate_pulse=False` ≡ r284 (unit test `test_no_gate_equals_r284`)
  - `pulse_strength=0` ≡ r280 (inherited from r284 superset)
  - `gate_pulse=True, pulse_strength=0` ≡ r280 (both gates close)

## 游戏层职责
- `scripts/bench_predictability_gated_pulse.py` (NEW, ~250 LOC):
  4 modes (static_tau, blend_gated, pulse_sin [r284],
  gated_pulse_sin [r285]) × 3 datasets (toy_sin / structured / random)
  × 2 seeds, 50 epochs. Each model evaluated on clean AND
  gap-corrupted test (temporal dropout p=0.3).
- `analysis/predictability_gated_pulse_bench.json` (NEW).
- `docs/research/2026-07-12_round285_predictability_gated_pulse_report.md`
  (NEW, full hypothesis evaluation).

## 验收标准 (H1-H5)
- H1 (robustness preserved): structured gap_ratio (gated_pulse) ≤
  blend (368) AND ≤ r284 (61). [matrix]
- H2 (safety restored, **THE FIX**): random Δ% (gated_pulse vs blend)
  ≤ +5%; r284 was +44.6%. [matrix]
- H3 (amplitude no longer chases noise): on random, final
  `pulse_amp.abs().mean() ≤ 0.20` (r284 grew to 0.40). [diag]
- H4 (superset): `gate_pulse=False` ≡ PulseGatedLiquidTauCfCCell
  forward output bit-equal within float tolerance. [unit test]
- H5 (gating not just clamping): on structured+gap, post-training
  `gate.mean() ≥ 0.5` — the pulse is still active when input is
  predictable, otherwise H1 would trivially hold by H2-clamping. [diag]

## 实现难度
**M** (2-3h). ~50 LOC cell delta (forward + _pulse_term signature
extension) + ~12 unit tests + ~150 LOC bench delta (1 new mode column
on r284 bench). Toy grid is small (24 cells, hidden=128, T=48, 50
epochs) to fit the loop window.

## 风险
- If H5 fails (gate collapses to 0 on structured+gap), gating is too
  aggressive and we lose H1 → HONEST NEGATIVE (gating too strong; try
  sqrt(gate) or EMA-smoothed gate in r286).
- If H1 holds but H2 fails, the gate is too weak on noise → try EMA
  smoothing the gate or using `gate^2` (sharper suppression).
- If both fail, the pulse + gate combination is destructive → back off
  to r284 (TD) and explore a different pulse-conditioning axis.

## 不在 scope
- Adding new hyperparameters (no `gate_β`, no `gate_floor`).
- Changing the r280 blend gate (its behaviour is already audited).
- Other datasets beyond the toy grid (Henry Hub / EMMA rover follow-up
  will be a separate round if r285 passes).