---
title: "PRD #10-118 — Blend Gate max(velocity, acceleration) for Liquid τ"
round: 280
date: 2026-07-03
author: "Claude (r280 /loop session)"
status: "draft"
parent: "r279 accel-gated liquid τ (+1 SP; toy_sin -77.5%, structured +0.4% neutral)"
variant: "B"
---

> **Selected** (round 280, 2026-07-03): a max(vel, accel) blend gate is
> predicted to be the first STRICT PARETO liquid-τ default — pre-bench
> signal analysis shows blend recovers accel's toy_sin gate (0.81) AND
> vel's structured gate (0.90) while keeping both collapsed on noise
> (0.08). Directly composes two proven r278/r279 mechanisms.

# PRD #10-118 — Blend Gate max(velocity, acceleration)

## 目标
Turn gated liquid τ into a strict Pareto default by gating on the
**max** of the r278 velocity gate and the r279 acceleration gate:
recover r279's toy_sin win AND r278's structured edge simultaneously,
without reopening the noise regression.

## 用户故事
- As an STE-line maintainer, I can enable ONE gate that is best (or
  tied-best) on every dataset — smooth, structured, and noise — so I
  never have to choose between gated_vel and gated_accel.
- As a researcher, I can test whether the r278↔r279 tradeoff is real
  or dissolvable by a simple gate composition, so I learn whether the
  two predictability signals are complementary.
- As a downstream user, I get a gate that trusts liquid τ whenever
  EITHER volatility OR acceleration says "predictable".

## Mechanism (parameter-free blend gate)
```
vol1_t  = EMA_γ(|Δ¹x|)          # velocity volatility (r278)
vol2_t  = EMA_γ(|Δ²x|)          # acceleration volatility (r279)
g_vel   = exp(-β · vol1_t)
g_acc   = exp(-β · vol2_t)
g_t     = max(g_vel, g_acc)     # trust liquid if EITHER says predictable
τ_i(t)  = tau_min + (tau_max - tau_min) *
          sigmoid( tau_bias_i + g_t · s · (W_τ·[x_t, h])_i )
```
Rationale: a smooth-fast sine has low acceleration (g_acc high) even
though velocity is high (g_vel low) → max keeps it high. A structured
signal has low velocity between jumps (g_vel high) even though jumps
spike acceleration (g_acc dips) → max keeps it high. Pure noise has
BOTH high → both gates collapse → max stays near 0. Measured blend
gates (β=4): sine 0.81, structured 0.90, noise 0.08.

- Parameter-free ⇒ cannot chase noise.
- `gate_mode ∈ {'velocity'(r278), 'acceleration'(r279), 'blend'}`.

## 引擎层职责 (canonical)
- `lnn/core/blend_gated_liquid_tau_cfc.py` (NEW) —
  `BlendGatedLiquidTauCfCCell(AccelGatedLiquidTauCfCCell)` that tracks
  BOTH EMA volatilities per-step and gates on their max.
  `gate_mode='velocity'` ⇒ r278, `'acceleration'` ⇒ r279 (supersets).

## 游戏层职责
- `tests/test_blend_gated_liquid_tau_cfc.py` (NEW, ≥14 tests):
  supersets (velocity ≡ r278, acceleration ≡ r279 to 1e-6 on same
  object), blend ≥ both component gates elementwise, sine gate > 0.7,
  structured gate > 0.8, noise gate < 0.15, grad flow, shapes.
- `scripts/bench_blend_gated_liquid_tau.py` (NEW) — modes
  static / liquid / gated_vel / gated_accel / gated_blend.
- `analysis/blend_gated_bench.json`, report.

## 验收标准
- H1 (headline, strict Pareto): gated_blend ≤ min(gated_vel,
  gated_accel) test_mse on EVERY dataset (best-of-both). [bench]
- H2: gated_blend recovers toy_sin toward -77.5% (like accel). [bench]
- H3: gated_blend recovers structured toward -2.5% (like vel), fixing
  r279's +0.4% neutrality. [bench]
- H4: gated_blend keeps random fix (Δ% ≤ +5% vs static). [bench]
- H5: gate_mode='velocity'≡r278 and 'acceleration'≡r279 exactly.
  [unit tests, 1e-6]

## 实现难度
**M** (2-6h). New subclass (~150 LOC) + ~14 tests + bench (reuse r279
harness, add one mode). 5 modes × 3 datasets × 3 seeds = 45 cells
≈ 40 min.

## 风险
- Blend noise gate (0.08) is slightly higher than accel's (0.05), so
  a touch more liquid may leak on noise → random could be marginally
  worse than gated_vel's +0.3%. Bench H4 checks it stays ≤ +5%.
- If blend is NOT strictly Pareto (e.g. worse than accel on toy_sin
  due to the small noise leak generalizing), we learn max-composition
  isn't free — still a useful result vs the clean theory.
- Load-bearing assumption: the two gates are complementary (one is
  high exactly when the other is low on predictable data). Pre-bench
  signal analysis supports this; the bench confirms in-training.
