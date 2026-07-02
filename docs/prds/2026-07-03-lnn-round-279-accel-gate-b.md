---
title: "PRD #10-117 — Acceleration-Gated Liquid τ (recover toy_sin without noise risk)"
round: 279
date: 2026-07-03
author: "Claude (r279 /loop session)"
status: "draft"
parent: "r278 predictability-gated liquid τ (+1 SP, H1 partial: toy_sin +41%)"
variant: "B"
---

> **Selected** (round 279, 2026-07-03): directly fixes r278's only
> weakness (toy_sin +41%) with a concrete falsifiable mechanism; if it
> holds, gated liquid τ becomes a strict Pareto default. Higher
> engine/game symmetry and alignment than the β-sweep alternative.

> **Pivot note** (during impl): the originally-selected *relative-
> volatility* gate (z-score vol by its running mean) was killed by
> pre-bench signal analysis — an EMA of a periodic signal's volatility
> is itself periodic, so sine (gate 0.65) did NOT separate from noise
> (gate 0.67). Root-cause analysis found the clean fix below:
> **gate on acceleration (2nd difference), not velocity (1st)**. A
> smooth sine has large velocity but small acceleration; noise has
> large acceleration everywhere. Measured β=4 gates: sine 0.80,
> structured 0.84, noise 0.018 (vs r278's sine 0.49). noise/sine
> volatility ratio jumps 6× → 35×.

# PRD #10-117 — Acceleration-Gated Liquid τ

## 目标
Fix r278's one weakness (toy_sin +41%, because a clean sine has large
first-difference volatility |Δx| even though it is predictable) by
gating the liquid τ on the **second difference** (acceleration)
|Δ²x| = |x_t - 2x_{t-1} + x_{t-2}|, which is small for any smooth
trajectory and large only for genuinely erratic input.

## 用户故事
- As an STE-line maintainer, I can enable one gate that wins on BOTH
  smooth-periodic and structured data without reopening the noise
  failure, so gated liquid τ becomes a strict Pareto default.
- As a researcher, I can test whether r278's toy_sin loss is an
  artifact of gating on velocity vs acceleration, so I know the
  tradeoff is fixable rather than fundamental.
- As a downstream user, I get a gate that trusts smooth motion (low
  acceleration) and distrusts only erratic jitter.

## Mechanism (parameter-free acceleration gate)
```
accel_t = mean_c |x_t - 2·x_{t-1} + x_{t-2}|          # 2nd difference
vol_t   = EMA_γ(accel_t)                              # causal smoothed
g_t     = exp( -β · vol_t )   ∈ (0, 1]                # smooth→1, erratic→0
τ_i(t)  = tau_min + (tau_max - tau_min) *
          sigmoid( tau_bias_i + g_t · s · (W_τ·[x_t, h])_i )
```
The 2nd difference is the **constant-velocity forecast error**: ~0 for
any locally-linear (predictable) trajectory, large only for erratic
input. Lets the predictable sine through (g≈0.80 ⇒ near-full liquid ⇒
recovers toy_sin) while still collapsing on noise (g≈0.018 ⇒ keeps
r278's random fix).

- Still **parameter-free** ⇒ cannot chase noise.
- `diff_order=1` ⇒ exactly r278 (velocity gate, superset).

## 引擎层职责 (canonical)
- `lnn/core/accel_gated_liquid_tau_cfc.py` (NEW) —
  `AccelGatedLiquidTauCfCCell(PredictabilityGatedLiquidTauCfCCell)`
  overriding forward to accumulate |Δ²x| instead of |Δ¹x|.
  `diff_order ∈ {1 (r278), 2 (accel)}`. First two steps have no 2nd
  difference ⇒ accel=0 ⇒ g=1 (consistent with r278 t=0 convention).

## 游戏层职责
- `tests/test_accel_gated_liquid_tau_cfc.py` (NEW, ≥12 tests):
  superset (diff_order=1 ≡ r278 forward to 1e-6 on same object),
  sine→gate high (>0.6), noise→gate low (<0.15), structured→gate high,
  gradient flow, shape checks, β=0 ⇒ gate≡1, invalid diff_order raises.
- `scripts/bench_accel_gated_liquid_tau.py` (NEW) — modes
  static / liquid / gated_vel(r278) / gated_accel.
- `analysis/accel_gated_bench.json`, report.

## 验收标准
- H1 (headline): gated_accel recovers toy_sin toward liquid's -59%
  (gated_accel toy_sin < gated_vel's +41%). [testable via bench]
- H2: gated_accel keeps the random fix (random Δ% ≤ +5% vs static).
  [testable — must not reopen r278's headline win]
- H3: gated_accel preserves structured win (≤ static, near liquid's
  -12%). [testable]
- H4: gate_mean(sine) >> gate_mean(noise) after training (mechanism —
  smoke shows 0.80 vs 0.048). [diagnostics]
- H5: diff_order=1 reproduces r278 forward exactly. [unit test, 1e-6]

## 实现难度
**M** (2-6h). New cell subclass (~130 LOC, done) + ~14 tests + bench
(reuse r278 harness). 4 modes × 3 datasets × 3 seeds = 36 cells
≈ 33 min.

## 风险
- Acceleration is noisier to estimate than velocity (2nd difference
  amplifies sampling jitter). Mitigation: EMA smoothing (γ=0.5) already
  in place; smoke shows clean separation.
- Structured regime changes are large 2nd-difference spikes → gate
  down momentarily at jumps. This is arguably *correct* (a jump IS
  locally unpredictable), but could hurt if it throttles the liquid τ
  exactly when regime adaptation is needed. Bench H3 is the check.
- If H1 fails (accel gate doesn't recover toy_sin in-training despite
  the favorable gate values), we learn the gate value isn't the
  binding constraint — still a useful negative.
