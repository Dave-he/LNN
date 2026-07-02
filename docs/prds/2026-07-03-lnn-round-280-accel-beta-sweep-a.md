---
title: "PRD #10-119 — Acceleration-Gate β Sweep (gate-sharpness map)"
round: 280
date: 2026-07-03
author: "Claude (r280 /loop session)"
status: "draft"
parent: "r279 accel-gated liquid τ (+1 SP)"
variant: "A"
---

> **Rejected** (round 280): sound and cheap (S), but it only tunes one
> knob of the r279 gate; the blend gate (PRD B) attacks the higher-value
> question (can we get r278's structured edge AND r279's toy_sin win at
> once). A β sweep of the *winning* gate is a better future round.

# PRD #10-119 — Acceleration-Gate β Sweep

## 目标
Map the acceleration-gate sharpness β to find whether a value other
than 4.0 improves r279's structured neutrality (+0.4%) or toy_sin win
(-77.5%) without reopening the noise regression — pure-bench, zero code.

## 用户故事
- As an STE-line maintainer, I can see the full β→(toy_sin, structured,
  random) surface for the accel gate, so the production β is evidenced.
- As a researcher, I can confirm β=4.0 is optimal or find a better one.
- As a downstream user, I can trade gate sharpness knowing the exact
  noise-safety boundary.

## 引擎层职责 (canonical)
- No new engine code. Reuse `lnn/core/accel_gated_liquid_tau_cfc.py`
  `AccelGatedLiquidTauCfCCell` (pred_gate_beta param, diff_order=2).
- `scripts/bench_accel_gate_beta.py` (NEW) — MODES:
  `accel_b1, accel_b2, accel_b4 (r279), accel_b8, accel_b16` +
  static/liquid references.

## 游戏层职责
- N/A. `analysis/accel_gate_beta_bench.json` + report.

## 验收标准
- H1: β=4.0 confirmed as random-safe (random Δ% ≤ +5% for β ≥ 4).
- H2: lower β recovers structured toward liquid's -12% (more liquid
  leaks through) but risks reopening random.
- H3: gate_mean(structured) rises as β falls (mechanism check).
- H4: there is (or isn't) a β that is Pareto over β=4.0.

## 实现难度
**S** (≤2h). Zero new engine code. 5 β × 3 datasets × 3 seeds = 45
cells ≈ 40 min.

## 风险
- 45 cells fits the 1h window. Low risk.
- Likely outcome: β=4 near-optimal (monotone tradeoff), making this a
  confirm rather than a new win — lower value than PRD B.
