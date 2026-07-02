---
title: "PRD #10-116 — STE Predictability-Gate β Sweep (gate-sharpness map)"
round: 279
date: 2026-07-03
author: "Claude (r279 /loop session)"
status: "draft"
parent: "r278 predictability-gated liquid τ (+1 SP, H1 partial: toy_sin +41%)"
variant: "A"
---

> **Rejected** (round 279): sound and cheap (S), but its most likely
> outcome (H5: no Pareto β) merely motivates PRD B. Deferred — a β
> sweep of the relative-volatility gate is a better future round.

# PRD #10-116 — STE Predictability-Gate β Sweep

## 目标
Map the predictability-gate sharpness knob β to find whether a value
other than 4.0 keeps r278's random fix while recovering r277's toy_sin
win — a pure-bench, zero-new-code sweep.

## 用户故事
- As an STE-line maintainer, I can see the full β→(toy_sin, structured,
  random) surface, so I can pick the production β with evidence.
- As a researcher, I can confirm whether β=4.0 is optimal or arbitrary,
  so the r278 default is justified rather than inherited.
- As a downstream user, I can trade gate sharpness for task fit knowing
  the exact noise-safety boundary.

## 引擎层职责 (canonical)
- No new engine code. Reuse `lnn/core/pred_gated_liquid_tau_cfc.py:67`
  `PredictabilityGatedLiquidTauCfCCell` (pred_gate_beta param).
- `scripts/bench_pred_gated_liquid_tau.py:103` MODES — add β variants:
  `gated_b0.5, gated_b1, gated_b2, gated_b4 (r278), gated_b8, gated_b16`
  plus `static_tau` + `liquid_tau` (β=0 anchor) as references.

## 游戏层职责
- N/A (research repo, no game layer). Mirror = analysis JSON + report.
- `analysis/ste_pred_gate_beta_bench.json`
- `docs/research/2026-07-03_round279_pred_gate_beta_report.md`

## 验收标准
- H1: β=4.0 confirmed as the random-safe boundary (random Δ% ≤ +5%
  for β ≥ 4). [testable: random means by β]
- H2: lower β (0.5-2) recovers toy_sin toward liquid's -59% but
  reopens the random regression (monotonic gate-leak). [testable]
- H3: gate_mean on random increases monotonically as β decreases
  (mechanism check: lower β = less collapse on noise). [testable]
- H4: structured is β-insensitive (gate stays ≥0.8 for all β because
  structured volatility is low). [testable]
- H5: there exists NO β that is Pareto-optimal (recovers toy_sin AND
  keeps random safe) — i.e. the tradeoff is fundamental to the
  absolute-volatility gate. [the headline negative-or-positive]

## 实现难度
**S** (≤2h). Zero new code. ~6 β × 3 datasets × 3 seeds = 54 cells
(+ 2 ref modes × 9 = 18) ≈ 72 cells @ ~55s = ~65 min bench. Trim to
2 seeds if time-bound (48 cells ≈ 44 min).

## 风险
- 72 cells may exceed the 1h loop window. Mitigation: 2 seeds, or drop
  β=16 (redundant with β=8 if both fully collapse).
- If H5 confirms (no Pareto β), the round is a clean NEGATIVE that
  motivates PRD B (relative-volatility gate) as the real fix. That is
  itself a valuable, publishable result.
- Load-bearing assumption: gate_mean is a faithful proxy for
  "how much liquid leaks through" — validated by r278 (gate 0.06 on
  random ⇒ near-static behavior).
