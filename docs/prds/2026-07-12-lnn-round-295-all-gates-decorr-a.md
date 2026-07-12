---
title: "PRD #10-136 — Decorrelation Default Extends to All 3 Gate Variants (r295)"
round: 295
date: 2026-07-12
author: "Claude (r295 /loop session)"
status: "selected"
parent: "r294 in-cell default at λ=1e-5 (SP on blend_gated -1.3%/-2.6%)"
paper: "decorrelation default promotion"
variant: "A"
---

> **Selected** (round 295, 2026-07-12): r294 promoted decorrelation
> at λ=1e-5 to default in blend_gated with SP result. This round
> extends the same default to the other 2 gate variants (pred_gated
> r278 and accel_gated r279) and validates on Henry Hub. If both
> variants also benefit, r295 adds 2 SP.

# PRD #10-136 — Decorrelation Default Extends to All 3 Gates

## 目标
Test whether the r294 in-cell decorrelation default at λ=1e-5 helps
pred_gated (r278) and accel_gated (r279) on Henry Hub. Hypothesis:
yes — the gate-line composition should not block the regularizer.

## 引擎层职责 (canonical)
- `lnn/core/pred_gated_liquid_tau_cfc.py` (EDIT): add `decorr_lambda`
  arg + `_last_outputs` cache + `extra_loss()` override (root change).
- `lnn/core/accel_gated_liquid_tau_cfc.py` (EDIT): add `decorr_lambda`
  arg (passes through to parent).

## 游戏层职责
- `scripts/bench_all_gates_decorr.py` (NEW, ~250 LOC): 7 modes ×
  2 seeds × 30 epochs, 14 cells.
- `analysis/all_gates_decorr_bench.json` (NEW).
- `docs/research/2026-07-12_round295_all_gates_decorr_report.md`.

## 验收标准
- H1: pred_gated_default Δ% ≤ +5% vs pred_gated_off.
- H2: accel_gated_default Δ% ≤ +5% vs accel_gated_off.
- H3: each is +1 SP → +2 SP total.

## 实现难度
**S** (1h). Mostly edits + bench reuse.