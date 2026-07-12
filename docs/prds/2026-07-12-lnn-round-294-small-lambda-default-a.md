---
title: "PRD #10-135 — Small-λ In-Cell Decorrelation Default (r294)"
round: 294
date: 2026-07-12
author: "Claude (r294 /loop session)"
status: "selected"
parent: "r293 in-cell default at λ=1e-4 (FAIL +5%)"
paper: "decorrelation sweep"
variant: "A"
---

> **Selected** (round 294, 2026-07-12): r293 found the in-cell
> default at λ=1e-4 regresses Henry Hub by +5%. But r291's toy SP was
> at λ=1e-5 (smaller). This round tests whether smaller λ in the
> in-cell default avoids the regression. If λ=1e-5 reproduces the
> toy SP on Henry Hub (negative Δ% or near-zero), then the in-cell
> default IS viable — just at a smaller scale.

# PRD #10-135 — Small-λ In-Cell Decorrelation

## 目标
Sweep λ ∈ {1e-5, 1e-6, 1e-7} for the in-cell default on Henry Hub.
If any λ gives Δ% within ±2% of baseline, mark it as a viable
in-cell default. Otherwise confirm r293's revert was correct.

## 用户故事
- As a researcher, I find the largest λ that doesn't regress Henry Hub.
- As a gate-line maintainer, I get a clearer answer on whether
  in-cell default is viable at all.

## 引擎层职责 (canonical)
- `scripts/bench_henry_hub_default_decorr.py` (EDIT): add λ ∈ {1e-5, 1e-6, 1e-7}
  to the mode table.

## 验收标准 (H1-H2)
- H1 (no regression): blend_new with in-cell default at λ* ∈ {1e-5,
  1e-6, 1e-7} gives overall Δ% ≤ +2% vs blend_old.
- H2 (improvement): blend_new with in-cell default at λ* gives overall
  Δ% ≤ -0.3% (matching r292's opt-in result).

## 实现难度
**S** (1h). Just re-run with new λ values.

## 风险
- If H1 fails for all λ: in-cell default is structurally wrong;
  confirm r293 revert and pivot to fresh mechanism.