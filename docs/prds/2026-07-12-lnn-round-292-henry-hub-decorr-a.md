---
title: "PRD #10-133 — Henry Hub Real-World Decorrelation Validation (r292)"
round: 292
date: 2026-07-12
author: "Claude (r292 /loop session)"
status: "selected"
parent: "r291 decorrelation λ=1e-5 (FIRST SP in 8 rounds)"
paper: "arXiv:2607.01986 + arXiv:2604.24788 Henry Hub"
variant: "A"
---

> **Selected** (round 292, 2026-07-12): r291 found decorrelation loss
> at λ=1e-5 is strict-positive on the 4-dataset toy bench. This round
> validates the SP result on **real-world data**: Henry Hub natural-gas
> spot prices (arXiv:2604.24788, the original liquid-τ motivation).
> If decorrelation at λ=1e-5 also helps on Henry Hub, the r291 SP
> result is **not a toy-bench artifact** and decorrelation can be
> recommended as a default regularizer for the gate line.

# PRD #10-133 — Henry Hub Real-World Decorrelation Validation

## 目标
Test whether the r291 finding — decorrelation loss at λ=1e-5 is
strict-positive on the toy bench — extends to **real Henry Hub
natural-gas spot prices**. Specifically:
1. Does decorrelation λ=1e-5 improve blend_gated on Henry Hub
   (overall test MSE)?
2. Does it improve on the **high-vol subset** (regime-shift stress)?
3. Does the 4-dataset SP result survive real-world validation?

## 用户故事
- As a gate-line maintainer, I confirm that decorrelation λ=1e-5 is
  a safe default to add to the blend gate line in production.
- As a researcher, I validate that the 4-dataset toy bench SP is a
  real mechanism, not a benchmark artifact.

## 引擎层职责 (canonical)
- `scripts/bench_henry_hub_decorrelation.py` (NEW, ~300 LOC):
  reuse the r282 data loader, add decorrelation loss on top of
  blend_gated, sweep λ ∈ {1e-5, 1e-4} (small only — r291 found
  larger λ hurts).
- `analysis/henry_hub_decorrelation_bench.json` (NEW).
- `docs/research/2026-07-12_round292_henry_hub_decorr_report.md`.

## 验收标准 (H1-H3)
- H1 (overall): blend_gated + decorr λ=1e-5 improves Henry Hub test
  MSE vs blend_gated alone.
- H2 (high-vol): blend_gated + decorr λ=1e-5 improves the high-vol
  subset MSE (regime-shift stress) — the most informative regime
  for the gate line.
- H3 (no-collapse): diag/off_ratio stays reasonable (≥ 1.0) after
  training — decorrelation doesn't blow up the state on real data.

## 实现难度
**S** (1-2h). Mostly reusing r282 loader.

## 风险
- If H1 ✗: toy SP doesn't transfer to real data; decorrelation may
  be toy-only artifact. Pivot to a different mechanism.
- If H2 ✓ but H1 ✗: decorrelation helps in stress regimes but hurts
  average; → +1 TD only.
- If both ✗: pivot away from decorrelation entirely.