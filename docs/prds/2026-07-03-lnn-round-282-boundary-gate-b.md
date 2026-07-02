---
title: "PRD #10-123 — Boundary-Aware (Transition-Reset) Gate"
round: 282
date: 2026-07-03
author: "Claude (r282 /loop session)"
status: "draft"
parent: "r281 mixed-regime (blend fails in transition zones)"
variant: "B"
---

> **Rejected** (round 282): a promising mechanism idea (r281 candidate
> #2), but it adds NEW engine complexity (state-reset logic) and only
> improves on the SYNTHETIC mixed-regime task, which r281 already
> validated. Testing transfer to REAL data (PRD A) is the higher-value,
> lower-risk next step; a boundary-aware gate is better pursued after we
> know the accel gate holds on real data.

# PRD #10-123 — Boundary-Aware (Transition-Reset) Gate

## 目标
Beat the r279 acceleration gate on within-sequence regime shifts by
explicitly detecting regime BOUNDARIES (a |Δ²x| spike that then
subsides) and briefly damping the carried hidden state so a new regime
starts cleaner.

## Mechanism
```
accel_t = EMA_γ(|Δ²x|)
boundary_t = relu(accel_t - accel_{t-1})   # rising acceleration = onset
reset_t = exp(-κ · boundary_t) ∈ (0,1]      # dips at a detected boundary
h_t = reset_t ⊙ h_carried + ... (dampen state through the transition)
τ gated as in r279 (acceleration gate)
```

## 引擎层职责 (canonical)
- `lnn/core/boundary_gated_liquid_tau_cfc.py` (NEW) — subclass of r279
  AccelGated with a state-reset term on detected boundaries.

## 游戏层职责
- `tests/test_boundary_gated_liquid_tau_cfc.py` (NEW, ≥12 tests).
- `scripts/bench_boundary_gate_mixed.py` (reuse r281 mixed-regime data).

## 验收标准
- H1: boundary gate beats r279 accel on mixed-regime overall.
- H2: the win concentrates at the structured segment (post-transition).
- H3: κ=0 reproduces r279 (superset).
- H4: no instability from state resets.

## 实现难度
**M-L** (4-8h). New engine cell + state-reset logic (subtle — resets can
destabilise BPTT) + tests + bench.

## 风险
- State resets can break gradient flow / destabilise training.
- Only validated on synthetic; real-data transfer still unknown.
- Higher complexity than PRD A for a synthetic-only gain.
