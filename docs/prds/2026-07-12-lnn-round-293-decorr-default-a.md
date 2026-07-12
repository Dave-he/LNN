---
title: "PRD #10-134 — Decorrelation as Default Regularizer in BlendGated Cell (r293)"
round: 293
date: 2026-07-12
author: "Claude (r293 /loop session)"
status: "selected"
parent: "r291 toy SP + r292 Henry Hub real-world confirmation"
paper: "arXiv:2607.01986 + arXiv:2604.24788"
variant: "A"
---

> **Selected** (round 293, 2026-07-12): r291 + r292 jointly establish
> that `state_decorrelation_loss(λ=1e-4)` is a safe default regularizer
> for the blend gate line on real-world data. This round promotes the
> finding from "available as opt-in loss" to **default behavior** of
> `BlendGatedLiquidTauCfCCell.extra_loss()`. The change is **single-line**:
> add a `λ_decorr` term to the cell's extra loss.

# PRD #10-134 — Decorrelation as Default in BlendGated

## 目标
1. Add `state_decorrelation_loss(λ=1e-4)` as a default term in
   `BlendGatedLiquidTauCfCCell.extra_loss()` so existing users get the
   SP benefit automatically.
2. Add a `decorr_lambda` constructor argument (default 1e-4) to allow
   opt-out (`decorr_lambda=0.0`).
3. Re-run the r292 Henry Hub validation to confirm the default
   `extra_loss()` matches the previous opt-in results.

## 用户故事
- As a downstream user, I get the SP benefit of decorrelation without
  having to know about the loss function.
- As a gate-line maintainer, I have a single hyperparameter
  (`decorr_lambda`) to control the new behavior.

## 引擎层职责 (canonical)
- `lnn/core/blend_gated_liquid_tau_cfc.py` (EDIT):
  - Add `decorr_lambda: float = 1e-4` to `__init__`.
  - Add `state_decorrelation_loss(self.h)` term to `extra_loss()`.
- `tests/test_blend_gated_liquid_tau_cfc.py` (EDIT): add tests for
  the new arg.
- `scripts/bench_henry_hub_decorrelation.py` (EDIT): re-run with the
  default extra_loss to confirm SP holds without explicit loss term.

## 验收标准 (H1-H3)
- H1: Existing tests still pass (14 tests in `test_blend_gated_*`).
- H2: New tests verify `decorr_lambda=0.0` reverts to old behavior.
- H3: r292 Henry Hub result (-0.3% overall, -1.0% hi_vol) holds when
  decorrelation is added inside `extra_loss()` (i.e. comparing
  blend_gated WITHOUT extra_loss vs WITH extra_loss at default λ).

## 实现难度
**S** (30min). ~10 LOC cell edit + ~5 LOC tests + re-run bench.

## 风险
- If H1 fails: the loss term interferes with entropy_lambda. May need
  careful scale balancing.
- If H3 fails: the default λ=1e-4 is too aggressive for some contexts.
  Default to 1e-5 instead.