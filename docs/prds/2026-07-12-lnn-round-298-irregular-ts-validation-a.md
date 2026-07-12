---
title: "PRD #10-139 — Irregular TS Validation of r295 Decorrelation Default (r298)"
round: 298
date: 2026-07-12
author: "Claude (r298 /loop session)"
status: "selected"
parent: "r295 decorrelation default on smooth TS (SP on Henry Hub)"
paper: "irregular TS validation of decorrelation SP"
variant: "A"
---

> **Selected** (round 298, 2026-07-12): r295 confirmed decorrelation
> default λ=1e-5 is SP on smooth Henry Hub. This round tests whether
> the SP generalizes to **irregular time series** (PhysioNet-style
> data with masked gaps and missing values, ~50% missing rate). If it
> also helps on irregular TS, r295's SP generalizes across data
> regimes — strong evidence the mechanism is real.

# PRD #10-139 — Irregular TS Decorrelation Validation

## 目标
Test whether the r295 in-cell decorrelation default at λ=1e-5 helps
on irregular time series (r102 QuITE datasets). If yes, the SP
generalizes beyond smooth Henry Hub.

## 引擎层职责 (canonical)
- Reuse r102 QuITE datasets (sin_irr, structured_irr, random_irr) and
  bench structure. Replace CfC with BlendGatedLiquidTauCfCCell
  (which has the r295 decorrelation default).
- New `scripts/bench_irregular_decorrelation.py` (~250 LOC).

## 验收标准
- H1: blend_gated + decorrelation (default λ=1e-5) improves task
  loss on irregular TS vs blend_gated without decorrelation.
- H2: generalizes across all 3 irregular datasets (sin_irr,
  structured_irr, random_irr).

## 实现难度
**S** (1-2h). Mostly reuses r102 data + cell swap.

## 风险
- If H1 fails: decorrelation SP may be data-specific to smooth TS.
  Pivot to a different mechanism.