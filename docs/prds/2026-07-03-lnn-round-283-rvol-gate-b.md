---
title: "PRD #10-125 — Realised-Volatility Gate (why does velocity win on real data?)"
round: 283
date: 2026-07-03
author: "Claude (r283 /loop session)"
status: "draft"
parent: "r282 Henry Hub (velocity gate wins — is it a volatility-clustering proxy?)"
variant: "B"
---

> **Rejected** (round 283): a good mechanistic follow-up, but it presumes
> the r282 velocity-win generalises — which is untested. The multi-series
> transfer (PRD A) must come first to establish WHETHER velocity is
> generally best before we build a new gate to explain WHY. Deferred.

# PRD #10-125 — Realised-Volatility Gate

## 目标
Test the hypothesis that r282's velocity gate wins on real return series
because |Δreturn| is a proxy for realised volatility (volatility
clustering), by building a gate that keys on rolling realised volatility
directly and comparing it to the velocity gate.

## Mechanism
```
rv_t = EMA_γ(return_t^2)          # rolling realised variance
g_t  = exp(-β · sqrt(rv_t))       # collapse liquid τ in high-vol regimes
```

## 引擎层职责 (canonical)
- `lnn/core/rvol_gated_liquid_tau_cfc.py` (NEW) — subclass of r278 cell,
  gate on rolling realised vol instead of |Δx|.

## 游戏层职责
- `tests/test_rvol_gated_liquid_tau_cfc.py` (NEW, ≥12 tests).
- `scripts/bench_rvol_gate_henry_hub.py` (reuse r282 loader).

## 验收标准
- H1: rvol gate matches or beats velocity gate on real Henry Hub.
- H2: if rvol ≈ velocity, confirms velocity is a vol-clustering proxy.
- H3: β=0 reproduces r277 (superset).

## 实现难度
**M** (3-5h). New cell + tests + bench.

## 风险
- Presumes velocity generalises (untested — PRD A first).
- realised-vol and |Δreturn| may be near-identical for daily returns,
  making the comparison uninformative.
