---
title: "PRD #10-124 — Multi-Series Real Gate Transfer (WTI, rates, equity, coal, gas)"
round: 283
date: 2026-07-03
author: "Claude (r283 /loop session)"
status: "draft"
parent: "r282 Henry Hub (velocity gate wins on real gas; is it gas-specific?)"
variant: "A"
---

> **Selected** (round 283, 2026-07-03): r282 found VELOCITY gate wins on
> real Henry Hub gas (not accel, contra synthetic r281). The single most
> valuable follow-up is whether that holds ACROSS real return series or
> is gas-specific. The Henry Hub CSV already caches 5 real financial
> series (gas, WTI oil, Treasury 10Y, S&P Energy, coal) with distinct
> volatility profiles — zero download risk, directly answers the r282
> open question.

# PRD #10-124 — Multi-Series Real Gate Transfer

## 目标
Determine whether r282's "velocity gate is the best liquid-τ gate on
real return series" generalises across five real financial series (gas,
WTI oil, Treasury 10Y, S&P Energy, coal) or is specific to natural gas.

## 用户故事
- As an STE-line maintainer, I can see whether the velocity-gate
  production recommendation holds across real domains, so I know if it's
  a general rule or a gas artifact.
- As a researcher, I can test whether gate ordering correlates with a
  series' volatility profile (nonstationarity), so I understand WHEN
  gating helps.
- As a downstream user, I get evidence for gate choice on my own real
  return series based on its volatility structure.

## Data (cached, no download)
`analysis/paper_replication/simulated_henry_hub.csv` columns as 5
independent real series (2645 daily obs each, 2015-2025):
| series        | ret_std | vol_ratio (nonstationarity) |
|---------------|--------:|----------------------------:|
| Spot Price    | 0.051   | 32.9× (gas — r282)          |
| WTI Price     | 0.033   | 3.2× (oil)                  |
| Treasury_10Y  | 0.043   | 3.6× (rates)                |
| SP_Energy     | 0.021   | 3.0× (equity)               |
| Coal_Index    | 0.058   | 4.1× (coal)                 |
Task per series: standardised one-step return prediction, T=64,
chronological split, train-only normalisation (no look-ahead).

## 引擎层职责 (canonical)
- No new engine code. Reuse the five gate cells (r267/277/278/279/280).
- Generalise the r282 loader to accept any column (reuse
  `scripts/bench_henry_hub_gates.py` loader pattern).

## 游戏层职责
- `scripts/bench_multiseries_gates.py` (NEW) — loop over 5 series × 5
  gate modes, report per-series overall + high-vol MSE and per-series
  gate ranking.
- `analysis/multiseries_gates_bench.json`, report with a series ×
  gate-mode matrix.

## 验收标准
- H1: gates transfer on ≥3 of 5 series (best gate beats static). [matrix]
- H2 (headline): velocity is the best or tied-best gate on a MAJORITY
  of series (r282 generalises), OR the best gate varies by series
  (r282 is gas-specific — also a clear result). [ranking]
- H3: the gate advantage correlates with the series' nonstationarity
  (vol_ratio) — high-vol series benefit more from gating. [correlation]
- H4: ungated liquid is safe (≤ static) on real series across the board
  (r282's H3 reversal generalises — synthetic worst-case doesn't occur
  in real markets). [matrix]
- H5: results not degenerate — finite MSE, converged, no look-ahead,
  per-series standardisation on train only. [sanity]

## 实现难度
**M** (2-6h). Generalise r282 loader to arbitrary column (~30 LOC) +
5-series loop bench + report. To fit the loop window: 5 series × 5 modes
× 2 seeds = 50 cells, but real series are ~170-280s/cell → too slow.
Mitigation: reduce to 40 epochs + 2 seeds, OR subset to 3 series
(gas/oil/equity) × 5 modes × 2 seeds = 30 cells. Start with 40 epochs,
2 seeds, 3 series (~30 cells) and expand if time permits.

## 风险
- 5 series × full grid may exceed 1h. Mitigation: 40 epochs, 2 seeds, 3
  representative series first (high/mid/low vol_ratio); note any dropped
  cells explicitly.
- Some columns may have flat/degenerate stretches (EQT had a vol_ratio
  blowup) → pick clean series (gas, WTI, Treasury, SP_Energy, coal).
- If the best gate varies unpredictably across series, that is still a
  valuable result: it means gate choice needs a data-driven selector
  (motivates a future auto-gate round).
