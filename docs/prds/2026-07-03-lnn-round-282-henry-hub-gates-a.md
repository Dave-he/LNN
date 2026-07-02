---
title: "PRD #10-122 — Real Henry Hub Gate Evaluation (natural-gas nonstationary transfer)"
round: 282
date: 2026-07-03
author: "Claude (r282 /loop session)"
status: "draft"
parent: "r281 mixed-regime (accel gate best on within-sequence shifts; recommend real-data transfer)"
variant: "A"
---

> **Selected** (round 282, 2026-07-03): the cached Henry Hub natural-gas
> series (analysis/paper_replication/simulated_henry_hub.csv, 2645 days)
> is the LITERAL motivating domain of the gate line (arXiv:2604.24788,
> "limit responsiveness when regimes shift rapidly") and is genuinely
> nonstationary (rolling-30 vol ranges 33×). It tests r281's finding
> (accel gate best on nonstationary data) on real data with zero
> download risk. Directly answers the r281 recommendation.

# PRD #10-122 — Real Henry Hub Gate Evaluation

## 目标
Test whether r281's synthetic-mixed-regime finding — acceleration gate
is the best liquid-τ gate on nonstationary data — transfers to the REAL
Henry Hub natural-gas spot-price series, the domain the gate line was
motivated by.

## 用户故事
- As an STE-line maintainer, I can see gate performance on the real
  nonstationary series the papers target, so the accel-gate production
  recommendation is grounded in reality, not just toys.
- As a researcher, I can test whether the r277-281 gate ordering
  (accel > vel/blend > ungated) holds on real data with genuine
  volatility clustering.
- As a downstream user, I get evidence that gated liquid τ works on
  real forecasting, not just synthetic benchmarks.

## Data (cached, no download)
`analysis/paper_replication/simulated_henry_hub.csv`:
- 2645 daily obs (2015-2025), Spot Price min 1.20 / max 34.91.
- Nonstationary: rolling-30 return vol 0.009→0.284 (33× range) — real
  regime shifts (calm periods vs volatility spikes).
- Task: one-step-ahead spot-return prediction from a sliding window
  (T=64), normalised. Chronological train/test split (no shuffle —
  respect the time ordering).

## 引擎层职责 (canonical)
- No new engine code. Reuse the five gate cells: STEWithEntropy (r267),
  LiquidTauSTECfCCell (r277), PredictabilityGatedLiquidTauCfCCell (r278),
  AccelGatedLiquidTauCfCCell (r279), BlendGatedLiquidTauCfCCell (r280).

## 游戏层职责
- `scripts/bench_henry_hub_gates.py` (NEW) — load CSV, build sliding
  windows, chronological split, normalise, run 5 gate modes. Report
  overall test MSE + a "high-vol subset" MSE (test windows whose target
  period falls in the top-quartile rolling volatility — the regime-shift
  stress subset where gates should matter most).
- `analysis/henry_hub_gates_bench.json`, report.
- 1-2 sanity unit tests for the data loader (shape, chronological split,
  no NaN, no look-ahead).

## 验收标准
- H1: on real Henry Hub, at least one gate beats static overall (the
  gate mechanism transfers to real data). [bench]
- H2 (headline): acceleration gate is best or tied-best overall,
  matching r281's synthetic finding. [bench]
- H3: ungated liquid is worst / unsafe on the high-vol subset (real
  version of the r277/r281 noise blowup). [high-vol subset]
- H4: the gates' advantage concentrates in the HIGH-VOL subset (where
  regime shifts happen), not the calm subset. [subset comparison]
- H5: results are not degenerate — test MSE is finite, > 1e-4, and
  train converges for all modes. [scale/sanity check]

## 实现难度
**M** (2-6h). Data loader + windowing + chronological split (~60 LOC) +
bench (reuse r281 harness) + 2 loader tests + report. 5 modes × 3 seeds
= 15 cells; real series is longer so ~60-90s/cell ≈ 20-25 min.

## 风险
- Normalisation choice matters (raw prices span 1.2-34.9; must use
  returns or log-prices + standardisation). Mitigation: predict
  standardised returns, report in that space; sanity-check scale.
- Chronological split means the test set (recent years) may have a
  different vol regime than train — this is realistic but could make
  ALL models look bad. Mitigation: report high-vol vs calm subsets so
  the gate story is visible even if absolute MSE is high.
- Look-ahead leakage risk in windowing/normalisation. Load-bearing:
  compute normalisation stats on TRAIN only. Unit-test this.
- If gates don't separate on real data (real noise is less extreme than
  synthetic i.i.d.), that is an honest, valuable NEGATIVE about transfer.
