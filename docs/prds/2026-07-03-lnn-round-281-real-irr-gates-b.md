---
title: "PRD #10-121 — Real Irregular-TS Gate Evaluation (PhysioNet-style via QuITE)"
round: 281
date: 2026-07-03
author: "Claude (r281 /loop session)"
status: "draft"
parent: "r280 blend gate (toy benchmark SATURATED)"
variant: "B"
---

> **Rejected** (round 281): the right long-term direction, but real-data
> loading (PhysioNet download / dependency / preprocessing) is high-risk
> in a 1h loop and the gate mechanism can be discriminated more cleanly
> and controllably by a synthetic mixed-regime task first (PRD A). Defer
> real data to a round with a pre-cached dataset.

# PRD #10-121 — Real Irregular-TS Gate Evaluation

## 目标
Evaluate the four liquid-τ gate variants on a real irregular-sampled
time series (PhysioNet-style) using the r102 QuITE embedding, to test
whether the gate findings transfer from synthetic toys to real data.

## 用户故事
- As an STE-line maintainer, I can see gate performance on real clinical
  irregular-TS, so production choice is grounded in reality.
- As a researcher, I can test whether the r277-280 gate ordering holds
  on real nonstationary data.

## 引擎层职责 (canonical)
- Reuse the four gate cells + r102 QueryIrregularEmbedding
  (`lnn/core/quite_embedding.py`) as the front-end for irregular input.

## 游戏层职责
- `scripts/bench_real_irr_gates.py` (NEW) — load a cached real dataset,
  QuITE-embed, run the 5 gate modes.
- `analysis/real_irr_gates_bench.json`, report.

## 验收标准
- H1: gate ordering on real data matches or diverges from toy findings.
- H2: QuITE + gated liquid τ is stable on real irregular sampling.
- H3: blend or accel is the best real-data gate.

## 实现难度
**L** (>6h). Requires: locating/caching a real dataset, preprocessing to
QuITE format, wiring the embedding to each gate cell, debugging real-data
NaN/scale issues. High risk of blowing the 1h window on data plumbing.

## 风险
- Real dataset may not be locally cached → download in a proxied env is
  unreliable. Load-bearing blocker.
- QuITE↔gate-cell wiring is untested (r102 used QuITE with plain CfC,
  not the gated liquid cells) — integration risk.
- Preprocessing choices (normalisation, missing-rate) could dominate the
  gate signal, confounding the comparison.
