---
title: "PRD #10-138 — TopologicalLiquidCfCCell (r297)"
round: 297
date: 2026-07-12
author: "Claude (r297 /loop session)"
status: "selected"
parent: "r295 default promotion complete; pivot to fresh mechanism"
paper: "arXiv:2606.21295 (Cai & Zhao 2026-06) — Topological Neural Dynamics"
variant: "A"
---

> **Selected** (round 297, 2026-07-12): after 12 rounds on the
> pulse + decorrelation lines, this round pivots to a fresh
> architectural mechanism: neuron-wise topological dynamics from
> arXiv:2606.21295. Hypothesis: per-neuron sparse graph-structured
> recurrent connections (vs the layer-wise dense `W @ h` of all
> existing cells) may improve Henry Hub by allowing individual
> neurons to specialise.

# PRD #10-138 — TopologicalLiquidCfCCell

## 目标
Test whether a graph-structured cell (per-neuron sparse recurrent
connections instead of layer-wise dense `W @ h`) improves Henry Hub
overall test MSE vs the r295 default blend_gated cell.

## 引擎层职责 (canonical)
- `lnn/core/topological_liquid_cfc.py` (NEW, ~250 LOC):
  `TopologicalLiquidCfCCell`. Each neuron has `n_incoming` random
  source connections sampled at init. Combines with r280 blend gate
  and r295 decorrelation default.
- `tests/test_topological_liquid_cfc.py` (NEW, 15 tests, all green).
- `scripts/bench_topological_liquid.py` (NEW, ~250 LOC).

## 验收标准
- H1: TopologicalLiquidCfCCell (n_incoming=8) improves Henry Hub
  overall MSE vs blend_gated default.
- H2: Different n_incoming values (4, 8, 16) all reasonable.
- H3: gradient flow works (unit-tested).

## 实现难度
**M** (2h). Already implemented and tested.