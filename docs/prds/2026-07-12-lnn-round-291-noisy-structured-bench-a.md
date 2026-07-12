---
title: "PRD #10-132 — Noisy-Structured 4th Dataset + Re-Test Decorrelation (r291)"
round: 291
date: 2026-07-12
author: "Claude (r291 /loop session)"
status: "selected"
parent: "r289 decorrelation (TD, H3 formulation bug); r290 BT (FAILURE)"
paper: "bench-improvement, no external paper"
variant: "A"
---

> **Selected** (round 291, 2026-07-12): after 7 rounds of TD/NEGATIVE
> on the (toy_sin / structured / random) triplet, the bench itself is
> suspected as too narrow. This round adds a 4th "hard" dataset —
> **noisy-structured**: piecewise-constant signal + additive Gaussian
> noise — that has both structure AND noise. Then re-tests the r289
> decorrelation loss (with a *scaled-down* version that doesn't
> catastrophically dominate) on the 4-dataset bench. Hypothesis:
> noisy-structured is the missing benchmark that distinguishes
> strict-positive mechanisms from target-dependent ones.

# PRD #10-132 — Noisy-Structured 4th Dataset

## 目标
1. Add `noisy_structured` to the bench — piecewise-constant signal +
   noise. This dataset has structure (which toy_sin has, and
   decorrelation should help) AND noise (which structured lacks and
   decorrelation should not break). If a mechanism is strict-positive
   across all 4, it's a real SP.
2. Re-run the r289 decorrelation loss on the 4-dataset bench. Use a
   *conservative λ* (e.g. 1e-5 to 1e-3) so the loss doesn't dominate
   the task loss on toy_sin. Hypothesis: with the right λ, decorrelation
   can be target-independent across all 4.

## 用户故事
- As a researcher, I get a 4-dataset bench that discriminates
  strict-positive mechanisms from target-dependent ones.
- As a gate-line maintainer, I confirm whether r289's decorrelation
  loss was *correctly formulated* but had a λ-scale bug, or was
  fundamentally target-dependent.

## 引擎层职责 (canonical)
- `scripts/bench_decorrelation_loss.py` (EXTENSION): add
  `noisy_structured` factory (piecewise-constant + Gaussian noise at
  SNR=2). Re-run with λ ∈ {1e-5, 1e-4, 1e-3} (much smaller than r289's
  {1e-4, 1e-3, 1e-2, 1e-1}).
- No new cell code; no new loss code.

## 游戏层职责
- Re-run existing bench with new dataset + new λ sweep.
- `analysis/decorrelation_loss_bench_v2.json` (overwrite or new file).
- `docs/research/2026-07-12_round291_noisy_structured_bench_report.md`.

## 验收标准 (H1-H3)
- H1 (target-independent on ALL 4 datasets): decorrelation loss
  improves-or-maintains task loss on ALL 4 at λ ∈ {1e-5, 1e-4, 1e-3}.
- H3 (decorrelated axes): mean_diag / max_off_diag ≥ 5 — same as r289.
- H4 (strict-positive default): H1 ∧ H3 → +1 SP, breaking the 7-round
  TD streak.

## 实现难度
**S** (1h). ~50 LOC change in bench script.

## 风险
- If H1 fails: the bench is still too narrow; need a 5th dataset or
  a totally different mechanism class.
- If H1 passes on 3/4: target-dependent; pivot to a different
  mechanism.