---
title: "PRD #10-131 — Barlow-Twins Decorrelation Loss (r290)"
round: 290
date: 2026-07-12
author: "Claude (r290 /loop session)"
status: "selected"
parent: "r289 decorrelation (TD, H3 formulation bug)"
paper: "Barlow-Twins (Zbontar et al. 2021) — applied to LNN hidden state"
variant: "A"
---

> **Selected** (round 290, 2026-07-12): r289's decorrelation loss
> achieved the best structured Δ% (-32.1%) in any r284-r289 variant,
> but H3 failed because the diag-normalization let the optimizer
> inflate the diagonal without decorrelating. This round reformulates
> the loss in **Barlow-Twins style**: split the hidden state into two
> halves along the feature dimension (Z_A, Z_B), compute the
> cross-correlation matrix `C = Z_A · Z_B^T / T`, and penalize
>   - off-diagonal of C (decorrelate across features)
>   - deviation of diag(C) from 1 (invariance: each feature predicts itself)
> This form is **scale-invariant by construction** (C is normalized by
> the per-feature std) so the optimizer cannot escape by inflating the
> diagonal. Hypothesis: H3 ✓ and possibly H1 ✓ (target-independent).

# PRD #10-131 — Barlow-Twins Decorrelation Loss

## 目标
Test whether the **Barlow-Twins-style** decorrelation loss achieves
H3 (actually decorrelates the hidden state) and H1 (improves task loss
across all 3 datasets), fixing the r289 loss-formulation bug.

## 用户故事
- As a researcher, I confirm the H3 fail in r289 was due to
  formulation (diag-normalization escape hatch), not the decorrelation
  idea itself.
- As a gate-line maintainer, I get the first non-pulse SP candidate
  in this 6-round TD streak.

## 引擎层职责 (canonical)
- `lnn/core/decorrelation_loss.py` (extension): add
  `barlow_twins_decorrelation_loss(h, lambda_off=0.005, lambda_on=0.005)`
  that splits `h` along the feature dim, computes cross-correlation,
  and returns `λ_off · off_diag(C)^2.sum() + λ_on · (diag(C) - 1)^2.sum()`.
- `state_covariance_diagnostics(h)` already exists.

## 游戏层职责
- `scripts/bench_barlow_twins_decorrelation.py` (NEW, ~290 LOC):
  modes = {static_tau, blend_gated, blend + BT λ_off=0.001,
  blend + BT λ_off=0.005, blend + BT λ_off=0.05, blend + BT λ_off=0.005
  λ_on=0.05}; 3 datasets × 2 seeds × 50 epochs.
- `analysis/barlow_twins_decorrelation_bench.json` (NEW, 30 cells).
- `docs/research/2026-07-12_round290_barlow_twins_decorrelation_report.md`.

## 验收标准 (H1-H5)
- H1 (target-independent): improves-or-maintains task loss on ALL 3
  datasets at λ_off=0.005, λ_on=0.005.
- H3 (decorrelated): mean_diag / max_off_diag ≥ 5 (the r289 fail
  test) — must pass now with proper normalization.
- H4 (strict-positive default): H1 ∧ H3 → +1 SP, breaking the 6-round
  TD streak.

## 实现难度
**S** (1-2h). ~30 LOC loss + ~5 unit tests + ~280 LOC bench.

## 风险
- If H3 still fails: the loss is wrong; try sample-level (per-timestep)
  whitening instead of feature-level.
- If H1 fails: the BT loss is target-dependent, not strict-positive;
  abandon and pivot to r99 / arXiv:2606.21295.