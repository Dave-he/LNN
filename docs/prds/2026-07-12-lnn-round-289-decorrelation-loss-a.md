---
title: "PRD #10-130 — Decorrelation Loss for State Disentanglement (r289)"
round: 289
date: 2026-07-12
author: "Claude (r289 /loop session)"
status: "selected"
parent: "r288 EMA-gate; r287 binary; r286 sqrt; r285 linear; r284 pulse (5-round pulse line closed)"
paper: "arXiv:2607.01986 (Nie, Wang, Su 2026-07) — Liquid Latent State Dynamics for Turbofan Degradation"
variant: "A"
---

> **Selected** (round 289, 2026-07-12): after 5 rounds (r284-r288) the
> pulse + any-gate family is fully characterized and exhausted (no
> strict-positive). This round pivots to a fresh mechanism: **state
> decorrelation loss** from arXiv:2607.01986. The paper trains a liquid
> NN as latent dynamics for turbofan RUL and adds a
> degradation/condition decorrelation loss that beats GRU on C-MAPSS.
> The mechanism (decorrelate the latent state across the two
> sub-processes — degradation vs operating-condition) is a *fresh
> disentanglement axis* in our 22-layer stack: orthogonal to the gate
> line, the pulse line, and the r100 SNNL line. Hypothesis: it adds a
> new dimension of variation that can be strict-positive on the toy
> benchmark.

# PRD #10-130 — Decorrelation Loss

## 目标
Test whether a **state decorrelation loss** — `L_decorr = λ · off_diag(C)`,
where `C = h_state_state^T` is the hidden-state covariance — improves
task loss on a multi-regime toy benchmark (structured + sinusoidal +
noisy segments in the same sequence) by forcing the hidden state to
spread across orthogonal dimensions rather than collapse onto a single
manifold. Hypothesis: the loss is **strict-positive** because it
adds an *unsupervised disentanglement signal* that doesn't conflict
with task loss, and is well-defined for any model that exposes a
hidden-state tensor.

## 用户故事
- As a gate-line maintainer, I get a fresh axis of variation after 5
  rounds of pulse-only work.
- As a researcher, I confirm that **decorrelation loss is target-data-
  independent** (the same λ value works across toy_sin / structured /
  random) — a property the pulse line lacked.

## 引擎层职责 (canonical)
- `lnn/core/decorrelation_loss.py` (NEW, ~150 LOC): pure-Python loss
  function `state_decorrelation_loss(h, lambda_coeff=0.01)` that takes
  a hidden-state tensor `(B, T, d_h)` and returns `off_diag(C)` where
  `C = h.permute(...).reshape(d_h, -1) @ h.permute(...).reshape(-1, d_h)`,
  normalized by diagonal. Optional EMA-based covariance tracking
  (`use_ema=True`) to stabilize the loss across batches.
- New loss is **plug-and-play** for any cell that exposes a hidden
  state — usable with blend_gated, CfC, MoE cells, etc.

## 游戏层职责
- `scripts/bench_decorrelation_loss.py` (NEW, ~280 LOC): modes =
  {static_tau, blend_gated, blend_gated + decorrelation λ=0.001,
  blend_gated + decorrelation λ=0.01, blend_gated + decorrelation
  λ=0.1}; 3 datasets × 2 seeds × 50 epochs.
- `analysis/decorrelation_loss_bench.json` (NEW, 30 cells).
- `docs/research/2026-07-12_round289_decorrelation_loss_report.md`.

## 验收标准 (H1-H5)
- H1 (target-independent): decorrelation loss improves or maintains
  task loss on ALL 3 datasets (toy_sin / structured / random) at
  λ ∈ {0.001, 0.01}.
- H2 (orthogonality): the loss adds *no new hyperparameters beyond λ*,
  and combines with blend gate without interference.
- H3 (no collapse): the learned state covariance has
  `mean_diag / max_off_diag ≥ 5` (decorrelated axes).
- H4 (strict-positive default): if H1 passes for any λ, the loss is
  +1 SP — first non-pulse SP in this line of work.
- H5 (gradients flow): the loss is differentiable end-to-end through
  the cell.

## 实现难度
**S** (1-2h). ~150 LOC loss + ~10 unit tests + ~250 LOC bench.

## 风险
- If H1 ✗: decorrelation loss conflicts with task loss on at least one
  dataset — try λ=0.0001 (much smaller).
- If H3 ✗: loss has no effect — λ is too small; try λ=1.0.
- If both ✗: the loss formulation is wrong; try whitening instead of
  off-diagonal penalty.