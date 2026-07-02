---
title: "PRD #10-113 — STE × Batch Size (Stochastic Optimization Sweep)"
round: 276
date: 2026-06-29
author: "Claude (r276 /loop 1h session)"
status: "draft"
parent: "r275 density sweep complete"
---

# PRD #10-113 — STE × Batch Size (Stochastic Optimization Sweep)

## Motivation

The (τ, λ, hidden, T, d_in, density) sweep is now complete
(r267-r275). The r275 report recommended **batch size** as the
next natural parameter to explore.

Open question: **is batch=16 optimal?** The r267-r275 bench all
used batch=16. Batch size affects:

1. **Gradient noise level**: smaller batch → noisier gradients →
   stronger implicit regularization, but more variance.
2. **Convergence speed**: larger batch → fewer updates per epoch →
   possibly worse learning for the same compute.
3. **STE-specific effect**: STE relies on the soft mask gradient.
   Noisier gradients (small batch) may interfere with the soft
   mask's concentration.

Three possible outcomes:

  1. **batch=16 is optimal** — confirms current production.
  2. **Smaller batch (4, 8)** helps structured (more gradient
     noise = better generalization).
  3. **Larger batch (32, 64)** helps structured (cleaner gradients
     converge to better minima).

## Why Batch Size Matters for STEWithEntropy

The STE mechanism has two distinct gradient sources:
- **Task gradient** from MSE loss (backpropagates through hard mask)
- **Entropy reg gradient** from soft mask distribution (backpropagates through soft mask)

These gradients add (model.extra_loss()) and the optimizer (Adam)
combines them. The relative magnitudes matter:
- At large batch: task gradient is accurate, entropy reg has
  consistent direction → balanced learning.
- At small batch: task gradient is noisy, entropy reg is noisy →
  possibly imbalanced.

The r267 finding was that λ=0.1 is the right entropy weight.
But λ may interact with batch size: at small batch, the
relative entropy contribution per step may be too high.

## Modes (5 total)

| mode                       | batch | hidden | notes |
|----------------------------|-------|--------|-------|
| ste_entropy_b4_h192        | **4** | 192    | NEW — small batch |
| ste_entropy_b8_h192        | **8** | 192    | NEW — medium-small |
| ste_entropy_b16_h192       | 16    | 192    | r267-r275 PRODUCTION |
| ste_entropy_b32_h192       | **32**| 192    | NEW — large batch |
| ste_entropy_b64_h192       | **64**| 192    | NEW — full batch (256/4=64 epochs) |

All other params identical to r275 production:
- input_size=1, T=64, density=0.3, ste_temperature=1.0, entropy_lambda=0.1
- 100 epochs, lr=1e-2

## Hypotheses

  **H1**: batch=16 is optimal on structured
  [predicted: CONFIRM — production-locked]

  **H2**: smaller batch (4, 8) doesn't hurt structured
  [predicted: LIKELY — implicit regularization helps]

  **H3**: larger batch (32, 64) ≈ batch=16 on structured
  [predicted: LIKELY — diminishing returns past 16]

  **H4**: top1_frac preserved across batch sizes
  [predicted: CONFIRM — mechanism is batch-invariant]

  **H5**: smaller batch reduces seed variance (better averaging)
  [predicted: LIKELY — more updates per epoch]

## Bench Config

  - 5 modes × 3 datasets × 3 seeds = 45 cells
  - 100 epochs, lr=1e-2
  - Datasets: toy_sin, structured, random (match r267-r275)

## Expected Outcomes

Best case: batch=16 strictly optimal → production unchanged.

Likely: batch=16 is best but batch=8 is competitive (production
could switch to batch=8 if seed variance is better).

Worst case: large batch (64) hurts because of fewer gradient
updates per epoch (256/64 = 4 updates per epoch).

## Pattern Audit Predictions

After r276:
  - 66 SP + 28 TD + 62 NEG = 156 (currently)
  - If batch=16 wins: 0 change (production confirm)
  - If batch=8 wins: 0 change (parameter tune)
  - If batch=64 hurts: +1 NEG (fewer updates hurts STE)

## Files to Add

  - `scripts/bench_ste_batch_size.py` (~370 LOC, reuses r275 bench)
  - `analysis/ste_batch_size_bench.json`
  - `docs/research/2026-06-29_round276_ste_batch_size_report.md`

## Cumulative Test Count

**0 new tests** (r276 is bench-only — reuse r267 STEWithEntropy).
No regressions.