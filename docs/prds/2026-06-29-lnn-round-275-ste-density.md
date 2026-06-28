---
title: "PRD #10-112 — STE × Different Density (Sparsity Sweep)"
round: 275
date: 2026-06-29
author: "Claude (r275 /loop 1h session)"
status: "draft"
parent: "r274 multi-channel d_in=4 finding"
---

# PRD #10-112 — STE × Different Density (Sparsity Sweep)

## Motivation

r267-r274 characterized the r267 STEWithEntropy win along the
**hidden_size**, **τ**, **λ**, **T**, and **d_in** dimensions.
The sweep left **density** (the hard top-k fraction of neurons
that get updated per timestep) unexplored.

Open question: **is density=0.3 optimal?** The current
production setting holds density fixed at 0.3, but the r267
mechanism uses the soft+hard mask entropy reg to softly select
which neurons participate. The hard mask fraction is **density**
of neurons kept (rounded up).

Three possible outcomes:

  1. **density=0.3 is optimal** — confirms current production.
  2. **Smaller density (0.1)** hurts structured (not enough
     capacity) but maybe improves toy_sin (more sparsity = more
     regularization).
  3. **Larger density (0.5, 0.7)** doesn't help (overfitting
     risk + less STE benefit) but may match on structured.

## Why Density Matters

The STEWithEntropy mechanism works as follows:
- **soft_mask**: learnable logits → sigmoid scores
- **hard_mask**: keep top-k = round(density × hidden) neurons
  with straight-through estimator `(hard - soft).detach() + soft`
- **entropy reg**: regularize soft_mask toward uniform across
  top-k (avoid collapse to single neuron)

The hard mask fraction (density) controls:
- **Capacity per step**: more active neurons → more expressive
  per-step update
- **Sparsity level**: lower density → more "competition" among
  neurons → stronger entropy reg signal
- **Gradient flow**: STE backprops through the soft mask while
  using hard forward values; density determines how much gradient
  flows to which neurons

## Modes (4 total)

| mode                       | density | hidden | notes |
|----------------------------|---------|--------|-------|
| ste_entropy_d0.1_h192      | **0.1** | 192    | NEW — sparse |
| ste_entropy_d0.3_h192      | 0.3     | 192    | r267 PRODUCTION |
| ste_entropy_d0.5_h192      | **0.5** | 192    | NEW — dense |
| ste_entropy_d0.7_h192      | **0.7** | 192    | NEW — very dense |

All other params identical to r272 production:
- input_size=1, T=64, ste_temperature=1.0, entropy_lambda=0.1

## Hypotheses

  **H1**: density=0.3 is optimal on structured
  [predicted: CONFIRM — production-locked]

  **H2**: density=0.1 hurts structured (not enough capacity)
  [predicted: CONFIRM — 19 active neurons of 192]

  **H3**: density=0.5+ ≈ density=0.3 on structured
  [predicted: LIKELY — diminishing returns past 0.3]

  **H4**: top1_frac preserved across densities
  [predicted: CONFIRM — mechanism is density-invariant]

  **H5**: logit_std grows with density (more neurons share logit
  budget)
  [predicted: LIKELY]

## Bench Config

  - 4 modes × 3 datasets × 3 seeds = 36 cells
  - 100 epochs, lr=1e-2, batch=16
  - Datasets: toy_sin, structured, random (match r267-r274)

## Expected Outcomes

Best case: density=0.3 strictly optimal → production unchanged.

Likely: density=0.3 ≈ density=0.5 (diminishing returns).
density=0.1 hurts structured. Production unchanged.

Worst case: density=0.1 helps toy_sin (regularization) →
suggests density should be task-conditioned.

## Pattern Audit Predictions

After r275:
  - 66 SP + 28 TD + 61 NEG = 155 (currently)
  - If density=0.3 wins: 0 change (production confirm)
  - If density=0.5 wins: 0 change (parameter tune)
  - If density=0.1 wins: 0 change (sparsity finding)
  - If density=0.7 ≈ density=0.3: 0 change (saturation)

## Files to Add

  - `scripts/bench_ste_density.py` (~370 LOC, reuses r274 bench)
  - `analysis/ste_density_bench.json`
  - `docs/research/2026-06-29_round275_ste_density_report.md`

## Cumulative Test Count

**0 new tests** (r275 is bench-only — reuse r267 STEWithEntropy).
No regressions.
