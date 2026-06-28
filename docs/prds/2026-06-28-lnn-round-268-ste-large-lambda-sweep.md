---
title: "PRD #10-105 — STE + Large λ Entropy Sweep"
round: 268
date: 2026-06-28
author: "Claude (r268 /loop 1h session)"
status: "draft"
parent: "r267 STE + entropy reg (STRICT WIN, 6.7× on structured)"
---

# PRD #10-105 — STE + Large λ Entropy Sweep

## Motivation

r267 found that soft-mask entropy reg improves STE on
structured by **6.7×** at λ=0.1 (ste_entropy_medium). The
trend is **monotonically improving** with λ:

| λ        | structured test_mse | improvement |
|----------|---------------------|-------------|
| 0 (r265) | 0.009218            | 1.0×        |
| 0.001    | 0.003095            | 3.0×        |
| 0.01     | 0.002279            | 4.0×        |
| 0.1      | 0.001374            | 6.7×        |

This suggests we haven't reached the optimum yet. r268
extends the sweep to **larger λ values**: 1.0, 10.0, 100.0.

## Why λ > 0.1 May Help

Entropy is **bounded** ∈ [0, log(d_h)]. With d_h=16, log(16) ≈
2.77. The penalty is `λ × H`, so the maximum loss contribution
is `λ × 2.77`.

  - At λ=0.1: max contribution = 0.277 (vs task loss ~0.001-0.01)
  - At λ=1.0: max contribution = 2.77 (vs task loss ~0.001-0.01)
  - At λ=10.0: max contribution = 27.7 (vs task loss ~0.001-0.01)

At λ=10, the entropy term **dominates** the task loss. This
will force the model to minimize entropy aggressively, but
may over-regularize and hurt task performance. r268 tests
whether the monotonic trend continues or breaks.

## Hypothesis

**The trend continues**: λ=1.0, 10.0, 100.0 give **even better**
structured test_mse than λ=0.1. Possible outcomes:

1. **Monotonic improvement** (best case): λ=100 is the best.
2. **Optimum at λ=1.0 or 10.0** (sweet spot): trend reverses
   after a peak.
3. **Overtraining/degeneration** at λ ≥ 10.0: model collapses
   to a degenerate "delta mask" (one edge per row).

The mechanism of "entropy reg grows logits apart" should
saturate once logits are sufficiently different. The
optimum is likely at **λ=1.0 or 10.0**.

## Modes (6 total, extending r267)

| mode                  | λ       | notes                          |
|-----------------------|---------|--------------------------------|
| ste_baseline          | 0.0     | r265 no-reg reference          |
| ste_entropy_small     | 0.01    | r267 reference (won 4× on structured) |
| ste_entropy_medium    | 0.1     | r267 best (won 6.7× on structured) |
| **ste_entropy_large** | **1.0** | **NEW** |
| **ste_entropy_xl**    | **10.0**| **NEW** |
| **ste_entropy_xxl**   | **100.0**| **NEW** |

## Hypotheses

  **H1**: λ=1.0 is at least as good as λ=0.1 on structured.
  [predicted: LIKELY — monotonic trend suggests yes]

  **H2**: λ=10.0 or 100.0 finds the global optimum.
  [predicted: PARTIAL — likely optimum at λ=10, may
  degrade at λ=100]

  **H3**: λ ≥ 10.0 starts to hurt toy_sin/random (other
  datasets over-regularized).
  [predicted: LIKELY — entropy reg is helpful on structured
  but may overfit smooth/noise data]

  **H4**: Soft-mask entropy drops to < 10% of max entropy
  (≈ 0.28) at λ ≥ 1.0 (vs ~99.5% at baseline).
  [predicted: CONFIRM — entropy should drop dramatically]

  **H5**: Logit std continues to grow with λ (no saturation
  or collapse).
  [predicted: CONFIRM up to λ=10, possible collapse at
  λ=100 if too aggressive]

## Bench Config

  - 6 modes × 3 datasets × 2 seeds = 36 cells
  - 100 epochs, hidden=16, lr=1e-2, batch=16
  - Datasets: toy_sin, structured, random (match r267)
  - Metrics: test_mse, soft_mask_entropy, neighbor_logits_std,
    neighbor_logits_abs_mean, hard_mask_top1_sparsity (delta
    check)

## Expected Outcomes

Best case: λ=10.0 is the new optimum on structured
(0.001374 → ~0.0005?). 7× improvement total over r265.

Likely: λ=1.0 is the optimum on structured, λ ≥ 10 starts
to hurt. r267's λ=0.1 is close to optimal.

Worst case: λ=1.0 already hurts structured. r267 was the
peak.

## Files to Add

  - `scripts/bench_ste_entropy_large_lambda.py` (~330 LOC)
  - `analysis/ste_entropy_large_lambda_bench.json`
  - `docs/research/2026-06-28_round268_ste_large_lambda_report.md`

  (No new code needed — reuse STEWithEntropy from r267.)

## Cumulative Test Count

**0 new tests** (r268 is bench-only — reuse r267 STEWithEntropy).
Predicted: 155 classes total (66 SP + 28 TD + 61 NEG).
If λ=10 is new best, may add a 2nd SP this round.

## Pattern Audit

After r268:
  - Currently: 66 SP + 28 TD + 61 NEG = 155 classes
  - Predicted: 
    - Best case: +1 SP (λ=10 or 100 wins)
    - Likely: 0 changes (λ=0.1 already optimum)
    - Worst case: +1 TD (wins structured, hurts others)

## Why This Matters

r267 was a clean win, but the **trend suggests more headroom**.
r268 tests whether the win compounds with larger λ or whether
λ=0.1 was already the sweet spot.

If larger λ wins, it unlocks:
1. **Even better structured performance** (target: 0.0005 or lower).
2. **A reusable λ schedule**: anneal λ from 0 → large to
   concentrate structure gradually.

If larger λ hurts, it confirms λ=0.1 is the optimum and the
mechanism has hit saturation.

## Why Not Just λ=100?

λ=100 may collapse the model (entropy term = 277 vs task loss
≈ 0.01 — 27000× larger). The optimizer would minimize entropy
at all costs, even if task loss explodes. r268 tests this
hypothesis by including λ=100 as the upper bound.