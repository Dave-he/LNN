---
title: "PRD #10-108 — STE × Hidden=128 (extend scale-up)"
round: 271
date: 2026-06-28
author: "Claude (r271 /loop 1h session)"
status: "draft"
parent: "r270 hidden=64 STRICT WIN"
---

# PRD #10-108 — STE × Hidden=128 (extend scale-up)

## Motivation

r270 found that the r267 win **compounds at scale**: h=16 → h=64
improves structured test MSE by **11.2×** and reduces seed
variance by **37×**. **No saturation observed at h=64**.

The natural question: **does the win continue at h=128?**

Three possible outcomes:

  1. **Compounds further** (best case): h=128 is even better
     than h=64. Production setting upgrades again.
  2. **Saturates**: h=64 is the sweet spot. h=128 ≈ h=64.
  3. **Reverts**: h=128 is **worse** than h=64 (overfitting
     or capacity-induced gradient issues from r269).

## Why h=128 May Saturate or Revert

  - **Capacity**: 128 × 128 × 0.3 ≈ 5000 active connections
    per layer. With 256 training samples (T=64), this is
    approaching the **underdetermined regime** (more
    parameters than data points).

  - **Gradient saturation** (from r269): the entropy reg
    needs logits to grow past `3τ` to concentrate. With
    more parameters, individual logit contributions become
    smaller (each connection has less individual impact),
    making it harder to push concentration.

  - **Overfitting**: with 8577 parameters and 256 samples,
    the model has 33× more parameters than samples. Easy to
    memorize toy_sin / random (already saturated).

## Why h=128 May Still Win

  - **Concentration headroom**: more connections = more
    space for the entropy reg to "spread" the mask.
    h=128 may produce even softer specialization than h=64
    (top1_frac < 0.022).

  - **Distributed representation**: per the r270 finding,
    the model uses DISTRIBUTION not MAGNITUDE at scale.
    h=128 should continue this trend.

## Modes (3 total)

| mode                  | hidden | τ   | λ     | notes |
|-----------------------|--------|-----|-------|-------|
| ste_entropy_h32 (r270)| 32     | 1.0 | 0.1   | r270 reference |
| ste_entropy_h64 (r270)| 64     | 1.0 | 0.1   | r270 PRODUCTION |
| **ste_entropy_h128**  | **128**| 1.0 | 0.1   | **NEW** — extend scale |

(plus optional comparison: `ste_baseline_h128` to disentangle
entropy reg from scale.)

## Hypotheses

  **H1**: prod_h128 ≥ prod_h64 on structured
  [predicted: LIKELY — compounds further, but possibly saturating]

  **H2**: prod_h128 ≈ prod_h64 (saturates at h=64)
  [predicted: LIKELY — capacity may be saturating]

  **H3**: h=128 reduces seed variance vs h=64
  [predicted: CONFIRM — central limit continues]

  **H4**: logit_std drops further at h=128 (more distribution)
  [predicted: LIKELY — extrapolation from h=64 finding]

  **H5**: top1_frac drops further at h=128 (softer specialization)
  [predicted: CONFIRM — r270 trend continues]

## Bench Config

  - 3 modes × 3 datasets × 3 seeds = 27 cells
  - 100 epochs, lr=1e-2, batch=16
  - Datasets: toy_sin, structured, random (match r267-r270)

## Expected Outcomes

Best case: prod_h128 wins on structured (5-10× better than h=64).

Likely: prod_h128 ≈ prod_h64 (saturation confirmed at h=64).

Worst case: prod_h128 > prod_h64 (overfitting or gradient issues).

## Pattern Audit Predictions

After r271:
  - 66 SP + 28 TD + 61 NEG = 155 (currently)
  - If H1 confirmed (h=128 wins): 0 change (parameter sweep)
  - If H2 confirmed (saturation): 0 change (cap reached)
  - If h=128 is catastrophic: 0 change (target-dep for very large hidden)

## Why Not Skip to h=256?

h=256 × density=0.3 = 20K active connections, 80× more
parameters than samples. Pure overfitting territory. r271
stops at h=128 as the **reasonable upper bound** for this
bench protocol.

## Files to Add

  - `scripts/bench_ste_hidden_128.py` (~360 LOC, reuses r270 bench)
  - `analysis/ste_hidden_128_bench.json`
  - `docs/research/2026-06-28_round271_ste_hidden_128_report.md`

## Cumulative Test Count

**0 new tests** (r271 is bench-only — reuse r267 STEWithEntropy).