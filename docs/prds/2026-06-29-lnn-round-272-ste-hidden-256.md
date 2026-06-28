---
title: "PRD #10-109 — STE × Hidden=256 (find saturation)"
round: 272
date: 2026-06-29
author: "Claude (r272 /loop 1h session)"
status: "draft"
parent: "r271 hidden=128 STRICT WIN"
---

# PRD #10-109 — STE × Hidden=256 (find saturation)

## Motivation

r267 + r270 + r271 established that the STEWithEntropy win
**compounds with hidden size** through h=128:

  - r267 (h=16): 0.004791
  - r270 (h=64): 0.000426 (11.2× better than h=16)
  - r271 (h=128): 0.000225 (1.9× better than h=64)

**Power-law fit (r271)**: test_mse ~ hidden^-1.83.
**Predicted h=256**: test_mse ≈ 0.000225 × (2)^-1.83 ≈ **0.000060**.

The natural question: **does the win continue at h=256, or
saturate?**

## Why h=256 May Saturate

  - **Finite task complexity**: structured has 4 segments × 4
    levels = 16 discrete states. At some point, more
    parameters just memorize.

  - **Data starvation**: 256 samples × T=64 = 16,384 input
    pairs. With 33K parameters at h=128 and 132K at h=256,
    we're approaching parameter:sample ratios of 130× and 515×.

  - **r269 gradient saturation**: smaller logit magnitudes
    with more capacity may push gradients into the "no signal"
    region.

## Why h=256 May Still Win

  - **Soft specialization continues**: r271 found top1_frac =
    0.010 at h=128. At h=256, this could drop to 0.005 or
    lower (using ~200 effective experts).

  - **Distribution-not-magnitude continues**: logit_std has
    dropped monotonically. Could continue at h=256.

## Modes (4 total)

| mode                  | hidden | τ   | λ     | notes |
|-----------------------|--------|-----|-------|-------|
| ste_entropy_h64 (r271)| 64     | 1.0 | 0.1   | r271 reference |
| ste_entropy_h128 (r271)| 128   | 1.0 | 0.1   | r271 PRODUCTION |
| **ste_entropy_h256**  | **256**| 1.0 | 0.1   | **NEW** — find saturation |
| **ste_entropy_h192**  | **192**| 1.0 | 0.1   | **NEW** — intermediate |

(Adding h=192 as intermediate point to refine the scaling curve.)

## Hypotheses

  **H1**: prod_h256 ≥ prod_h128 on structured
  [predicted: LIKELY — power-law predicts 2.7× improvement]

  **H2**: prod_h256 ≈ prod_h128 (saturation at h=128)
  [predicted: UNLIKELY — power-law still steep]

  **H3**: logit_std drops further at h=256
  [predicted: CONFIRM — r271 trend continues]

  **H4**: top1_frac drops further at h=256
  [predicted: CONFIRM — softer specialization]

  **H5**: Power-law fit improves with h=256 added
  [predicted: LIKELY — more data points]

  **H6**: h=256 doesn't degrade (vs r269 gradient saturation)
  [predicted: LIKELY — logit_std still > 0]

## Bench Config

  - 4 modes × 3 datasets × 3 seeds = 36 cells
  - 100 epochs, lr=1e-2, batch=16
  - Datasets: toy_sin, structured, random (match r267-r271)

## Expected Outcomes

Best case: prod_h256 wins on structured (~0.000060 from
power-law). Production setting upgrades.

Likely: prod_h256 still improves (1.3-1.5× from h=128).
Diminishing returns confirmed.

Worst case: prod_h256 is worse (overfitting or gradient
saturation). h=128 documented as production saturation point.

## Pattern Audit Predictions

After r272:
  - 66 SP + 28 TD + 61 NEG = 155 (currently)
  - If H1 confirmed (h=256 wins): 0 change (parameter sweep)
  - If H2 confirmed (saturation): 0 change (cap documented)
  - If h=256 is catastrophic: 0 change (target-dep for very large hidden)

## Why h=256 not h=512?

h=512 × density=0.3 = 78K active connections, 300× more
parameters than samples. Pure overfitting territory. h=256
is the **reasonable upper bound** for this bench protocol.

## Files to Add

  - `scripts/bench_ste_hidden_256.py` (~360 LOC, reuses r271 bench)
  - `analysis/ste_hidden_256_bench.json`
  - `docs/research/2026-06-29_round272_ste_hidden_256_report.md`

## Cumulative Test Count

**0 new tests** (r272 is bench-only — reuse r267 STEWithEntropy).