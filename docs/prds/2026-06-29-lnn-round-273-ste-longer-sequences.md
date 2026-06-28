---
title: "PRD #10-110 — STE × Longer Sequences (T=128)"
round: 273
date: 2026-06-29
author: "Claude (r273 /loop 1h session)"
status: "draft"
parent: "r272 hidden=192 saturation"
---

# PRD #10-110 — STE × Longer Sequences (T=128)

## Motivation

r267-r272 characterized the r267 STEWithEntropy win along the
**hidden_size** dimension (h=16 → 192). All bench rounds used
**T=64** sequence length.

Open question: **does the win hold for longer sequences?**

Three possible outcomes:

  1. **Compounds at longer T**: structured learning benefits
     from more timesteps (more averaging). Production T
     upgrades from 64 to 128.
  2. **Stable at longer T**: the win is **independent of T**
     — same test_mse at T=128 as T=64. T=128 is "more data"
     but not "more learning".
  3. **Degrades at longer T**: τ dynamics become harder to
     train over longer horizons (gradient issues with long
     BPTT).

## Why Longer Sequences May Help

  - **More samples per epoch**: T=128 = 2× more (x_t, y_t)
    pairs per sequence. 4× more total data per batch.
  - **Longer-range dependencies**: structured has 4 segments
    spread over T=64 (16 timesteps each). At T=128, segments
    are 32 timesteps each. The model can learn the same
    structure with more temporal margin.
  - **τ dynamics**: per-neuron time constants τ have more
    time to evolve. Less averaging noise per τ step.

## Why Longer Sequences May Hurt

  - **BPTT cost**: 2× longer sequence = 2× more BPTT steps.
    Already at h=192 with 100 epochs, training is heavy.
  - **τ stability**: with longer T, very slow τ might lead
    to "memory decay" issues. Fast τ might over-fit.
  - **Hidden state drift**: if τ integration is unstable,
    errors compound over longer T.

## Modes (4 total)

| mode                  | T   | hidden | τ   | λ     | notes |
|-----------------------|-----|--------|-----|-------|-------|
| ste_entropy_t64_h64   | 64  | 64     | 1.0 | 0.1   | r270 reference |
| ste_entropy_t64_h192  | 64  | **192**| 1.0 | 0.1   | r272 PRODUCTION |
| **ste_entropy_t128_h64** | **128** | 64 | 1.0 | 0.1 | NEW — same hidden, longer T |
| **ste_entropy_t128_h192**| **128** | **192** | 1.0 | 0.1 | NEW — longer T, full prod |

(Adding T=128 × hidden=64 to disentangle T effect from
hidden effect.)

## Hypotheses

  **H1**: T=128 ≥ T=64 on structured (longer helps)
  [predicted: LIKELY for structured — more averaging]

  **H2**: h=192 still wins at T=128
  [predicted: CONFIRM — hidden effect is independent of T]

  **H3**: τ dynamics more stable at T=128
  [predicted: CONFIRM — more time for τ to evolve]

  **H4**: logit_std pattern preserved at T=128
  [predicted: CONFIRM — mechanism is T-independent]

  **H5**: top1_frac pattern preserved at T=128
  [predicted: CONFIRM — mechanism is T-independent]

## Bench Config

  - 4 modes × 3 datasets × 3 seeds = 36 cells
  - 100 epochs, lr=1e-2, batch=16
  - Datasets: toy_sin, structured, random (match r267-r272)

## Expected Outcomes

Best case: T=128 is significantly better than T=64 (maybe
2× improvement on structured).

Likely: T=128 is approximately equal to T=64 (same task,
more data → marginal benefit).

Worst case: T=128 is worse (BPTT instability, τ drift).

## Pattern Audit Predictions

After r273:
  - 66 SP + 28 TD + 61 NEG = 155 (currently)
  - If T=128 wins: 0 change (parameter sweep)
  - If T=128 same: 0 change (T-invariant finding)
  - If T=128 worse: 0 change (T-limit finding)

## Files to Add

  - `scripts/bench_ste_longer_sequences.py` (~370 LOC, reuses r272 bench)
  - `analysis/ste_longer_sequences_bench.json`
  - `docs/research/2026-06-29_round273_ste_longer_sequences_report.md`

## Cumulative Test Count

**0 new tests** (r273 is bench-only — reuse r267 STEWithEntropy).

## Why T=128 not T=256?

T=256 with 100 epochs and h=192 = 4× compute cost of T=64
bench. r273 stays at T=128 (2× cost) to keep within bench
budget. r274 can extend to T=256 if T=128 wins.