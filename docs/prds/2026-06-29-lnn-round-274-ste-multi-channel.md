---
title: "PRD #10-111 — STE × Multi-Channel Input (d_in=4)"
round: 274
date: 2026-06-29
author: "Claude (r274 /loop 1h session)"
status: "draft"
parent: "r273 T=64 production lock"
---

# PRD #10-111 — STE × Multi-Channel Input (d_in=4)

## Motivation

r267-r273 characterized the r267 STEWithEntropy win along the
**hidden_size**, **τ**, **λ**, and **T** dimensions. All bench
rounds used **d_in=1** (single-channel input).

Open question: **does the win hold with more input channels?**

Three possible outcomes:

  1. **Helps**: more input diversity → better structured learning.
     Production d_in upgrades from 1 to 4.
  2. **Independent**: the win is **d_in-invariant** — same
     test_mse at d_in=4 as d_in=1. Multi-channel is just
     "more data" but not "more learning".
  3. **Hurts**: more channels make the task harder (more
     parameters needed to process input).

## Why Multi-Channel May Help

  - **More information per timestep**: structured has 4 levels.
     Multiple channels could encode different aspects of the
     signal.
  - **Better generalization**: with d_in=4, the model has 4
     features to learn from, may find easier representations.
  - **Real-world alignment**: real time series usually have
     multiple channels (sensors, modalities).

## Why Multi-Channel May Hurt

  - **Noise dilution**: extra channels (if noise) could
     obscure the signal.
  - **Parameter overhead**: more input → more input projection
     parameters.
  - **For our protocol**: extra channels are pure noise
     (random gaussian, see r268 bench). They add no signal.

## How Extra Channels are Generated

In our r267-r273 bench protocol, d_in=1 means input is just
[y_t]. To make d_in=4, we add 3 noise channels:
```
x = [y_t, 0.01 * randn, 0.01 * randn, 0.01 * randn]
```

The 3 extra channels are **small noise** (std=0.01). They
should add no signal but may test whether the model can
**ignore irrelevant inputs** (a generalization test).

## Modes (4 total)

| mode                  | d_in | hidden | T   | notes |
|-----------------------|------|--------|-----|-------|
| ste_entropy_d1_h64    | 1    | 64     | 64  | r270 reference |
| ste_entropy_d1_h192   | 1    | **192**| 64  | r272 PRODUCTION |
| **ste_entropy_d4_h64**| **4**| 64     | 64  | NEW — same hidden, multi-channel |
| **ste_entropy_d4_h192**| **4**| **192**| 64  | NEW — multi-channel, full prod |

## Hypotheses

  **H1**: d=4 ≥ d=1 on structured (more channels help)
  [predicted: UNLIKELY — extra channels are noise]

  **H2**: h=192 still wins at d=4
  [predicted: CONFIRM — hidden effect is independent of d_in]

  **H3**: d=4 ≈ d=1 on toy_sin (T-invariant)
  [predicted: CONFIRM — toy_sin is independent of d_in]

  **H4**: d=4 doesn't degrade structured significantly
  [predicted: LIKELY — model should ignore noise channels]

  **H5**: top1_frac pattern preserved at d=4
  [predicted: CONFIRM — mechanism is d_in-invariant]

## Bench Config

  - 4 modes × 3 datasets × 3 seeds = 36 cells
  - 100 epochs, lr=1e-2, batch=16
  - Datasets: toy_sin, structured, random (match r267-r273)

## Expected Outcomes

Best case: d=4 helps structured by 1.5×. Production d_in
upgrades.

Likely: d=4 ≈ d=1 (mechanism is d_in-invariant). No change to
production.

Worst case: d=4 hurts significantly (model can't ignore noise
channels).

## Pattern Audit Predictions

After r274:
  - 66 SP + 28 TD + 61 NEG = 155 (currently)
  - If d=4 wins: 0 change (parameter sweep)
  - If d=4 same: 0 change (d_in-invariant finding)
  - If d=4 hurts: 0 change (d_in-target-dep)

## Files to Add

  - `scripts/bench_ste_multi_channel.py` (~370 LOC, reuses r273 bench)
  - `analysis/ste_multi_channel_bench.json`
  - `docs/research/2026-06-29_round274_ste_multi_channel_report.md`

## Cumulative Test Count

**0 new tests** (r274 is bench-only — reuse r267 STEWithEntropy).