---
title: "PRD #10-107 — STE × Hidden Size Scale-up"
round: 270
date: 2026-06-28
author: "Claude (r270 /loop 1h session)"
status: "draft"
parent: "r267 STEWithEntropy + r268 λ sweep + r269 τ sweep"
---

# PRD #10-107 — STE × Hidden Size Scale-up

## Motivation

r267 + r268 + r269 confirmed `STEWithEntropy(density=0.3, τ=1.0, λ=0.1)` as the production winner at **hidden=16**. All three rounds were parameter sweeps at fixed capacity.

Open question: **does the win scale?**

Three possibilities:

  1. **Compounds**: hidden=32 or 64 also benefits from entropy reg. Mechanism (concentration pressure) is capacity-independent.
  2. **Saturates**: hidden=16 is sufficient for the structured-task regime; larger hidden just adds noise.
  3. **Reverts**: at scale, the gradient saturation issue from r269 (large logits near 3τ) becomes worse with more parameters, and STEWithEntropy's gain shrinks.

## Why hidden=16 May Be the Sweet Spot

With hidden=16 and density=0.3, the active mask has 16 × 16 × 0.3 ≈ 77 active connections per layer (a small, dense graph). At hidden=64, it's 16 × 64 × 0.3 ≈ 1230 active connections — **16× more**. The entropy reg at λ=0.1 must concentrate a much larger soft-mask, possibly overwhelming the signal.

Counter-argument: more capacity = more logit room = easier to reach large separation without gradient saturation.

## Modes (4 total)

| mode                  | hidden | τ   | λ     | notes |
|-----------------------|--------|-----|-------|-------|
| ste_baseline_h16      | 16     | 1.0 | 0.0   | r265 ref (production scale) |
| ste_entropy_h16       | 16     | 1.0 | 0.1   | r267 prod (PROD scale) |
| ste_entropy_h32       | **32** | 1.0 | 0.1   | **NEW** — 2× capacity |
| ste_entropy_h64       | **64** | 1.0 | 0.1   | **NEW** — 4× capacity |

(Plus an optional 5th mode: `ste_baseline_h32` to disentangle scale vs entropy effect.)

## Hypotheses

  **H1**: prod_h16 ≤ prod_h32 on structured (entropy reg compounds at scale)
  [predicted: LIKELY — reg mechanism is capacity-independent]

  **H2**: prod_h32 ≤ prod_h64 on structured (further compounds)
  [predicted: UNLIKELY — capacity may already be saturated]

  **H3**: Larger hidden reduces seed variance (more averaging)
  [predicted: CONFIRM — central limit theorem]

  **H4**: Logit std grows with hidden size (more capacity → larger logit separation possible)
  [predicted: CONFIRM — more connections = more space]

  **H5**: prod_h32 ≥ base_h32 (entropy reg still helps at scale)
  [predicted: CONFIRM — entropy reg mechanism preserved at scale]

## Bench Config

  - 4 modes × 3 datasets × 3 seeds = 36 cells
  - 100 epochs, lr=1e-2, batch=16
  - Datasets: toy_sin, structured, random (match r267/r268/r269)
  - Metrics: test_mse, soft_mask_entropy, neighbor_logits_std, entropy_fraction, top1_frac

## Expected Outcomes

Best case: prod_h64 wins on structured (5-10× better than prod_h16). 2-round "compounds" finding.

Likely: prod_h32 ≈ prod_h16 (capacity already saturated); seed variance drops.

Worst case: prod_h64 > prod_h16 (overfitting or gradient saturation from r269 gets worse at scale).

## Pattern Audit Predictions

After r270:

  - 66 SP + 28 TD + 61 NEG = 155 (currently)

  - If H1/H2 confirmed: 0 change (parameter sweep, not new mechanism)
  - If H2 confirmed with new best: +1 SP (compounds at scale)
  - If H2 rejected: 0 change (capacity saturates)

## Files to Add

  - `scripts/bench_ste_hidden_size.py` (~340 LOC, reuses r269 bench infrastructure)
  - `analysis/ste_hidden_size_bench.json`
  - `docs/research/2026-06-28_round270_ste_hidden_size_report.md`

## Cumulative Test Count

**0 new tests** (r270 is bench-only — reuse r267 STEWithEntropy + r265 STENeuronWiseCfCCell).

## Why This Matters

r267/r268/r269 confirmed the win. r270 tests whether it scales. If it does, we unlock 3-4× more capacity for production. If it doesn't, hidden=16 is documented as the production sweet spot.

## Why Not Just hidden=128?

hidden=128 × density=0.3 = ~5000 active connections — too many for 256 samples. Risk of pure noise. r270 stops at hidden=64.