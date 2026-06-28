---
title: "PRD #10-106 — STE + Temperature Sweep (sharper sigmoid)"
round: 269
date: 2026-06-28
author: "Claude (r269 /loop 1h session)"
status: "draft"
parent: "r267 STE + entropy reg, r268 large λ sweep"
---

# PRD #10-106 — STE + Temperature Sweep

## Motivation

r268 found that even with strong entropy reg (λ=10), soft-mask
entropy **saturates at 96% of max** (not <10% as predicted).
The hypothesis: this is a **sigmoid saturation** problem —
when logit values >> τ_ste, the sigmoid is already near 1.0
and further logit growth has no effect.

Concretely, with τ_ste = 1.0 (current setting):
  - sigmoid(logit/1.0) at logit=5 = 0.993
  - sigmoid(logit/1.0) at logit=10 = 0.99995

So once logit std reaches ~5, the soft mask is already
near-binary. To concentrate further, we need a **sharper
sigmoid** (smaller τ).

r269 sweeps τ_ste ∈ {1.0, 0.5, 0.3, 0.1} at λ=0.1 (r267's
production λ).

## Why τ Matters

The STE soft mask is `sigmoid(logit/τ)`. The **gradient**
through this sigmoid is `sigmoid'(logit/τ) / τ`, which is
maximal when `logit/τ ≈ 0` and decays as `logit/τ` moves away.

With smaller τ:
  - The sigmoid is sharper (more binary).
  - Gradients are larger near `logit = 0` (saturated regions
    still have ~zero gradient, but the **transition region**
    is narrower).
  - The row-softmax of the sigmoid has **lower entropy floor**.
  - Entropy reg can push the model further toward concentration.

The trade-off: smaller τ means the **STE approximation
becomes less accurate** (forward mask is hard binary, backward
mask is sharper sigmoid — the approximation error grows).

## Hypothesis

Smaller τ at λ=0.1 improves over τ=1.0 + λ=0.1 on structured.
Possible outcomes:

1. **Monotonic improvement** (best case): τ=0.1 is the new
   optimum (perhaps 2-3× better than r267).
2. **Sweet spot at τ=0.3 or 0.5**: smaller is better up to a
   point, then breaks (gradient noise).
3. **No improvement**: τ=1.0 is already optimal because the
   entropy reg at λ=0.1 doesn't need sharper sigmoid.

## Modes (5 total)

| mode                        | τ     | λ     | notes |
|-----------------------------|-------|-------|-------|
| ste_baseline_t1_l0          | 1.0   | 0.0   | r265 no-reg reference |
| ste_entropy_t1_l0p1 (r267)  | 1.0   | 0.1   | r267 best (production) |
| **ste_entropy_t0p5_l0p1**   | **0.5** | **0.1** | **NEW** |
| **ste_entropy_t0p3_l0p1**   | **0.3** | **0.1** | **NEW** |
| **ste_entropy_t0p1_l0p1**   | **0.1** | **0.1** | **NEW** |

## Hypotheses

  **H1**: Smaller τ at λ=0.1 is at least as good as τ=1.0 on
  structured.
  [predicted: LIKELY at τ=0.3 or 0.5]

  **H2**: Soft-mask entropy drops below 96% of max with
  smaller τ.
  [predicted: CONFIRM — sharper sigmoid has lower floor]

  **H3**: Logit std SATURATES at lower value with smaller τ
  (because saturation is reached earlier).
  [predicted: CONFIRM — logit/τ → ±∞ faster]

  **H4**: Best τ is not the smallest (τ=0.1 may have gradient
  noise).
  [predicted: τ=0.3 is the sweet spot]

  **H5**: Smaller τ doesn't hurt toy_sin/random.
  [predicted: CONFIRM — toy_sin/random insensitive to τ in r267]

## Bench Config

  - 5 modes × 3 datasets × 2 seeds = 30 cells
  - 100 epochs, hidden=16, lr=1e-2, batch=16
  - Datasets: toy_sin, structured, random (match r268)
  - Metrics: test_mse, soft_mask_entropy, neighbor_logits_std,
    entropy_fraction, top1_frac

## Expected Outcomes

Best case: τ=0.3 + λ=0.1 is the new optimum on structured
(~0.0005 or lower). 2-3× better than r267.

Likely: τ=0.5 is the optimum, with monotonic improvement
from τ=1.0 → τ=0.5 and plateau thereafter.

Worst case: τ=1.0 is already optimal (entropy reg at λ=0.1
doesn't need sharper sigmoid).

## Files to Add

  - `scripts/bench_ste_temperature_sweep.py` (~340 LOC)
  - `analysis/ste_temperature_sweep_bench.json`
  - `docs/research/2026-06-28_round269_ste_temperature_report.md`

  (No new code needed — reuse STEWithEntropy from r267.)

## Cumulative Test Count

**0 new tests** (r269 is bench-only — reuse r267 STEWithEntropy).

## Pattern Audit

After r269:
  - Currently: 66 SP + 28 TD + 61 NEG = 155 classes
  - Predicted:
    - Best case: +1 SP (smaller τ + λ=0.1 wins)
    - Likely: 0 changes (τ=1.0 already optimal)
    - Worst case: +1 NEG (smaller τ hurts)

## Why This Matters

r267/r268 established that λ=0.1 + τ=1.0 is a STRICT WIN.
r269 tests whether the win can be **compounded** by adjusting
τ.

If τ=0.5 or τ=0.3 wins, it unlocks:
1. **Even better structured performance** (compound effect).
2. **New "sharper STE" default** — replace τ=1.0 with τ=0.5
   as the production setting.

If τ=1.0 wins, it confirms the r267 setting is globally
optimal.

## Why Not Just τ=0.1?

τ=0.1 may produce gradient noise because the sigmoid is
nearly step-function. The forward mask is hard (binary), but
the backward sigmoid is sharp — small logit changes produce
huge gradient changes. This can destabilize training. r269
includes τ=0.1 as the upper bound on sharpness.