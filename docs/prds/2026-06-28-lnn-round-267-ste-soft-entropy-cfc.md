---
title: "PRD #10-104 — STE + Soft-Mask Entropy Reg"
round: 267
date: 2026-06-28
author: "Claude (r267 /loop 1h session)"
status: "draft"
parent: "r265 STE, r266 STE+L1 (HONEST NEGATIVE)"
---

# PRD #10-104 — STE + Soft-Mask Entropy Reg

## Motivation

r265 introduced STENeuronWiseCfCCell: forward uses hard top-k
sparsity (true binary mask), backward uses soft sigmoid (gradient
flows). **STRICT WIN** on structured (-25% test_mse vs r263).

r266 tested L1 sparsity reg on neighbor_logits as a way to
encourage concentrated structure. **HONEST NEGATIVE**: L1
collapses logits to near-zero (88-100% near-zero with λ=0.01),
destroying both the hard top-k structure (random tiebreaks)
and the soft gradient (uniform mask → zero grad).

**Root cause of r266 failure**: L1 targets the **magnitude**
of logits. STE ranks by magnitude (top-k), so collapsing
magnitudes destroys the ranking.

## Hypothesis

A better target is the **entropy of the soft sigmoid mask**,
not the logit magnitudes:

  L_total = L_task + λ × H(soft_mask)

Where H is the per-row Shannon entropy of the soft sigmoid
output. Low entropy means the soft mask is **concentrated**
(a few dominant edges). High entropy means **uniform**
(ambiguous edges).

**Why this should help**:
1. **Concentration is the right goal**: a few clear edges vs.
   many ambiguous ones — directly targets the property we
   want.
2. **Preserves magnitude information**: low-entropy masks can
   still have varied magnitudes (concentrated vs. uniform).
3. **Backward-friendly**: entropy gradient is smooth and
   bounded (unlike L1's non-smooth at zero).
4. **Complements hard top-k**: top-k ensures forward sparsity,
   entropy ensures soft mask is also concentrated → harder
   ambiguity at the boundary.

## Mechanism

For each row `i` of the soft mask `S` (shape `[d_h, d_h]`):

```
p_i = softmax(soft_mask[i])  # row-normalized probability
H_i = -sum_j(p_i[j] * log(p_i[j] + eps))
entropy_loss = mean over rows i of H_i
```

For a uniform row: H_i = log(d_h) (max entropy).
For a delta row: H_i ≈ 0 (min entropy).

`L_total = L_task + λ × entropy_loss`

Range of entropy: [0, log(d_h)]. With d_h=16, log(16) ≈ 2.77.
Reasonable λ range: [0.001, 0.1] (entropy is bounded, so λ
doesn't need to scale with logit magnitude).

## Modes (4 total)

| mode                  | λ       | notes                          |
|-----------------------|---------|--------------------------------|
| ste_baseline          | 0.0     | r265 no-reg reference          |
| ste_entropy_tiny      | 0.001   | light entropy reg              |
| ste_entropy_small     | 0.01    | moderate                       |
| ste_entropy_medium    | 0.1     | strong (but bounded)           |

## Hypotheses

  **H1**: STE + entropy reg (any λ) beats r265 no-reg on
  ≥ 1 dataset. [predicted: LIKELY on structured]

  **H2**: Sweet spot for λ exists — too small (no effect)
  or too large (over-regularizes) is worse.
  [predicted: λ=0.01 likely sweet spot]

  **H3**: Entropy reg REDUCES soft-mask entropy (clearer
  concentration) without collapsing logits.
  [predicted: CONFIRM — entropy should drop]

  **H4**: Entropy reg preserves logit std (no L1 collapse).
  [predicted: CONFIRM — entropy targets mask, not logits]

  **H5**: Entropy reg is a strict superset of no-reg (λ → 0
  recovers r265).
  [predicted: CONFIRM]

## Bench Config

  - 4 modes × 3 datasets × 2 seeds = 24 cells
  - 100 epochs, hidden=16, lr=1e-2, batch=16
  - Datasets: toy_sin, structured, random (match r266)
  - Metrics: test_mse, soft_mask_entropy, neighbor_logits_std,
    neighbor_logits_abs_mean, frac_near_zero, hard_mask_count

## Expected Outcomes

If r267 is a strict win:
  - structured test_mse < 0.0092 (r265 baseline)
  - entropy drops ~50% on structured (high concentration)
  - logit std preserved (no L1 collapse)
  - toy_sin unchanged or slightly better

If r267 is target-dep:
  - Wins on structured, neutral on toy_sin, regression on random

If r267 is honest negative:
  - All datasets degrade (entropy reg unhelpful in 1D)

## Files to Add

  - `lnn/core/ste_entropy_neuron_wise_cfc.py` (~110 LOC)
  - `tests/test_ste_entropy_neuron_wise_cfc.py` (~150 LOC)
  - `scripts/bench_ste_entropy_neuron_wise_cfc.py` (~280 LOC)
  - `analysis/ste_entropy_neuron_wise_cfc_bench.json`
  - `docs/research/2026-06-28_round267_ste_entropy_report.md`

## Cumulative Test Count

~14 new tests (STEWithEntropy: basics, loss, forward, mask).
Target: 14/14 pass.

## Pattern Audit

After r267:
  - Currently: 65 SP + 28 TD + 61 NEG = 154 classes
  - Predicted: +1 (likely SP or TD if hypothesis holds)
  - Worst case: +1 NEG (entropy reg unhelpful)

## Why This Matters

r266 demonstrated that the wrong regularizer (L1) destroys
STE's structure learning. r267 tests whether the RIGHT
regularizer (entropy) helps STE concentrate its structure
without destroying the magnitude/ranking information.

If entropy reg wins, it unlocks:
1. **Larger models**: with entropy reg, STE scales to deeper
   layers without top-k ambiguity.
2. **Composition with other regs**: entropy can stack with
   weight reg, activation reg without interfering.
3. **Interpretable structure**: low-entropy masks are easier
   to visualize and explain.

If entropy reg loses, we have a clean rejection of the
"concentration helps" hypothesis and the next step is to
test ranking-based regularizers (e.g., pairwise margin loss
on the top-k boundary).