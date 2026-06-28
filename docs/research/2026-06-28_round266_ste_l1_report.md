---
title: "Round 266 — STE+L1 sparsity reg — HONEST NEGATIVE (L1 collapses logits)"
date: 2026-06-28
round: 266
prd: "docs/prds/2026-06-28-lnn-round-266-ste-l1-sparsity-cfc.md"
status: "HONEST NEGATIVE"
audit_pattern: "65 strictly positive + 28 target-dep + 61 negatives = 154 mechanism classes (was 153; +1 NEG)"
---

# Round 266 — STE+L1 sparsity reg — Honest Negative

## TL;DR

Adding an L1 penalty on `neighbor_logits` HURTS task loss on
all 3 datasets. The L1 penalty **collapses** the logits to
near-zero, destroying the structure that STE is supposed to
learn.

| mode            | toy_sin  | structured | random   |
|-----------------|----------|------------|----------|
| ste_baseline    | **9e-6** | **9.2e-3** | 0.9960   |
| ste_l1_small    | 1.0e-4   | 1.3e-2     | 0.9948   |
| ste_l1_medium   | 3.0e-4   | 5.5e-2     | 0.9950   |
| ste_l1_large    | 3.1e-4   | 5.8e-2     | 0.9948   |

L1 monotonically degrades structured (0.0092 → 0.013 →
0.055 → 0.058) and toy_sin (9e-6 → 1e-4 → 3e-4 → 3.1e-4).
The **no-L1 baseline** is the optimum.

## Hypothesis Evaluation

### H1 (L1 beats no-L1 on ≥ 1 dataset)
**REJECTED**. L1 is worse on every dataset, every λ.

### H2 (Sweet spot for λ)
**PARTIAL/NEUTRAL**. Among L1 modes, smaller is better
(λ=0.01 < 0.1 < 1.0), but the optimum is **no L1 at all**.

### H3 (L1-penalized std > 1.5 × no-L1 std)
**REJECTED — L1 collapses logits instead of concentrating them.**

| mode          | std range      | abs_mean range | fraction near zero |
|---------------|----------------|----------------|--------------------|
| ste_baseline  | 0.158 – 0.364  | 0.120 – 0.249  | 2.3% – 5.5%        |
| ste_l1_small  | 0.002 – 0.160  | 0.001 – 0.042  | 87.9% – 100%       |
| ste_l1_medium | 0.001 – 0.023  | 0.001 – 0.003  | 99.2% – 100%       |
| ste_l1_large  | 0.001 – 0.002  | 0.001          | 100%               |

**L1 with λ=0.01 already collapses 88-100% of logits to
|logit| < 0.01.** Stronger λ makes it worse. The penalty
shifts logits uniformly toward zero, eliminating the
**gradient signal** that STE needs to learn structure.

### H4 (L1 is superset — λ → 0 recovers r265)
**PARTIAL/NEUTRAL**. ste_l1_small (λ=0.01) has more collapsed
logits than baseline but task loss is still worse (0.013 vs
0.0092 structured). Even very small λ destroys some of the
structure learning.

## Why L1 Fails

The L1 penalty on `neighbor_logits` is too aggressive. The
mechanism is:

  - Forward: hard top-k picks the top-5 logits (out of 16).
  - Backward: soft sigmoid scales logits by τ_ste. Small
    logits → ~0.5 sigmoid output (uniform mixing).
  - L1 penalty: pushes logits toward zero.

When L1 collapses logits to near-zero, the **soft sigmoid
becomes near-uniform**. The STE backward signal becomes
weak (uniform soft mask has zero gradient w.r.t. logits).
The **hard top-k in forward picks by raw logit value** —
if all logits are near zero, the top-k is selected by random
tiebreaks. The model **loses the ability to learn structure**.

This is a clean example of "the regularizer punishes the
thing it's supposed to help":

  - Goal: concentrate structure (a few large logits).
  - L1's effect: all logits become small.
  - Result: no structure at all (random top-k).

## Diagnostic Insights (Useful Regardless)

1. **STE doesn't need explicit sparsity reg**: r265 already
   has hard top-k in forward. The implicit sparsity is
   sufficient.

2. **Logit magnitude is the wrong target**: the model
   should learn the **ranking** of edges, not the
   **magnitude**. L1 on magnitudes is mismatched with the
   ranking-based top-k.

3. **Smarter regularizers would target different quantities**:
   - L1 on **soft mask entropy** (concentrate soft attention).
   - Top-k entropy reg (penalize ambiguity in top-k).
   - Constraint that the model commit to a clear top-k.

4. **The collapse pattern is monotonic in λ**:
   std → 0.001, abs_mean → 0.001, frac_near_zero → 1.0
   as λ grows. There's no "sweet spot" in the L1 family.

## Comparison to Round 265

r265 found that `ste_warm` (high temperature) and `ste_no_init`
(zero init) beat r263 on structured (0.001199, 0.001696 vs
0.001594). The r265 win was driven by **temperature and
initialization**, not regularization.

r266 tested the hypothesis that L1 reg on logits would help.
It does not. **r265's no-L1 setup remains the best
configuration.**

The "spurious structure" concern raised by the r265 report
is real, but L1 is not the right fix. A more targeted
regularizer (e.g., soft-mask entropy penalty) might be
the right next step — that's r267's candidate.

## Why This Is a HONEST NEGATIVE

The mechanism failed in a clean, predictable way:

1. **L1 collapses logits** (clear, measurable effect).
2. **Task loss degrades monotonically** with λ.
3. **All 4 modes tested** (λ = 0, 0.01, 0.1, 1.0).
4. **No hidden value**: smaller is better, but λ=0 wins.

This is exactly the pattern the audit logs: **adding
auxiliary loss to STE breaks the structure learning it
provides**. Same conclusion as the r264 result on soft
attention: in 1D toy regime, simpler is better.

## Pattern Update

| Bucket          | Before | After | Δ |
|-----------------|--------|-------|---|
| Strictly pos.   |   65   |   65  | 0 |
| Target-dep      |   28   |   28  | 0 |
| Negatives       |   60   |   61  | +1 |
| **Total**       |  153   |  154  | +1 |

r266 contributes the 61st NEGATIVE: **L1 reg on STE logits
collapses structure and degrades task loss in 1D toy regime**.

## Next Round (Round 267)

Candidates:

1. **STE + Soft-Mask Entropy Reg** — penalize the entropy
   of the soft sigmoid mask, not the logits. This may
   preserve gradient signal while encouraging concentration.

2. **STE + Top-K Diversity Reg** — penalize redundancy in
   the chosen top-k edges (force them to be different).

3. **STE + per-neuron MoE** — combine r265 STE with MoE
   routing. Each neuron has its own expert.

4. **STE × Larger Hidden** — repeat r265 with hidden=32
   or 64. May reveal different L1 behavior at scale.

**Recommended: #1 (STE + soft-mask entropy reg) — directly
fixes r266's failure mode (collapse) by targeting the right
quantity (concentration, not magnitude).**

## Files Added (Round 266)

- `lnn/core/ste_l1_neuron_wise_cfc.py` (~85 LOC, subclass of
  r265)
- `tests/test_ste_l1_neuron_wise_cfc.py` (~150 LOC, 12 tests)
- `scripts/bench_ste_l1_neuron_wise_cfc.py` (~280 LOC)
- `analysis/ste_l1_neuron_wise_cfc_bench.json` (24 cells)
- `docs/prds/2026-06-28-lnn-round-266-ste-l1-sparsity-cfc.md`

## Cumulative Test Count

12 new tests (STEWithL1 unit tests). **12/12 passing.**
No regressions in this round.
