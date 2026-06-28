---
title: "PRD #10-103 — Round 266 — STE-NeuronWiseCfCCell + L1 sparsity reg"
date: 2026-06-28
round: 266
branch: master
audit_context: "65 strictly positive + 28 target-dep + 60 negatives = 153 mechanism classes"
predecessor: "Round 265 (STENeuronWiseCfCCell, STRICT WIN) — STE for hard top-k + soft gradient"
report_anchor: "docs/research/2026-06-28_round265_ste_neuron_wise_report.md"
---

# PRD #10-103 — Round 266 — STE-NeuronWiseCfCCell + L1 sparsity reg

## Background

Round 265 (STENeuronWiseCfCCell) was a **STRICT WIN**: STE
combines r263's hard top-k forward (true sparsity) with r264's
soft backward (gradients flow) to beat both predecessors on
structured data:

  - r263_baseline: 0.001594 structured
  - r265 ste_no_init: 0.001199 structured (25% better)

The r265 report identified a key remaining question:

> "neighbor_logits become structured (std 0.084-0.482). But what
> if some of the structure is 'noise'? The model might learn
> spurious top-k selections."

This round tests whether **explicit L1 regularization on
neighbor_logits** improves over the implicit sparsity STE
already provides.

The L1 penalty:

  L_total = L_task + λ × ||neighbor_logits||_1

The L1 encourages **concentrated** structure: the model
should use the largest few logits and push the rest toward
zero. This is a soft version of "only the top-k matter".

Why this might help:
1. **Concentrated structure**: a few clear edges vs. many
   ambiguous ones.
2. **Reduced overfitting**: spurious edges in the top-k
   are penalized.
3. **Better generalization**: simpler structure may transfer
   better.

Why this might hurt:
1. **L1 is well-known to be sub-optimal** for sparsity vs.
   L0 or hard top-k.
2. **STE already provides sparsity** in the forward pass.
3. **Strong L1 may collapse** neighbor_logits to zero, which
   makes top-k selection purely random.

## Goal

Test if L1 regularization on neighbor_logits improves over
the r265 baseline. The hypothesis is that L1 helps with
spurious structure, but the right amount matters.

## Mechanism

```python
class STEWithL1(STENeuronWiseCfCCell):
    def __init__(self, ..., l1_lambda=0.0):
        super().__init__(...)
        self.l1_lambda = float(l1_lambda)

    def extra_loss(self) -> torch.Tensor:
        if self.l1_lambda <= 0:
            return torch.tensor(0.0)
        return self.l1_lambda * self.neighbor_logits.abs().mean()
```

The forward pass is unchanged from r265. The L1 penalty is
added to the training loss.

## Hypotheses (PRD #10-103)

- **H1**: STE with L1 (any λ) beats r265's no-L1 baseline on
  at least one dataset.
- **H2**: There is a **sweet spot** for λ — too small (no
  effect) or too large (collapses structure) is worse.
- **H3**: After training, the L1-penalized model has
  neighbor_logits std LARGER than no-L1 (more concentrated
  structure, even if mean decreases).
- **H4**: STE + L1 is a strict superset of r265 no-L1: with
  λ → 0, it recovers r265 behavior.

## Configurations (4 modes × 3 datasets × 2 seeds = 24 cells)

1. `ste_baseline` (r265, l1_lambda=0) — no-L1 reference
2. `ste_l1_small` (l1_lambda=0.01) — light L1
3. `ste_l1_medium` (l1_lambda=0.1) — moderate L1
4. `ste_l1_large` (l1_lambda=1.0) — strong L1

The 4 modes give an L1 sweep. The expected sweet spot is
λ=0.01 or λ=0.1 (light-to-moderate).

## Expected Pattern

If H1 holds (L1 helps): **STRICTLY POSITIVE**, the 65+1=66
SP bucket grows.

If L1 hurts (penalty too strong): **HONEST NEGATIVE** or
**TARGET-DEP**, the NEG or TD bucket grows.

Given the r265 success with ste_no_init (zero init was best),
it's possible that L1 has minimal effect because the model
already has clean structure. The most likely outcome is:

  - **ste_l1_small** ≈ r265 (small L1 has minimal effect)
  - **ste_l1_medium** slightly better or slightly worse
  - **ste_l1_large** clearly worse (collapse)

## Files to add

1. `lnn/core/ste_l1_neuron_wise_cfc.py` (~80 LOC) — subclass
   of r265 with L1 reg
2. `tests/test_ste_l1_neuron_wise_cfc.py` (~100 LOC) — 8 unit
   tests
3. `scripts/bench_ste_l1_neuron_wise_cfc.py` (~280 LOC) —
   24-cell bench
4. `lnn/core/__init__.py` — re-export STEWithL1
5. `docs/research/2026-06-28_round266_ste_l1_report.md` — bench
   report

## Bench config

- 3 datasets: toy_sin, structured, random
- hidden_size = 16
- 100 epochs, lr=1e-2, batch=16, 2 seeds
- 4 modes (above)
- Loss: MSE + λ × mean(|neighbor_logits|)
- Metrics: test_mse, neighbor_logits stats (mean, std, min, max,
  fraction near zero), L1 loss magnitude

## Why This Round

1. **Direct refinement of r265**: tests if L1 reg adds value
   on top of STE.
2. **Simple, low-risk experiment**: ~80 LOC + 100 LOC tests.
3. **Tests H1+H2+H3 (falsifiable)**: clear L1 ablation across
   4 values.
4. **Fills a gap in the r265 report**: the "spurious
   structure" concern is testable.
5. **Closes the r263→r264→r265→r266 audit chain**: 4 rounds
   on per-neuron dynamics with progressively refined structure
   learning.

## Risk Assessment

- **Risk: L1 collapse**: medium — mitigated by ablations
  across λ values.
- **Risk: L1 has no effect at small λ**: low — we report
  the effect honestly.
- **Risk: neighbor_logits diverges with no L1**: very low —
  r265 already showed stability.
- **Risk: hidden=16 too small for L1 to matter**:
  acknowledged.

## Pattern Update Expectation

After r266:
- **66 strictly positive** (if H1+H2 confirmed)
- 28 target-dep (if H2 holds but H1 doesn't)
- 60 negatives (unchanged)
- Total: **153 → 154 mechanism classes** (most likely: 65 or
  66 SP, depending on L1 effect)

## Caveats / Pre-registered Decisions

- **Pre-registered**: H1 PASS = ste_l1_medium beats
  ste_baseline on ≥ 1 dataset.
- **Pre-registered**: H2 PASS = λ=1.0 is worse than
  λ=0.01 (sweet spot exists).
- **Pre-registered**: H3 PASS = L1-penalized model has
  std > 1.5 × no-L1 std (more concentrated).
- **Pre-registered**: report final λ × std(neighbor_logits)
  for each cell.