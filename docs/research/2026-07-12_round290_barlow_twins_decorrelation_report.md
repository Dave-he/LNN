---
title: "Round 290 — Barlow-Twins Decorrelation Loss (STRAIGHT FAILURE — 7-round TD streak)"
date: 2026-07-12
round: 290
prd: "docs/prds/2026-07-12-lnn-round-290-barlow-twins-decorr-a.md"
paper: "Barlow-Twins (Zbontar et al. 2021) — applied to LNN hidden state"
status: "FAIL — task loss explodes; H3 still fails; bench may be fundamentally narrow"
parent: "r289 decorrelation (TD, H3 formulation bug)"
---

# Round 290 — Barlow-Twins Decorrelation Loss

## TL;DR

Reformulated r289's decorrelation loss in **Barlow-Twins style**
(Zbontar et al. 2021) to fix the H3 loss-formulation bug. **Result:
STRAIGHT FAILURE — task loss explodes (+12184% on toy_sin at the
smallest λ, vs r289's +239%) and H3 still fails.** The BT loss is
**MUCH WORSE** than the basic r289 formulation.

**7 consecutive TD/NEGATIVE results (r284-r290) on this bench.** The
hypothesis that this bench can discriminate strict-positive
regularizers is now seriously in question. Two paths:

1. **Different data regime** — the 1D regression bench is too narrow.
   Add a 4th "hard" dataset (noisy-structured) or move to irregular
   time series.
2. **Different mechanism class** — abandon regularizers entirely;
   pivot to a new backbone (e.g. arXiv:2606.21295 neuron-wise
   topological dynamics).

## Results (128-hidden, T=48, 50 epochs, 2 seeds, gap p=0.3)

Clean test MSE, gap_ratio, bt_ratio:

| mode              | toy_sin mse / gr / ratio | structured mse / gr / ratio | random mse / gr / ratio |
|-------------------|--------------------------|-----------------------------|--------------------------|
| static_tau        | 0.00024 / 1707 / 0.76    | 0.00028 / 449 / 0.46        | 0.981 / 1.00 / 0.48      |
| blend_gated (r280)| 0.00001 / 26464 / 0.78   | 0.00024 / 368 / 0.37        | 0.984 / 1.00 / 0.39      |
| bt_a0001          | 0.00129 / 217 / 0.13     | 0.00069 / 199 / 0.55        | 1.005 / 1.00 / 0.41      |
| bt_a0005          | 0.00065 / 1333 / 0.34    | 0.00775 / 31 / 0.61         | 0.991 / 1.00 / 0.55      |
| bt_a005           | 0.00471 / 131 / 0.33     | 0.01316 / 23 / 0.80         | 0.991 / 1.00 / 0.77      |
| bt_a0005_offonly  | 0.00342 / 1272 / 0.09    | 0.01967 / 81 / 0.15         | 0.995 / 1.01 / 0.20      |

Δ% clean MSE vs blend_gated:
- toy_sin: bt_a0001 **+12184%** / bt_a0005 +6123% / bt_a005 +44936% / bt_a0005_offonly +32591%
- structured: bt_a0001 +184% / bt_a0005 +3078% / bt_a005 +5293% / bt_a0005_offonly +7963%
- random: all +0.7% to +2.1%

## Hypothesis evaluation

### H1 (improves-or-maintains on ALL 3 datasets) — REJECTED (catastrophic)
Every BT configuration makes ALL three datasets worse, not just
target-dependent. toy_sin goes from ~1e-5 (blend) to 0.001-0.005
(+12000% to +45000%). structured also explodes.

### H3 (bt_ratio ≥ 5) — REJECTED (worse than r289)
| λ | avg bt_ratio |
|---|--------------:|
| 0.001 | 0.36 |
| 0.005 | 0.50 |
| 0.05  | 0.63 |
| 0.005 (off only) | 0.15 |

The BT loss is making the cross-correlation matrix **worse**, not
better. The reason: the BT loss forces `diag(C) → 1` by amplifying the
per-feature self-prediction, which can drive individual features to
near-constant values. With constant features, `diag → 1` (perfect
prediction) but `off → 0` (no off-diagonal structure). The ratio
becomes ill-defined (high diag, low off → ratio can be anything).

### H4 (H1 ∧ H3) — REJECTED
Both fail. r290 is **+1 NEGATIVE** — actually worse than r289.

## Interpretation

### Why BT failed catastrophically

The BT loss has a **d² penalty** (where d = hidden_size / 2 = 64 here,
so 4096 elements). Even at λ=0.001 the total loss has 4096 × 0.001 =
4.1 magnitude from off-diag alone. This **dominates** the task loss
which is on the order of 1e-5 to 1e-3. The optimizer is forced to
spend all its capacity on making the BT loss small.

This is a **scale mismatch**: BT loss is in the wrong units relative to
task loss. To balance them, λ would need to be ~1e-8, but at that
scale the loss is a no-op.

### Why H3 still fails

The BT ratio measures `mean_diag / max_off_diag` of the BT
cross-correlation matrix C. For random initial state:
- C ≈ 0 everywhere (independent noise)
- ratio is ill-defined (denominator near 0)

After training with BT loss:
- Some features get pushed to constant (to make diag → 1)
- diag is now 1, off is near 0
- ratio: 1 / 0.5 = 2? Or sometimes very high

But the *desired* state is: high diag (each feature has variance) +
near-zero off (features are uncorrelated). The loss formulation
*rewards* making features constant (cheap way to get diag=1), which
destroys the state.

### The bench may be the problem

After 7 rounds (r284-r290) of mechanism variants, **none** achieved
strict-positive on the (toy_sin / structured / random) triplet. The
pattern is:
- toy_sin baseline ~1e-5 — any extra loss dominates → hurts
- random has no signal — extra loss can't help
- structured is the only discriminating dataset — but most mechanisms
  either hurt or marginally help

This suggests the bench needs a **4th "hard" dataset** that has both
structure and noise (e.g. noisy-structured: piecewise-constant signal
+ additive Gaussian noise). Such a dataset would discriminate
mechanisms that handle the noise without destroying the structure.

## Mechanism map update

| Bucket        | Before | After | Δ |
|---------------|--------|-------|---|
| Strictly pos. |   71   |   71  | 0 |
| Target-dep    |   36   |   36  | 0 |
| Negatives     |   62   |   63  | **+1** |
| **Total**     |  168   |  169 | +1 |

r290 adds **+1 NEGATIVE** (BT loss doesn't work, even at small λ).

## Files (Round 290)

- `lnn/core/decorrelation_loss.py` (EXTENDED): added
  `barlow_twins_decorrelation_loss` and `barlow_twins_covariance_diagnostics`.
- `tests/test_barlow_twins_decorrelation.py` (NEW, 11 tests, all green).
- `scripts/bench_barlow_twins_decorrelation.py` (NEW, ~300 LOC):
  6 modes × 3 datasets × 2 seeds × 50 epochs.
- `analysis/barlow_twins_decorrelation_bench.json` (NEW, 30 cells).
- `docs/prds/2026-07-12-lnn-round-290-barlow-twins-decorr-a.md`
- `docs/research/2026-07-12_round290_barlow_twins_decorrelation_report.md` (this).

## Next round (r291) decision

After 7 rounds of failure on this bench, **I recommend** one of:

1. **Add a 4th "hard" dataset** (noisy-structured) to the bench to
   give mechanisms a fair chance. Then re-test the r289 decorrelation
   loss. Single-cell-delta change.
2. **Pivot to r99 segment reliability gate on irregular TS** — apply
   an already-validated mechanism (round 99 was +1 SP) to a fresh
   data domain.
3. **Pivot to arXiv:2606.21295 neuron-wise topological dynamics** —
   new backbone class. Larger change but plausibly strict-positive.

Top recommendation: **r291 = option 1** (add noisy-structured dataset,
re-test decorrelation). This is the cheapest test of whether the bench
is the bottleneck.

## Citation

- Zbontar, J., Jing, L., Misra, I., LeCun, Y., Deny, S. (2021). *Barlow
  Twins: Self-Supervised Learning via Redundancy Reduction*. arXiv:2103.03230.
- r289 decorrelation report: `docs/research/2026-07-12_round289_decorrelation_loss_report.md`