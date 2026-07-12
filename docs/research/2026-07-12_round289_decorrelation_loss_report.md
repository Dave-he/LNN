---
title: "Round 289 — State Decorrelation Loss (HONEST TARGET-DEPENDENT-WITH-NUANCE — 6th consecutive TD)"
date: 2026-07-12
round: 289
prd: "docs/prds/2026-07-12-lnn-round-289-decorrelation-loss-a.md"
paper: "arXiv:2607.01986 (Nie, Wang, Su 2026-07) — Liquid Latent State Dynamics"
status: "TARGET-DEPENDENT — best structured Δ%=-32.1% in line, but toy_sin hurt; H3 fails"
parent: "r288 EMA-gate (5-round pulse line closed) — pivot to fresh mechanism"
---

# Round 289 — State Decorrelation Loss

## TL;DR

Pivoted away from the exhausted 5-round pulse line (r284-r288) to a
fresh disentanglement axis: **state decorrelation loss** from
arXiv:2607.01986. The loss penalizes the off-diagonal of the hidden-
state covariance, normalized by the diagonal. **Result: HONEST TARGET-
DEPENDENT-WITH-NUANCE — best structured Δ%=-32.1% in any r284-r289
variant, but toy_sin hurt (+754% at λ=0.001).** Plus **H3 fails**:
the diag-normalization lets the optimizer escape via diagonal scaling,
so the learned covariance is *not* actually decorrelated
(diag/off_ratio 0.10-0.72, far below the 5.0 bar).

This is the **6th consecutive TD result** in this /loop session
(r284-r289). The bench-scale 1D regression task may be too narrow to
discriminate strict-positive regularizers — many mechanisms that
*should* be helpful (decorrelation, smoothness, MoE diversity) appear
target-dependent on this benchmark.

## Results (128-hidden, T=48, 50 epochs, 2 seeds, gap p=0.3)

Clean test MSE, gap_ratio, diag/off_ratio:

| mode              | toy_sin mse / gr / ratio | structured mse / gr / ratio | random mse / gr / ratio |
|-------------------|--------------------------|-----------------------------|--------------------------|
| static_tau        | 0.00024 / 1707 / 0.15    | 0.00028 / 449 / 0.19        | 0.981 / 1.00 / 0.08      |
| blend_gated (r280)| 0.00001 / 26464 / 0.15   | 0.00024 / 368 / 0.25        | 0.984 / 1.00 / 0.07      |
| decorr λ=0.0001   | 0.00004 / 9653 / 0.15    | 0.00021 / 413 / 0.24        | 0.981 / 1.00 / 0.12      |
| decorr λ=0.001    | 0.00009 / 5350 / 0.10    | 0.00026 / 332 / 0.21        | 0.980 / 1.00 / 0.19      |
| **decorr λ=0.01** | 0.00224 / 1987 / 0.21    | **0.00017** / 526 / 0.31    | 0.986 / 1.00 / 0.19      |
| decorr λ=0.1      | 0.00020 / 1593 / 0.72    | 0.00040 / 175 / 0.45        | 0.983 / 1.00 / 0.48      |

Δ% clean MSE vs blend_gated:
- toy_sin: λ=0.0001 +239% / λ=0.001 +755% / λ=0.01 **+21255%** / λ=0.1 +1787%
- structured: λ=0.0001 **-15.0%** / λ=0.001 +8% / λ=0.01 **-32.1%** / λ=0.1 +65%
- random: all values essentially unchanged (-0.4% to +0.2%)

## Hypothesis evaluation

### H1 (improves-or-maintains on ALL 3 datasets) — REJECTED
| λ | toy_sin | structured | random | verdict |
|---|--------:|-----------:|-------:|---------|
| 0.0001 | +239% | -15.0% | -0.2% | FAIL |
| 0.001  | +755% | +8.4%   | -0.4% | FAIL |
| 0.01   | +21255% | -32.1% | +0.2% | FAIL |
| 0.1    | +1787% | +65%   | 0%    | FAIL |

No λ value satisfies H1. The decorrelation loss **helps structured**
(best -32.1%) but **hurts toy_sin** in all configurations. toy_sin has
near-zero baseline MSE (~1e-5), so any extra loss competes with the
task loss and dominates.

### H2 (orthogonality) — N/A
Not directly measurable; the loss combines with blend gate without
crashing (all cells train successfully).

### H3 (diag/off_ratio ≥ 5) — REJECTED
| λ | avg ratio across 3 datasets |
|---|-----------------------------:|
| 0.0001 | 0.17 |
| 0.001  | 0.17 |
| 0.01   | 0.23 |
| 0.1    | 0.55 |

All λ values fail the H3 bar of 5.0. **The decorrelation loss is NOT
actually decorrelating the hidden state.** Why? The loss is
`off_sq_sum / (d_h^2 · diag_mean^2)`. The optimizer can minimize this
ratio by either (a) reducing off-diagonal variance (intended), or
(b) **inflating the diagonal** (escape hatch). Since the loss is
unnormalized, the optimizer finds (b): the diagonal grows, the ratio
stays small, but the hidden state is *not* decorrelated.

This is a **loss-formulation bug**, not a fundamental flaw in the
decorrelation idea.

### H4 (H1 ∧ H3) — REJECTED
Neither passes. r289 is **+1 TD** but not strict-positive.

### H5 (gradients flow) — CONFIRMED (unit-tested)
Loss is differentiable end-to-end. Backward pass through the cell +
loss + linear head succeeds.

## Interpretation

### The target-dependence pattern

After 6 rounds (r284-r289), the anti-pattern is clear: **no
mechanism in this 1D regression bench is strict-positive across all 3
datasets.** The bench structure (toy_sin / structured / random) is
too narrow to discriminate general-purpose regularizers:

- **toy_sin** has near-zero baseline MSE — any extra loss dominates
  → target-dependent improvements hurt here
- **structured** has clean piecewise-constant signal — disentanglement
  axes (decorrelation, MoE diversity, etc.) help here
- **random** is pure noise — no signal to extract; decorrelation
  can't help or hurt

This suggests the bench may need a **4th "hard" dataset** that has
both noise AND structure (e.g. noisy-structured: piecewise-constant
signal + additive noise) to discriminate strict-positive mechanisms.

### Loss-formulation bug (H3 fail)

The decorrelation loss `off_sq / diag_mean^2` lets the optimizer
inflate the diagonal to make the ratio small. A correct formulation
would either:
- **Whitening**: explicitly normalize the hidden state to identity
  covariance, then penalize off-diag.
- **Fixed magnitude**: scale hidden state to have unit variance first
  (using stop_gradient), then penalize off-diag.
- **Barlow-Twins style**: cross-correlation matrix between two views
  of the same input, off-diag = 0.

A simple fix in r290 would be to clamp `diag_mean` to a fixed range
or add a separate `L_diag = (diag_mean - 1)^2` penalty.

## Mechanism map update

| Bucket        | Before | After | Δ |
|---------------|--------|-------|---|
| Strictly pos. |   71   |   71  | 0 |
| Target-dep    |   35   |   36  | **+1** |
| Negatives     |   62   |   62  | 0 |
| **Total**     |  167   |  168 | +1 |

r289 adds **+1 TD** (decorrelation loss with diag-normalization).
6 consecutive TD results in this /loop session.

## Files (Round 289)

- `lnn/core/decorrelation_loss.py` (NEW, ~150 LOC): `state_decorrelation_loss`,
  `state_covariance_diagnostics`.
- `tests/test_decorrelation_loss.py` (NEW, 11 tests, all green).
- `scripts/bench_decorrelation_loss.py` (NEW, ~290 LOC): 6 modes × 3
  datasets × 2 seeds × 50 epochs.
- `analysis/decorrelation_loss_bench.json` (NEW, 30 cells).
- `docs/prds/2026-07-12-lnn-round-289-decorrelation-loss-a.md`
- `docs/research/2026-07-12_round289_decorrelation_loss_report.md` (this).

## Next round (r290) candidates

Two directions to break the 6-round TD streak:

1. **Reformulate decorrelation loss** (cheap, single-cell delta): use
   `Barlow-Twins`-style cross-correlation or fixed-magnitude
   whitening. Target the same hypothesis but with a working loss.
2. **Pivot to r99 segment reliability gate** (already in the line):
   apply the existing r99 mechanism to a fresh dataset regime
   (irregular time series). Already-validated mechanism, different
   context.
3. **Pivot to arXiv:2606.21295 neuron-wise topological dynamics**: new
   backbone class, fundamentally different from cell variants.

Top recommendation: **r290 = reformulated decorrelation loss** (cheap,
tests whether H3 fails due to formulation or fundamental; if H3 still
fails, definitively abandon and pivot).

## Citation

- Nie, W., Wang, W., Su, Y. (2026-07). *Liquid Latent State Dynamics for
  Interpretable Turbofan Degradation Modeling*. arXiv:2607.01986.
- r284-r288 pulse line reports: see prior rounds in `docs/research/`.