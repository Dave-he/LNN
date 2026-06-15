# Round 189 — DistDF Bures-Wasserstein Loss for CfC — Research Report

**Date**: 2026-06-16
**Round**: 189
**Branch**: master
**Audit context (91-188)**: 46 strictly positive + 19 target-dep
+ 47 negatives = 112 mechanism classes.

## TL;DR

**NEGATIVE for Round 189**: Adding **Bures-Wasserstein
distributional loss** (DistDF, arXiv:2510.24574, ICLR 2026)
on top of round 187's winner (lbps_lnkhlfft_5_3_2) at γ=0.5
**regresses on all 3 datasets**: sin 6× worse, structured
2.7× worse, random 1.5× worse.

## What was tested

**DistDF joint-distribution Wasserstein alignment** as
auxiliary loss to MSE:
```python
ℒ = γ · ℒ_BW(Y, Ŷ) + (1-γ) · ℒ_MSE(Y, Ŷ)
```
where ℒ_BW is the squared **Bures-Wasserstein** distance
between Gaussian fits of target and prediction
distributions over batch:
```
BW²(μ₁, Σ₁; μ₂, Σ₂) = ||μ₁-μ₂||² + Tr(Σ₁+Σ₂-2√(Σ₁^(1/2)·Σ₂·Σ₁^(1/2)))
```

Implementation:
- `lnn/core/bures_wasserstein_loss.py` (~120 lines):
  matrix sqrt via eigendecomposition, Gaussian fit, BW²,
  combined loss
- Differentiable, closed-form (no Sinkhorn iterations)

## Bench (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| lbps_lnkhlfft_5_3_2_mse | 0.0026±0.0001 | 0.0059±0.0023 | 0.1462±0.0107 | 21434 |
| lbps_lnkhlfft_5_3_2_distdf | 0.0157±0.0131 | 0.0162±0.0043 | 0.2160±0.0304 | 21434 |

**DistDF regresses on ALL 3 datasets at γ=0.5.**

## Per-seed detail

### Baseline (MSE only, round 187)

| Dataset | Seed 0 | Seed 1 | Mean |
|---------|--------|--------|------|
| sin | 0.0027 | 0.0024 | 0.0026 |
| structured | 0.0082 | 0.0036 | 0.0059 |
| random | 0.1355 | 0.1570 | 0.1462 |

### DistDF (γ=0.5)

| Dataset | Seed 0 | Seed 1 | Mean | Δ vs MSE |
|---------|--------|--------|------|----------|
| sin | **0.0026** | 0.0288 | 0.0157 | +504% |
| structured | 0.0205 | 0.0119 | 0.0162 | +175% |
| random | 0.1856 | 0.2464 | 0.2160 | +48% |

**DistDF seed 0 on sin** matches the baseline (0.0026) —
suggesting BW loss CAN help. But **seed 1 is 12× worse**
than baseline seed 1 (0.0288 vs 0.0024), so the average
regression is high.

## Hypotheses revisited

- **H1 (positive, distributional helps)**: REJECTED in
  this regime. DistDF regresses on all 3 datasets.
- **H2 (negative, γ=0.5 too high)**: LIKELY TRUE. γ=0.5
  gives equal weight to BW and MSE, but BW has many more
  effective dimensions (32 covariance features vs 1 MSE
  feature), so it dominates gradient signal.
- **H3 (mixed, helps periodic only)**: PARTIAL. Seed 0
  on sin matches baseline. But high variance prevents
  robust improvement.

## Why DistDF regresses in our 1D toy regime

1. **BW has 32× more effective dimensions than MSE**
   - MSE loss: scalar (one number per batch element)
   - BW loss: distance between two Gaussians in 32-dim
     (T×D = 32×1)
   - BW loss dominates gradient signal at γ=0.5

2. **Covariance estimation is noisy at B=32**
   - Need B ≫ D for stable covariance estimation
   - We have B=32 and D=32 (target flattened) → ill-
     conditioned

3. **The 1D toy data has narrow distribution per timestep**
   - sin_irr: y values cluster around [-1, 1]
   - structured_irr: y values cluster around [-1, 1]
   - random_irr: y values are smooth walks in narrow range
   - Distributional alignment has nothing to align —
   predictions are already close in distribution

4. **Per-timestep MSE already captures the structure**
   - For smooth sequences, MSE per-timestep works
   - DistDF helps when distributions are far apart
   - Our model + MSE gets predictions close in mean AND
   distribution

## Why DistDF seed 0 on sin matches baseline

- This is the **best-case scenario**: BW loss didn't hurt
- Suggests lower γ (e.g., 0.1) might give a small boost
- But high seed-1 variance (0.0288) is the issue — the
  gradient from BW is destabilizing training in some
  initializations

## Pattern (46 + 19 + 48 = 113 mechanism classes)

- **46 strictly positive** (unchanged)
- **19 target-dep** (unchanged)
- **48 negatives** (UP from 47, round 189 adds 1)
- Total: **113 mechanism classes**

## Critical implementation details

1. **Matrix sqrt via `torch.linalg.eigh`** — symmetric PSD
   eigendecomposition, eigenvalues clamped to ≥ 0 for
   numerical stability
2. **Cov estimation uses `B-1` denominator** — unbiased
   covariance estimator
3. **Joint distribution over `[B, T*D]`** — flattens time
   and feature dims
4. **Random_irr y reduced to D_out=1** — was [B,T,2]
   causing shape mismatch with model output [B,T,1]; now
   `[B,T,1]` for BW compatibility
5. **γ=0.5 fixed** — could try γ=0.1, γ=0.01 in future

## Why this is a useful negative

1. **DistDF does NOT transfer to 1D synthetic data** at
   default γ=0.5
2. **Confirms pointwise loss is sufficient** when
   distribution is narrow
3. **Highlights BW gradient dominance** — large γ
   creates training instability
4. **Suggests DistDF needs larger batch + lower γ** —
   different operating regime than our 32-element batch

## Caveats

- **Single γ tested (0.5)** — lower γ may help
- **2 seeds only** — high seed variance for DistDF
  suggests more seeds needed
- **No batch size sweep** — larger B may stabilize cov
  estimation
- **Paper uses real-world data** — may benefit
  differently from synthetic

## Next ideas

1. **Try γ=0.1 or γ=0.01** — much smaller BW weight
2. **Try larger batch (B=128)** — stable cov estimation
3. **Try energy distance** instead of BW — simpler,
   kernel-free
4. **Try sliced Wasserstein** — 1D projections, cheaper
5. **Try sliced BW per-feature** — captures per-timestep
   distribution

## Files

- `lnn/core/bures_wasserstein_loss.py` (~120 lines)
- `tests/test_bures_wasserstein_loss.py` (16 tests)
- `scripts/bench_bures_wasserstein_loss_cfc.py`
  (12-cell bench)
- `results/bench_bures_wasserstein_loss_cfc.json`
- `docs/research/2026-06-16_bures_wasserstein_loss_cfc_report.md`

**Why:** Round 189 is **NEGATIVE** (DistDF γ=0.5 regresses
all 3 datasets).

**How to apply:** Don't use DistDF at γ=0.5 in 1D regime.
Try lower γ or larger batch. Audit becomes 46+19+48=113.
