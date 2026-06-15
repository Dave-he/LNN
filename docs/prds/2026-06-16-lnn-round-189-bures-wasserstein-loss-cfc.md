# PRD #10-151 — Round 189 — DistDF Bures-Wasserstein Loss for CfC

**Date**: 2026-06-16
**Round**: 189
**Branch**: master
**Audit context (91-188)**: 46 strictly positive + 19 target-dep
+ 47 negatives = 112 mechanism classes.

## Background

Round 188 (FFT2 mag+phase) was NEGATIVE-WITH-NUANCE — no
cond beats SOTA. Round 187 (FFT + Kh ladder) is the current
sin SOTA at 0.0026.

## Goal

Test **DistDF** distributional alignment loss (Wang et al.,
ICLR 2026, arXiv:2510.24574) as auxiliary loss to MSE.

## Mechanism

**Bures-Wasserstein loss** between Gaussian fits of target
and prediction:
```python
ℒ = γ · ℒ_BW(Y, Ŷ) + (1-γ) · ℒ_MSE(Y, Ŷ)
```
where:
```
BW² = ||μ_Y - μ_Ŷ||² + Tr(Σ_Y + Σ_Ŷ - 2·√(Σ_Y^(1/2)·Σ_Ŷ·Σ_Y^(1/2)))
```

μ and Σ estimated from [B, T·D] flattened joint distribution.

## Hypotheses

- **H1 (positive)**: distributional alignment helps
- **H2 (negative, γ=0.5 too high)**: BW dominates gradient
- **H3 (mixed)**: helps periodic only

## Configurations (2 conds)

1. `lbps_lnkhlfft_5_3_2_mse`: round 187 winner, pure MSE
2. `lbps_lnkhlfft_5_3_2_distdf`: same, BW + MSE (γ=0.5)

## Datasets

sin_irr, structured_irr, random_irr (D=2, T=32, missing_rate=0.3).

## Bench

12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs.

## Result

**NEGATIVE**: γ=0.5 regresses all 3 datasets. DistDF seed 0
on sin matches baseline (0.0026), but seed 1 is 12× worse.

## Files

- `lnn/core/bures_wasserstein_loss.py` (~120 lines)
- `tests/test_bures_wasserstein_loss.py` (16 tests)
- `scripts/bench_bures_wasserstein_loss_cfc.py` (12-cell bench)

**Why:** Test if distributional loss helps CfC on 1D
synthetic data (DistDF ICLR 2026).

**How to apply:** Don't use DistDF at γ=0.5 in 1D regime.
Try lower γ or larger batch next.
