# PRD #10-152 — Round 190 — Sliced Wasserstein Loss for CfC

**Date**: 2026-06-16
**Round**: 190
**Branch**: master
**Audit context (91-189)**: 46 strictly positive + 19 target-dep
+ 48 negatives = 113 mechanism classes.

## Background

Round 189 (Bures-Wasserstein loss) was NEGATIVE — γ=0.5
regresses all 3 datasets due to gradient dominance from
32-dim cov matrix.

## Goal

Replace BW with **Sliced Wasserstein Distance (SWD)** —
project to random 1D, compute 1D W2 = L2 of sorted values,
average. No matrix sqrt, gradient-friendly.

## Mechanism

```python
def sliced_wasserstein2(target, prediction, n_projections=20):
    B = target.shape[0]
    target_flat = target.reshape(B, -1)  # [B, T*D]
    pred_flat = prediction.reshape(B, -1)
    # Random unit directions
    theta = randn(n_projections, T*D)
    theta = theta / ||theta||
    proj_t = target_flat @ theta.T  # [B, n_projections]
    proj_p = pred_flat @ theta.T
    # 1D W2 per projection: mean((sorted_x - sorted_y)²)
    return mean over i of W2²(proj_t[:, i], proj_p[:, i])

loss = γ * SWD(target, pred) + (1-γ) * MSE(target, pred)
```

## Hypotheses

- H1 (positive): SWD helps 1D
- H2 (negative): γ=0.1 still too high
- H3 (mixed): helps structured

## Configurations (2 conds)

1. `lbps_lnkhlfft_5_3_2_mse`: pure MSE baseline
2. `lbps_lnkhlfft_5_3_2_swd`: SWD + MSE (γ=0.1, K=20)

## Datasets

sin_irr, structured_irr, random_irr.

## Result

**NEGATIVE-WITH-NUANCE**: SWD sin regresses (0.0046 vs 0.0026),
structured tied (one seed better), random tied.

## Files

- `lnn/core/sliced_wasserstein_loss.py` (~90 lines)
- `tests/test_sliced_wasserstein_loss.py` (14 tests)
- `scripts/bench_sliced_wasserstein_loss_cfc.py` (12-cell bench)

**Why:** Test if SWD is a lighter alternative to BW
that avoids gradient dominance.

**How to apply:** SWD is closer-to-baseline than BW but
still doesn't help 1D synthetic. Try γ=0.01 next.
