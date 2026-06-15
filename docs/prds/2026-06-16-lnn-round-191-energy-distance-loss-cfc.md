# PRD #10-153 — Round 191 — Energy Distance Loss for CfC

**Date**: 2026-06-16
**Round**: 191
**Branch**: master
**Audit context (91-190)**: 46 strictly positive + 19 target-dep
+ 49 negatives = 114 mechanism classes.

## Background

Rounds 189 (BW) and 190 (SWD) both NEGATIVE — distributional
losses don't transfer to 1D synthetic.

## Goal

Test **Energy Distance** (Székely & Rizzo 2004) as simplest
distributional loss: NO sort, NO projection, NO matrix sqrt,
just pairwise L2 distances.

## Mechanism

```python
def energy_distance2(target, prediction):
    B = target.shape[0]
    target_flat = target.reshape(B, -1)
    pred_flat = prediction.reshape(B, -1)
    cross = torch.cdist(target_flat, pred_flat, p=2).mean()
    within_t = torch.cdist(target_flat, target_flat, p=2).mean()
    within_p = torch.cdist(pred_flat, pred_flat, p=2).mean()
    return 2 * cross - within_t - within_p
```

Combined: `γ · ℒ_ED + (1-γ) · ℒ_MSE`

## Hypotheses

- H1 (positive, ED helps): may help
- H2 (negative, γ=0.1 still too high): possible
- H3 (mixed, helps structured): unlikely

## Configurations (2 conds)

1. `lbps_lnkhlfft_5_3_2_mse`: pure MSE baseline
2. `lbps_lnkhlfft_5_3_2_ed`: ED + MSE (γ=0.1)

## Result

**NEGATIVE**: ED regresses on all 3 datasets.
3-round distributional loss streak all fail.

## Files

- `lnn/core/energy_distance_loss.py` (~60 lines)
- `tests/test_energy_distance_loss.py` (13 tests)
- `scripts/bench_energy_distance_loss_cfc.py` (12-cell bench)

**Why:** Test if ED is a better-behaved distributional
loss than SWD/BW.

**How to apply:** ED fails too. Don't use distributional
losses in 1D regime. Try different mechanism class next.
