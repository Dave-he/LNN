# Round 190 — Sliced Wasserstein Loss for CfC — Research Report

**Date**: 2026-06-16
**Round**: 190
**Branch**: master
**Audit context (91-189)**: 46 strictly positive + 19 target-dep
+ 48 negatives = 113 mechanism classes.

## TL;DR

**NEGATIVE-WITH-NUANCE for Round 190**: Sliced Wasserstein
Distance (SWD) as auxiliary loss at γ=0.1 is much closer to
MSE baseline than round 189's Bures-Wasserstein (BW), but
**does not robustly improve any dataset**:
- sin: 0.0046 vs 0.0026 baseline (+77%, **REGRESSION both seeds**)
- structured: 0.0063 vs 0.0059 (essentially tie, one seed
  better)
- random: 0.1473 vs 0.1462 (tie)

The lighter SWD loss is a better match for the 1D toy
regime than BW loss.

## What was tested

**Sliced Wasserstein Distance** (Rabin 2012, GOTO-SWAP 2026):
project target/prediction onto random 1D directions, compute
1D W2 = L2 of sorted values, average. Drop-in auxiliary loss:
```python
ℒ = γ · ℒ_SWD(Y, Ŷ) + (1-γ) · ℒ_MSE(Y, Ŷ)
```

Implementation: `lnn/core/sliced_wasserstein_loss.py`
(~90 lines)
- 1D W2 = mean of (sorted_x_i - sorted_y_i)²
- K=20 random projections from N(0,1)/||·||
- Per-epoch seed for projection diversity
- γ=0.1 (lower than BW's 0.5)

## Bench (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| lbps_lnkhlfft_5_3_2_mse | 0.0026±0.0001 | 0.0059±0.0023 | 0.1462±0.0107 | 21434 |
| lbps_lnkhlfft_5_3_2_swd | 0.0046±0.0001 | 0.0063±0.0035 | 0.1473±0.0119 | 21434 |

## Per-seed detail

### Baseline (MSE only, round 187)

| Dataset | Seed 0 | Seed 1 | Mean |
|---------|--------|--------|------|
| sin | 0.0027 | 0.0024 | 0.0026 |
| structured | 0.0082 | 0.0036 | 0.0059 |
| random | 0.1355 | 0.1570 | 0.1462 |

### SWD (γ=0.1, K=20 projections)

| Dataset | Seed 0 | Seed 1 | Mean | Δ vs MSE |
|---------|--------|--------|------|----------|
| sin | 0.0044 | 0.0047 | 0.0046 | +77% |
| structured | 0.0097 | **0.0028** | 0.0063 | +7% |
| random | 0.1354 | 0.1592 | 0.1473 | +1% |

**Interesting**: SWD seed 1 on structured (0.0028) is
**better than MSE seed 1 (0.0036)** and close to SOTA
0.0024. But seed 0 is worse (0.0097 vs 0.0082). Mean is
essentially the same.

## Hypotheses revisited

- **H1 (positive, SWD helps)**: REJECTED. No robust
  improvement.
- **H2 (negative, γ=0.1 too high)**: PARTIAL. sin
  regression suggests γ=0.1 still too high.
- **H3 (mixed, helps structured)**: PARTIAL CONFIRMED.
  SWD seed 1 on structured hits 0.0028 (near SOTA) but
  seed 0 is worse.

## SWD vs BW (round 189 comparison)

| Loss | sin | structured | random | γ |
|------|-----|------------|--------|---|
| MSE (round 187) | 0.0026 | 0.0059 | 0.1462 | — |
| BW (round 189) | 0.0157 | 0.0162 | 0.2160 | 0.5 |
| SWD (round 190) | 0.0046 | 0.0063 | 0.1473 | 0.1 |

**SWD is much closer to MSE baseline** than BW at γ=0.5.
This confirms the analysis that **BW's 32-dim cov matrix
was dominating the gradient signal**, while SWD's 1D
projections are gradient-friendly.

## Why SWD doesn't help in 1D toy

1. **Distribution is already narrow** — sin values cluster
   in [-1, 1], structured similarly, random smooth walks
   in narrow range
2. **Per-timestep MSE captures the structure** — at T=32,
   each timestep has clear target value
3. **γ=0.1 still adds noise** — SWD loss is still
   different from MSE per-timestep
4. **Random projections sample noise** — K=20 projections
   on T·D=32 dim joint distribution still has variance

## Why SWD seed 1 on structured wins

- **Lucky projection alignment**: K=20 projections may
  happen to align with the regime-change direction
- **Distributional diversity helps regime change**:
  structured has bimodal distribution (sin vs 2·sin), SWD
  may capture this

## Pattern (46 + 19 + 49 = 114 mechanism classes)

- **46 strictly positive** (unchanged)
- **19 target-dep** (unchanged)
- **49 negatives** (UP from 48, round 190 adds 1)
- Total: **114 mechanism classes**

## Critical implementation details

1. **Per-epoch seed** for projection diversity
2. **Normalize random directions** to unit vectors
3. **Truncate to common length** in 1D W2 (assumes B equal
   sizes)
4. **No matrix sqrt** — much cheaper than BW
5. **γ=0.1 fixed** — could try γ=0.01, 0.05

## Why this is a useful negative

1. **SWD doesn't transfer to 1D synthetic** at γ=0.1
2. **Confirms pointwise loss is sufficient** when
   distribution is narrow
3. **Confirms BW failure was gradient dominance**, not
   distribution alignment itself
4. **Suggests very low γ (0.01)** might help in larger
   batch settings

## Caveats

- **Single γ tested (0.1)** — could try 0.01, 0.05
- **K=20 projections** — could try K=50, 100
- **Per-epoch seed** — could fix seed for reproducibility
- **2 seeds only** — high seed variance

## Next ideas

1. **Try γ=0.01** — much smaller SWD weight
2. **Try energy distance** — even simpler than SWD
3. **Try sliced Wasserstein per-feature** — capture
   per-timestep distribution
4. **Try smaller K** — K=5, K=10 may be enough

## Files

- `lnn/core/sliced_wasserstein_loss.py` (~90 lines)
- `tests/test_sliced_wasserstein_loss.py` (14 tests)
- `scripts/bench_sliced_wasserstein_loss_cfc.py`
  (12-cell bench)
- `results/bench_sliced_wasserstein_loss_cfc.json`
- `docs/research/2026-06-16_sliced_wasserstein_loss_cfc_report.md`

**Why:** Round 190 is **NEGATIVE-WITH-NUANCE** (no
robust improvement; sin regression; structured tied;
random tied).

**How to apply:** SWD is a closer-to-baseline failure mode
than BW. Don't use SWD as auxiliary loss in 1D regime.
Try γ=0.01 or per-feature SWD next. Audit becomes
46+19+49=114.
