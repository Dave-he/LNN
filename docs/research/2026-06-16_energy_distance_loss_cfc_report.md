# Round 191 — Energy Distance Loss for CfC — Research Report

**Date**: 2026-06-16
**Round**: 191
**Branch**: master
**Audit context (91-190)**: 46 strictly positive + 19 target-dep
+ 49 negatives = 114 mechanism classes.

## TL;DR

**NEGATIVE for Round 191**: Energy Distance (Székely &
Rizzo 2004) as auxiliary loss at γ=0.1 regresses on ALL 3
datasets: sin 2.8× worse, structured 2.6× worse, random
6% worse. **Third consecutive negative for distributional
losses** (BW r189, SWD r190, ED r191) — distributional
alignment does NOT transfer to 1D synthetic data.

## What was tested

**Energy Distance** (Székely & Rizzo):
```python
D²(F, G) = 2 E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]
```
where X, X' ~ F and Y, Y' ~ G iid.

Combined loss:
```python
ℒ = γ · ℒ_ED(Y, Ŷ) + (1-γ) · ℒ_MSE(Y, Ŷ)
```

Implementation: `lnn/core/energy_distance_loss.py`
(~60 lines)
- ED via `torch.cdist` for pairwise L2 distances
- NO sorting (unlike SWD 1D W2)
- NO random projections (unlike SWD)
- NO matrix sqrt (unlike BW)
- O(B²) compute per pair, B=32 → ~1000 pairs
- γ=0.1 (matching SWD)

## Bench (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| lbps_lnkhlfft_5_3_2_mse | 0.0026±0.0001 | 0.0059±0.0023 | 0.1462±0.0107 | 21434 |
| lbps_lnkhlfft_5_3_2_ed | 0.0072±0.0025 | 0.0156±0.0056 | 0.1554±0.0176 | 21434 |

## Per-seed detail

### Baseline (MSE)

| Dataset | Seed 0 | Seed 1 | Mean |
|---------|--------|--------|------|
| sin | 0.0027 | 0.0024 | 0.0026 |
| structured | 0.0082 | 0.0036 | 0.0059 |
| random | 0.1355 | 0.1570 | 0.1462 |

### Energy Distance (γ=0.1)

| Dataset | Seed 0 | Seed 1 | Mean | Δ vs MSE |
|---------|--------|--------|------|----------|
| sin | 0.0097 | 0.0047 | 0.0072 | +177% |
| structured | 0.0212 | 0.0100 | 0.0156 | +164% |
| random | **0.1378** | 0.1731 | 0.1554 | +6% |

ED sin seed 1 (0.0047) is interesting — close to SWD's
sin seed 1 (0.0047). ED structured seeds (0.0212, 0.0100)
are both worse than MSE baseline (0.0082, 0.0036).

## 3-round distributional loss comparison (r189-r191)

| Loss | sin | structured | random | γ |
|------|-----|------------|--------|---|
| MSE | 0.0026 | 0.0059 | 0.1462 | — |
| BW (r189) | 0.0157 (+504%) | 0.0162 (+175%) | 0.2160 (+48%) | 0.5 |
| SWD (r190) | 0.0046 (+77%) | 0.0063 (+7%) | 0.1473 (+1%) | 0.1 |
| ED (r191) | 0.0072 (+177%) | 0.0156 (+164%) | 0.1554 (+6%) | 0.1 |

**Order of damage**: BW > ED > SWD > MSE.
**SWD is least harmful** despite being more complex than
ED. ED's pairwise distances capture more "drift" away from
target distribution than SWD's sorted projections.

## Hypotheses revisited

- **H1 (positive, ED helps)**: REJECTED. ED regresses all 3.
- **H2 (negative, γ=0.1 still too high)**: PARTIAL CONFIRMED.
  ED has 32-dim effective dims (T*D=32) via cdist, larger
  gradient than expected.
- **H3 (mixed)**: REJECTED. ED doesn't help structured.

## Why ED regresses MORE than SWD

1. **ED has effective dim = 32** (full T*D via cdist L2
   norm), larger than SWD's 1D projections
2. **Pairwise distances add complexity** — SWD's sorted
   1D is more constrained
3. **ED encourages mean alignment** (cross term) AND
   within-distribution variance (within terms) — both
   compete with MSE

## Pattern (46 + 19 + 50 = 115 mechanism classes)

- **46 strictly positive** (unchanged)
- **19 target-dep** (unchanged)
- **50 negatives** (UP from 49, round 191 adds 1)
- Total: **115 mechanism classes**

## Critical implementation details

1. **ED via `torch.cdist(p=2)`** — exact pairwise L2
2. **ED² = 2·cross - within_t - within_p** — biased
   estimator with 1/B² correction
3. **γ=0.1 fixed** — same as SWD for fair comparison
4. **No projections or sorts** — simplest of the three
   distributional losses

## Why this is a useful negative

1. **Distributional losses don't help 1D synthetic** — 3
   rounds of evidence (BW, SWD, ED)
2. **Per-timestep MSE is the right objective** for narrow
   distribution time series
3. **ED is the simplest yet still fails** — confirms the
   fundamental issue is distribution alignment itself, not
   implementation complexity
4. **Suggests next round should try a completely different
   mechanism class** (not loss function)

## Caveats

- **γ=0.1 only** — could try γ=0.01 (1% ED weight)
- **ED² is biased estimator** — small batch amplifies
- **No comparison with kernel MMD** — could try that

## Next ideas (different mechanism class, NOT loss)

1. **Input noise augmentation** — Gaussian noise during
   training
2. **Hidden state regularization** — penalize ||h||²
3. **Different time constants** — per-cell learnable τ
4. **Multi-head CfC** — parallel CfC cells
5. **Conv preprocessing** — 1D causal conv before CfC

## Files

- `lnn/core/energy_distance_loss.py` (~60 lines)
- `tests/test_energy_distance_loss.py` (13 tests)
- `scripts/bench_energy_distance_loss_cfc.py` (12-cell bench)
- `results/bench_energy_distance_loss_cfc.json`
- `docs/research/2026-06-16_energy_distance_loss_cfc_report.md`

**Why:** Round 191 is **NEGATIVE** (ED regresses all 3
datasets). 3-round distributional loss streak
(BW→SWD→ED) all fail.

**How to apply:** Don't use distributional losses in 1D
regime. Try completely different mechanism class next.
Audit becomes 46+19+50=115.
