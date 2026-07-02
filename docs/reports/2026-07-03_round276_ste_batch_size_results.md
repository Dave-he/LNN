# Round 276 — STE × Batch Size Sweep — Results

**PRD**: #10-113 · **Date**: 2026-07-03 (completed) · **Session**:
finished as background job during /loop (r278 session)
**Verdict**: **PRODUCTION CONFIRMED** — batch=16 is optimal on the
production-critical structured dataset.

## Context

The (τ, λ, hidden, T, d_in, density) sweep of the STE line was complete
after r275. r276 closes the final hyperparameter axis: **batch size**.
The prior session left this bench 1/45; it was completed as a background
job.

## Results (45 cells: 5 batch sizes × 3 datasets × 3 seeds, 100 epochs)

### Mean test_mse
| batch | toy_sin  | structured | random   |
|-------|---------:|-----------:|---------:|
| 4     | 0.000051 | 0.001762   | 1.071268 |
| 8     | 0.000005 | 0.000855   | 1.004578 |
| **16**| 0.000031 | **0.000171** | 1.002469 |
| 32    | 0.000026 | 0.000734   | 1.003011 |
| 64    | 0.000012 | 0.007419   | 1.003196 |

### Seed variance (std across 3 seeds)
| batch | toy_sin  | structured | random   |
|-------|---------:|-----------:|---------:|
| 4     | 0.000070 | 0.002267   | 0.079221 |
| **16**| 0.000037 | **0.000021** | 0.016894 |
| 64    | 0.000005 | 0.005292   | 0.016841 |

## Hypothesis scorecard

- **H1 (batch=16 optimal on structured)**: ✅ **CONFIRMED** — 0.000171
  is best by 4-43× over all other batch sizes. Production locked.
- **H2 (small batch 4,8 doesn't hurt structured)**: ❌ REJECTED —
  batch=4 is **10× worse** (0.001762), batch=8 is 5× worse (0.000855).
- **H3 (large batch 32,64 ≈ 16 on structured)**: ❌ REJECTED —
  batch=32 is 4× worse, batch=64 is **43× worse** (0.007419).
- **H5 (smaller batch reduces seed variance)**: ❌ REJECTED — batch=16
  has the LOWEST structured variance (0.000021); batch=4 is 100× noisier.

## Interpretation

batch=16 is a genuine sweet spot on the production-critical structured
dataset — both lowest mean error AND lowest seed variance. The
structured task's piecewise-constant segments need enough gradient
updates per epoch (256/16 = 16 updates) to resolve the segment
boundaries; too few (b64 = 4 updates/epoch) underfits badly, too many
(b4 = 64 updates/epoch) injects gradient noise that destabilises the STE
soft mask.

toy_sin and random are batch-insensitive (toy_sin is trivial at any
batch; random is unlearnable at any batch, all ≈1.0).

**Production unchanged**: batch=16 stays. The full STE hyperparameter
sweep (τ, λ, hidden, T, d_in, density, batch) is now complete —
r267-r276.

## Files
- `scripts/bench_ste_batch_size.py` (fixed summary IndexError for
  <3-dataset runs, committed in r277)
- `analysis/ste_batch_size_bench.json` (45 cells, complete)

## Pattern audit
Hyperparameter confirmation (production-locked), no class change.
Closes the STE sweep line; r277-r278 then pivoted to architectural
changes (liquid τ).
