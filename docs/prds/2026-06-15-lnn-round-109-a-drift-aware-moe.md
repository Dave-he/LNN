# PRD #10-71 — Round 109: Drift-Aware Dynamic MoE (response to arXiv:2605.20678)

**Date**: 2026-06-15
**Round**: 109
**Paper**: arXiv:2605.20678 — *Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting* (Zhu, Liu, Weng, Wu — May 2026, ICML 2026)
**Status**: To implement

## Motivation

Our 91-108 audit shows:
- Static MoE routers (FAME 78, FAME 103, AuxLF 106) all collapse to H=0 or fail to improve test_mse
- Static structural fixes (QuITE 102, SETA 105, Soft MoE 107, Anchored MoE 108) succeed by changing the architecture
- **The fundamental limit**: in non-stationary time series, the **expert pool itself** may need to evolve as the data distribution shifts

Dynamic TMoE proposes exactly this: **detect distribution shifts via MMD, then add/prune experts in response**. This is the **strongest structural fix** in our audit — it changes the architecture dynamically, not just the routing.

## What Dynamic TMoE does (in 60 seconds)

Three mechanisms:
1. **MMD shift detector**: Maximum Mean Discrepancy between two consecutive windows; if MMD > threshold, drift is detected
2. **Dynamic expert pool**: experts are added (when drift detected) or pruned (when redundant) — fixed pool size is replaced by an evolving pool
3. **Temporal memory router**: uses recurrent state + an anomaly repository for context-aware expert selection, **without test-time updates**

The framework unifies architectural evolution with temporal continuity — experts persist across time, but new ones are instantiated when needed.

## Hypotheses

**H1 — Drift detection is structurally meaningful**: MMD should fire on structured_irr (regime switch) and stay quiet on sin_irr (no drift) — this is the cleanest diagnostic of the mechanism.

**H2 — test_mse on structured_irr improves**: dynamic experts should specialize on each regime better than a fixed pool.

**H3 — test_mse on sin_irr preserved or neutral**: no drift, no experts added, should match baseline.

**H4 — test_mse on random_irr preserved or neutral**: random has no real structure, drift detection may fire spuriously, but should not hurt.

## Why this should help (per audit)

- **Architectural change**: the expert pool itself evolves
- **Addresses the H=0 problem structurally**: when drift happens, new experts are added → all regimes have coverage
- **Composes with existing mechanisms** (SETA 105 shared+unique, Soft MoE 107 soft dispatch, Anchored MoE 108 structural prior)

## Architecture

```
input: x (B, T, D), history: prev window (B, T, D)
  │
  ├── MMD shift detector: MMD(x, history) → drift_score
  │   - threshold: configurable (default 0.1)
  │
  ├── ExpertPool: list of K experts (K changes over time)
  │   - on drift: add new expert initialized from KMeans of recent
  │   - on redundancy: prune least-used expert
  │   - max_pool_size: cap to prevent unbounded growth
  │
  ├── TemporalMemoryRouter: logit = RecurrentRouter([x_t, h, memory, anomaly])
  │   - memory: recurrent state across timesteps
  │   - anomaly: MMD scores history
  │
  └── Output: weighted sum of expert outputs
```

## Test plan

- MMD detector returns scalar score in [0, +inf)
- ExpertPool can add/prune experts
- TemporalMemoryRouter uses recurrent state correctly
- NaN-aware (MMD on NaN gives finite value or 0)
- Integration: full network runs end-to-end with dynamic expert pool

## Bench plan

12 cells:
- 4 conditions: `baseline_fixed` (fixed K=4), `dynamic_add_only` (add but no prune), `dynamic_full` (add+prune), `dynamic_tiny_pool` (max K=2)
- 3 datasets: sin_irr, structured_irr, random_irr
- 2 seeds × 100 epochs

Measure: test_mse, expert count over time, drift detection count, expert specialization.

## Files to create

- `lnn/core/dynamic_tmoe.py` (NEW, ~350 lines)
- `tests/test_dynamic_tmoe.py` (NEW, 15+ tests)
- `scripts/bench_dynamic_tmoe.py` (NEW, 12-cell bench)
- `docs/research/2026-06-15_dynamic_tmoe_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v35.md`
- `README.md` (new section)
- `lnn-round-109-dynamic-tmoe.md` (memory)

## Risks

1. **MMD kernel choice**: Gaussian RBF with bandwidth heuristic; may need tuning per dataset
2. **Pool growth**: without pruning, the pool could grow unbounded; cap at max_pool_size
3. **Drift detection threshold**: too sensitive → false positives on sin/random; too lax → miss real drift
4. **NaN in MMD**: if both windows have NaN, MMD is undefined; handle gracefully

## References

- arXiv:2605.20678 — Zhu, Liu, Weng, Wu (ICML 2026) *Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting*
- arXiv:2606.08896 — round 78 (FAME, fixed-pool baseline)
- arXiv:2308.00951 — round 107 (Soft MoE, complementary)
- arXiv:2605.25166 — round 108 (Anchored MoE, complementary)
- arXiv:2606.07500 — round 105 (SETA, complementary)
