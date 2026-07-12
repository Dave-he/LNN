---
title: "Round 297 — TopologicalLiquidCfCCell (HONEST NEGATIVE — sparse topology loses to dense)"
date: 2026-07-12
round: 297
prd: "docs/prds/2026-07-12-lnn-round-297-topological-cell-a.md"
paper: "arXiv:2606.21295 (Cai & Zhao 2026-06) — Topological Neural Dynamics"
status: "FAIL — all n_incoming ∈ {4,8,16,32} regress blend_gated by +5% to +21%"
parent: "r295 default promotion complete; pivot to fresh mechanism"
---

# Round 297 — TopologicalLiquidCfCCell

## TL;DR

After 12 rounds on the pulse + decorrelation lines, this round
pivots to a fresh architectural mechanism: per-neuron sparse graph-
structured recurrent connections from arXiv:2606.21295. **Result:
HONEST NEGATIVE — all 4 n_incoming configurations regress blend_gated
by +5% to +21% on Henry Hub.**

The pattern: sparser topology (smaller n_incoming) regresses MORE.
At n_incoming=4 (most sparse, 1153 params), Δ%=+16%. At n_incoming=32
(less sparse, 4737 params), Δ%=+5%. The dense blend_gated has 50049
params and serves as baseline.

The paper's claim (beating CfC/S4/Transformer on Pong BC) likely
applies to a specific setting where graph structure helps (perhaps
irregular tasks). On smooth Henry Hub time series, **dense layer-wise
recurrent connections are more expressive** than sparse graph
topology at this parameter scale.

## Results (Henry Hub, 30 epochs, 2 seeds, 6 modes × 2 seeds = 12 cells)

| mode | overall MSE | hi_vol MSE | n_params |
|---|---:|---:|---:|
| static_tau | 3.145 | 312.7 | 33537 |
| **blend_gated_default (r295)** | **2.690** | **271.4** | **50049** |
| topo_n4 | 3.121 (+16.0%) | 313.4 (+15.5%) | 1153 |
| topo_n8 | 3.260 (+21.2%) | 330.7 (+21.8%) | 1665 |
| topo_n16 | 2.875 (+6.9%) | 280.0 (+3.2%) | 2689 |
| topo_n32 | 2.832 (+5.3%) | 276.3 (+1.8%) | 4737 |

Δ% vs blend_gated_default:
- topo_n4: +16.0% / +15.5% (FAIL)
- topo_n8: +21.2% / +21.8% (FAIL, worst)
- topo_n16: +6.9% / +3.2% (FAIL)
- topo_n32: +5.3% / +1.8% (FAIL, best of topo)

## Hypothesis evaluation

### H1 (topological cell improves over blend_gated) — REJECTED
All 4 n_incoming configurations regress blend_gated. The topological
cell is NOT a strict-positive default on Henry Hub.

### H2 (different n_incoming values all reasonable) — PARTIAL
The trend is clear: more connections (higher n_incoming) → closer
to blend_gated. But all values regress. None is "reasonable" in the
sense of matching or beating dense.

### H3 (gradient flow works) — CONFIRMED (unit-tested)
All 15 unit tests pass; gradients flow to W_rec_sparse and W_in.

## Interpretation

### Why topological loses to dense

The paper's "neuron-wise topological dynamics" claim assumes that
*graph structure helps specialize neurons*. On a smooth time-series
like Henry Hub, this specialization benefit is small — most neurons
need access to most of the hidden state to model the trend + noise
dynamics. Sparse graph connections starve neurons of the information
they need.

In contrast, **dense layer-wise recurrent connections** give every
neuron full access to the entire hidden state, which is what works
on smooth time series.

### Parameter count is also a factor

The topo cells have 10-30× fewer parameters (1153-4737 vs blend's
50049). At this small parameter count, expressive capacity is
limited. A fairer comparison would be a topo cell with n_incoming=d_h
(equivalent to dense) — but that's just the dense cell.

### Why the paper's claim may still hold

The paper claims to beat CfC/S4/Transformer on **Pong BC** (a
control task with discrete actions and partial observability). On
such tasks, the graph structure may help model the structured
relationships between game state variables. Our bench is a smooth
1D time series, which is a different problem class.

**The topological approach is plausible for control tasks but not
for dense time-series regression.**

## Mechanism map update

| Bucket        | Before | After | Δ |
|---------------|--------|-------|---|
| Strictly pos. |   75   |   75  | 0 |
| Target-dep    |   36   |   36  | 0 |
| Negatives     |   64   |   65  | **+1** |
| **Total**     |  174   |  175 | +1 |

r297 adds **+1 NEGATIVE**.

## Files (Round 297)

- `lnn/core/topological_liquid_cfc.py` (NEW, ~250 LOC):
  `TopologicalLiquidCfCCell` with per-neuron sparse recurrent weights.
- `tests/test_topological_liquid_cfc.py` (NEW, 15 tests, all green).
- `scripts/bench_topological_liquid.py` (NEW, ~250 LOC):
  6 modes × 2 seeds × 30 epochs, 12 cells.
- `analysis/topological_liquid_bench.json` (NEW, 12 cells).
- `docs/prds/2026-07-12-lnn-round-297-topological-cell-a.md`
- `docs/research/2026-07-12_round297_topological_liquid_report.md` (this).

## Decision for r298

The 14-round /loop session has explored:
- 5 pulse-line rounds (r284-r288, exhausted, +5 TD)
- 7 decorrelation rounds (r289-r295, +4 SP)
- 1 regression round (r296, pass)
- 1 topological pivot (r297, +1 NEG)

Top recommendations for r298:
1. **Explore a different fresh mechanism** — e.g. arXiv:2607.01986
   decorrelation loss on irregular TS (different data regime than
   Henry Hub).
2. **Test r295 decorrelation default on irregular TS** — apply the
   successful decorrelation to a different data domain to validate
   the SP generalization.
3. **Combine r295 + r297** — add decorrelation default to a new
   variant (e.g. n_tau multi-time-scale).

Top recommendation: **r298 = option 2** — apply r295's
decorrelation default to irregular TS benchmarks to validate the SP
generalizes across data regimes.

## Citation

- Cai, B., Zhao, Y. (2026-06). *Topological Neural Dynamics: A
  Neuron-wise Framework for Sequence Modeling*. arXiv:2606.21295.