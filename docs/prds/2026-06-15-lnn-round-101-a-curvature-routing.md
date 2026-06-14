# PRD #10-63 — Curvature-Guided Routing via Ollivier-Ricci (Round 101)

**Date**: 2026-06-15
**Round**: 101
**Status**: Drafted.

## 1. Why round 101

arXiv:2603.22317 (Cao et al., March 2026) — *Geometric Mixture-of-Experts with Curvature-Guided Adaptive Routing* (GeoMoE) — uses **Ollivier-Ricci Curvature (ORC)** of the expert-feature graph as a routing regularizer. The claim is that ORC provides an "intrinsic geometric prior" that produces more interpretable and topology-consistent routing than softmax/top-K routing on raw features.

The mechanism is **fundamentally different** from all our existing routing signals:

| Router | Signal source | Round |
|--------|---------------|-------|
| Standard softmax | router logits | baseline |
| CosineRouter | cosine similarity of features | 82 |
| ForecastabilityRouter | forecastability metric | 78 |
| Ecology-gated | per-expert utilization | 84-86 |
| Causality-gated | per-expert gradient imbalance | 89 |
| Reliability-gated | per-input noise | 99 |
| **Curvature-routing** | **ORC of expert-feature graph** | **101** |

ORC captures **local graph geometry** of the expert manifold — high positive ORC means experts cluster tightly (similar features), high negative ORC means experts are spread out (tree-like). This is a **topological** signal, distinct from pairwise distance.

## 2. The Ollivier-Ricci Curvature formula

For an edge (i, j) in a k-NN graph:
```
ORC(i, j) = 1 - W_1(mu_i, mu_j) / d(x_i, x_j)
```
where:
- `mu_i` = uniform distribution over i's k-NN (including i)
- `mu_j` = uniform distribution over j's k-NN (including j)
- `W_1` = Wasserstein-1 distance (earth mover's distance)
- `d(x_i, x_j)` = Euclidean distance

Interpretation:
- `ORC ≈ 1`: neighborhoods are far apart (local tree-like structure)
- `ORC ≈ 0`: neighborhoods overlap proportionally to edge length
- `ORC < 0`: neighborhoods overlap MORE than edge length (clustered, dense)

For MoE routing, **high ORC between experts means the experts are in "different" regions of feature space** (good for diversity), while **low/negative ORC means experts overlap** (bad for diversity).

## 3. Hypotheses

- **H1 (ORC is high when experts are diverse)**: compute ORC on per-expert features → high ORC = high diversity
- **H2 (curvature-routing regularizer increases diversity)**: penalize low-ORC (encourage tree-like) → diversity_ratio goes up
- **H3 (curvature-routing is safe for task loss)**: at λ=0.001, task loss within ±10%

## 4. Plan

### 4.1 Implementation (`lnn/core/curvature.py` — NEW file)

Add 3 new functions:
- `ollivier_ricci_curvature(points, k=2)` — compute ORC for each edge in the k-NN graph of `points` (shape: (N, d)). Returns a (N, N) symmetric matrix of ORC values.
- `mean_ollivier_ricci(points, k=2)` — scalar: mean ORC over all edges.
- `curvature_routing_loss(expert_features, k=2, lambda_coeff=0.001)` — returns `λ * mean(1 - ORC)` (penalizes low ORC, i.e., encourages tree-like manifold).

The ORC computation:
1. Build k-NN graph from `points`
2. For each edge (i, j):
   a. mu_i = uniform over {i} ∪ N_k(i)
   b. mu_j = uniform over {j} ∪ N_k(j)
   c. Cost matrix C[a, b] = ||n_a - n_b||
   d. Solve optimal transport via Sinkhorn (10 iterations, default)
   e. W_1 = <C, T*>
   f. ORC(i, j) = 1 - W_1 / d(x_i, x_j)

### 4.2 Tests (`tests/test_curvature.py` — NEW file)

8 new tests:
1. `test_orc_zero_for_identical_points` — all same point → ORC = 0 (degenerate)
2. `test_orc_high_for_spread_points` — well-spread points → ORC > 0
3. `test_orc_low_for_dense_points` — clustered points → ORC < 0
4. `test_orc_symmetric` — ORC(i, j) = ORC(j, i)
5. `test_mean_orc_is_scalar` — mean returns scalar
6. `test_curvature_loss_zero_for_diverse` — high ORC → loss ≈ 0
7. `test_curvature_loss_high_for_clustered` — low ORC → loss > 0
8. `test_gradient_flows` — autograd check

### 4.3 Bench (`scripts/bench_curvature_routing.py` — NEW)

24 cells:
- 3 datasets: toy_sin, structured, random
- 4 conditions: baseline, +ORC λ=0.001, +orth λ=0.001 (round 80), +ORC+orth combined
- 2 seeds, 100 epochs
- 1 model: FAMECfC with K=4 experts

For each cell measure:
- `task_loss`
- `mean_ollivier_ricci` (the ORC of the expert manifold at end of training)
- `diversity_ratio` (round 95)
- `mean_eff_rank` (round 94)

H1: +ORC has higher mean_oric than baseline. H2: +ORC has higher diversity_ratio. H3: task loss preserved.

## 5. Expected outcomes

| condition | task_loss | mean_orc | diversity_ratio | mean_er |
|-----------|-----------|----------|------------------|----------|
| baseline  | 0.13      | 0.3      | 1.20             | 5.0     |
| +ORC      | 0.14      | **0.5**  | **1.35**         | 4.8     |
| +orth     | 0.13      | 0.4      | 1.30             | 4.5     |
| +ORC+orth | 0.14      | 0.45     | 1.32             | 4.6     |

H1 ✓ if mean_orc up. H2 ✓ if diversity up. H3 ✓ if task loss within 10%.

## 6. Why this matters

- **New routing signal**: ORC captures **topological** properties of the expert manifold that pairwise similarity misses
- **Composes with weight/activation orth**: ORC is a different axis (geometry vs. correlation)
- **Diagnostic value**: ORC can be measured as a property of any trained MoE — useful for understanding the expert manifold

## 7. Files

- `docs/prds/2026-06-15-lnn-round-101-a-curvature-routing.md` (this file)
- `lnn/core/curvature.py` (NEW) — 3 new functions
- `lnn/core/__init__.py` — export
- `tests/test_curvature.py` (NEW) — 8 tests
- `scripts/bench_curvature_routing.py` (NEW) — 24-cell bench
- `results/bench_curvature_routing.json`
- `docs/research/2026-06-15_curvature_routing_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v27.md`
- `README.md` — new section

## 8. Risk

Low. The ORC computation is well-defined and the Sinkhorn approximation is a standard tool. The bench reuses the round 95/97/100 infrastructure.

## 9. Backlog for round 102+

1. **Compose 4-axis gates** in single FAMECfC stack (from round 99)
2. **Per-expert reliability** — extend round 99 to per-expert
3. **Adaptive σ_min** — make round 99's σ_min learnable
4. **arXiv:2606.07500 SETA** — subspace-to-expert sharing
5. **K=20, hidden=32, full recurrent training** — paper-scale
6. **PhysioNet-style irregular time-series** — most important untested
