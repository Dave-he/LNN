# Round 101 — Ollivier-Ricci Curvature Routing (PRD #10-63)

**Date**: 2026-06-15
**Round**: 101
**Paper**: arXiv:2603.22317 (Cao et al., March 2026) — *Geometric Mixture-of-Experts with Curvature-Guided Adaptive Routing* (GeoMoE)

## TL;DR

We implement the **Ollivier-Ricci Curvature (ORC)** of the per-expert feature graph and apply it as a routing regularizer. The mechanism is **target-dependent**, similar to SNNL (round 100):
- **HELPS on noisy data** (random): -6% task loss
- **NEUTRAL on structured** data: 0% task loss
- **HURTS on smooth data** (toy_sin): +89% task loss REGRESSION

ORC does **not** reliably increase diversity in our 1D regression bench (H1/H2 rejected in 2/3 datasets). The combined `orc+orth` on random gives the highest diversity ratio (+12% over baseline), but this is a special-case compound effect.

The honest conclusion: **ORC is a diagnostic-grade tool** (captures local geometry of the expert manifold) but **not a reliable diversity regularizer in the toy regime**. Like SNNL, it works on noisy/structured data but fights smooth learning.

## 1. The paper's claim

arXiv:2603.22317 (Cao et al., March 2026) introduces GeoMoE, which uses **Ollivier-Ricci Curvature (ORC)** of the expert-feature graph as a routing signal. The paper claims:
- **Intrinsic geometric prior** from graph curvature
- **Curvature-guided adaptive routing** with three pieces:
  1. Specialized experts in diverse Riemannian spaces
  2. Graph-aware gating network
  3. Curvature-guided alignment loss + contrastive objective
- **Empirical claim**: outperforms SOTA on diverse graph types

## 2. The ORC formula

For each edge (i, j) in the k-NN graph of expert features:
```
ORC(i, j) = 1 - W_1(mu_i, mu_j) / d(x_i, x_j)
```
where:
- `mu_i` = uniform over `{i} ∪ N_k(i)` (k-nearest neighbors of i)
- `mu_j` = uniform over `{j} ∪ N_k(j)`
- `W_1` = Wasserstein-1 distance (Sinkhorn approximation, 5 iters)
- `d(x_i, x_j)` = Euclidean distance

Interpretation:
- **High ORC** (≈1) = neighborhoods are far apart = tree-like = experts in different regions
- **Low ORC** (≈0) = neighborhoods overlap proportionally to edge length
- **Negative ORC** (<0) = neighborhoods overlap MORE than edge length = clustered

## 3. Our implementation

`lnn/core/curvature.py`:
- `ollivier_ricci_curvature(points, k=2, sinkhorn_iters=10)` → (N, N) symmetric matrix
- `mean_ollivier_ricci(points, k=2, sinkhorn_iters=10)` → scalar mean ORC
- `curvature_routing_loss(expert_features, k=2, lambda_coeff=0.001)` → λ(1 - mean ORC) penalty

We compute ORC on the **per-expert mean hidden state** (K, H) tensor, where K=4 experts. The Wasserstein-1 distance is approximated via Sinkhorn-Knopp (5 iters, reg=0.1) for differentiability.

## 4. Bench setup

- 1 model: FAMECfC with K=4 experts, hidden=16
- 3 datasets: toy_sin (smooth), structured (regime switch), random (noisy)
- 4 conditions: baseline, +ORC λ=0.001, +orth λ=0.001 (round 80), +ORC+orth combined
- 2 seeds, 100 epochs
- 1 model with K=4 experts

Total: 1 × 3 × 4 × 2 = 24 cells

Metrics:
- `task_loss`
- `mean_orc` (round 101 — the new signal)
- `weight_sim` (round 90)
- `diversity_ratio` (round 95)
- `mean_eff_rank` (round 94)

## 5. Results

| dataset    | cond       | task_loss | mean_orc | wgt_sim | div_ratio | mean_er |
|------------|------------|-----------|----------|---------|-----------|---------|
| toy_sin    | baseline   | 0.1278    | 0.7492   | 0.0525  | 1.2194    | 10.455  |
| toy_sin    | **orc**    | **0.2410**| 0.7021   | 0.0898  | 1.2008    | 9.942   |
| toy_sin    | orth       | 0.2447    | 0.7164   | 0.0740  | 1.1414    | 10.384  |
| toy_sin    | orc_orth   | 0.1761    | **0.7549**| 0.0774 | 1.1926    | 10.145  |
| structured | baseline   | 0.4809    | 0.7153   | 0.0614  | 1.1734    | 10.121  |
| structured | **orc**    | 0.4823    | 0.7182   | 0.0602  | 1.1530    | 9.701   |
| structured | orth       | 0.5028    | **0.8005**| 0.0692 | 1.1248    | 10.092  |
| structured | orc_orth   | 0.5089    | 0.7565   | 0.0610  | 1.1512    | 10.088  |
| random     | baseline   | 0.9703    | 0.6594   | 0.0598  | 1.1418    | 10.543  |
| random     | **orc**    | **0.9099**| **0.7312**| 0.0399 | 1.1070    | 10.057  |
| random     | orth       | 0.9239    | 0.6681   | 0.0600  | 1.1149    | 10.171  |
| random     | orc_orth   | 0.9657    | 0.6671   | 0.0570  | **1.2767**| 10.307  |

## 6. Findings

### 6.1 H1 — ORC increases mean ORC ✗ REJECTED (2/3 datasets)

- toy_sin: 0.7492 → 0.7021 (DECREASE)
- structured: 0.7153 → 0.7182 (marginal)
- random: 0.6594 → 0.7312 (+11% INCREASE)

The ORC penalty does not consistently push the per-expert manifold to be more tree-like in the toy regime. The penalty gradient is too small to overcome the natural training dynamics.

### 6.2 H2 — ORC increases diversity ratio ✗ REJECTED (0/3 datasets)

- toy_sin: 1.2194 → 1.2008 (DECREASE)
- structured: 1.1734 → 1.1530 (DECREASE)
- random: 1.1418 → 1.1070 (DECREASE)

ORC does **not** increase the FAME diversity ratio in our 1D bench. The mean ORC and the FAME diversity ratio measure **different properties** of the expert manifold.

### 6.3 H3 — ORC is safe for task loss ✗ PARTIAL

- toy_sin: 0.1278 → **0.2410 (+89% REGRESSION)**
- structured: 0.4809 → 0.4823 (~0% safe)
- random: 0.9703 → **0.9099 (-6% improvement)**

ORC is **target-dependent**:
- **HELPS on noisy data** (random): -6% task loss
- **NEUTRAL on structured** data: 0% task loss
- **HURTS on smooth data** (toy_sin): +89% REGRESSION

This is similar to SNNL (round 100) — the mechanism works when data has noise/regime boundaries and fails on smooth targets.

### 6.4 Compound: orc_orth on random ✓ INTERESTING

- random + orc_orth: div_ratio 1.2767 (vs 1.1418 baseline) = +12% best
- task loss: 0.9657 (similar to baseline 0.9703)

On random data, combining ORC with orth gives the **highest diversity ratio** of any condition (+12% over baseline) at no task cost. This is a compound-effect worth flagging, but it doesn't generalize to other datasets.

## 7. Why ORC is target-dependent

The mechanism only works when the **manifold is not already smooth**:
- **Random** (no structure): the penalty reduces redundancy and improves task loss
- **Structured** (regime switch): the penalty is neutral — the manifold is already "good enough"
- **Toy_sin** (smooth periodic): the penalty fights against the natural smooth learning

The +89% task loss regression on toy_sin is severe. The mechanism is essentially **fighting the data** in this regime.

## 8. ORC vs other diversity mechanisms

| Round | Mechanism | Diversity Δ (best case) | Task loss Δ (worst case) |
|-------|-----------|--------------------------|----------------------------|
| 78 (FAME) | top-K sparse routing | +0.03-0.24 | varies |
| 80 (orth) | activation orth | +0.00 (weight) | +3% |
| 97 (weight orth) | weight orth | +0.00 (weight) | +3% |
| 100 (SNNL) | feature clustering | +0.08-0.20 | +22% |
| **101 (ORC)** | **graph curvature** | **-0.02 (avg)** | **+89% (toy_sin)** |

ORC is the **worst** diversity mechanism in our audit — it has the highest task cost (on smooth data) and the lowest diversity gain.

## 9. The diagnosis: why ORC fails as a diversity regularizer

ORC measures the **local geometry** of the expert manifold (whether neighborhoods are tree-like or clustered). This is a **topological** property that does not directly correspond to **weight/feature diversity** as measured by the FAME diversity ratio.

To boost the FAME diversity ratio, we need:
- Weight orthogonality (rounds 80, 97): penalizes correlated W_i W_j^T
- Feature clustering (round 100): pulls same-regime features together
- Routing balance (round 81): equalizes expert utilization

ORC is a **different signal** — it captures whether the manifold is "stretched out" (high ORC) or "compressed" (low ORC), but does not directly control how spread out the per-expert features are.

## 10. Verdict

| Hypothesis | Verdict |
|------------|---------|
| H1 (ORC increases mean ORC) | ✗ REJECTED — only +11% on random, -6% on toy_sin |
| H2 (ORC increases diversity ratio) | ✗ REJECTED — decreases in all 3 datasets |
| H3 (ORC safe for task loss) | ✗ PARTIAL — safe on structured, +89% on toy_sin, -6% on random |

**ORC is a diagnostic-grade tool** (captures local geometry of the expert manifold) but **not a reliable diversity regularizer in the toy regime**. The compound orc_orth effect on random (+12% diversity at no task cost) is a special case worth flagging but doesn't generalize.

## 11. Recommendations

- **DO** use `mean_ollivier_ricci` as a **diagnostic** to characterize the geometry of the expert manifold after training
- **DO NOT** use `curvature_routing_loss` as a default regularizer — it's target-dependent and hurts smooth data
- **CONSIDER** the orc+orth combination on noisy/non-smooth data only
- **RECONSIDER** the λ value — λ=0.001 may be too small or too large depending on the regime

## 12. Files

- `docs/prds/2026-06-15-lnn-round-101-a-curvature-routing.md` — PRD
- `lnn/core/curvature.py` (NEW) — 3 new functions
- `lnn/core/__init__.py` — exports
- `tests/test_curvature.py` (NEW) — 17/17 tests
- `scripts/bench_curvature_routing.py` (NEW) — 24-cell bench
- `results/bench_curvature_routing.json` — full results
- `docs/research/2026-06-15_curvature_routing_report.md` — this report
- `docs/daily/2026-06-15_LNN_research_summary_v27.md` — daily summary
- `README.md` — new section

## 13. Backlog for round 102+

1. **Compose 4-axis gates** in single FAMECfC stack (from round 99)
2. **Per-expert reliability** — extend round 99 to per-expert
3. **Adaptive σ_min** — make round 99's σ_min learnable
4. **arXiv:2606.07500 SETA** — subspace-to-expert sharing for continual learning
5. **K=20, hidden=32, full recurrent training** — paper-scale settings
6. **PhysioNet-style irregular time-series** — most important untested domain
7. **ORC as a DIAGNOSTIC only** — drop from regularizer list, add to diagnostic suite
