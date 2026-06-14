# PRD #10-64 — QuITE Query-Based Irregular TS Embedding for CfC (Round 102)

**Date**: 2026-06-15
**Round**: 102
**Status**: Drafted.

## 1. Why round 102

arXiv:2605.28166 (Lim, ICML 2026) — *QuITE: Query-Based Irregular Time Series Embedding*. The paper identifies that **the bottleneck in IMTS modeling is the conventional embedding layer** (which assumes uniform sampling), not the backbone. It introduces a plug-and-play embedding module:
- **N learnable query tokens** that aggregate irregular observations
- **Single masked self-attention layer**
- Output: backbone-compatible latent representations
- **No artificial value generation, no backbone modification**

Reported results: **+54.7% forecasting gains, +15.8% classification gains** across 7 benchmarks and 6 MTS backbones (iTransformer, PatchTST, S-Mamba, etc.). QuITE++ (hierarchical variant) wins 20/24 forecasting settings.

This is the **most important untested domain** in our backlog (round 100+): **PhysioNet-style irregular time series**. Our CfC currently assumes uniform sampling — QuITE is the natural fix.

## 2. The QuITE architecture

```
Input: irregular observations (T, D) at times (T,) with mask (T, D)
       ↓
Value embedding: Linear(D, d_model)
Time embedding: positional encoding of (T,) — relative or learned
       ↓
Concatenate value + time embeddings → (T, d_model)
       ↓
N learnable query tokens (N, d_model)
       ↓
Masked self-attention: queries × observations
       ↓
Output: (N, d_model) — N latent tokens, feed to backbone
```

Key features:
- **Mask**: handles missing observations (NaN values)
- **Single attention layer**: O(N*T*d_model) — fast even for T=200
- **Plug-and-play**: queries → flatten → backbone input

## 3. Hypotheses

- **H1 (QuITE reduces PhysioNet-style error vs baseline CfC)**: train on irregular TS with synthetic PhysioNet-style data; +QuITE has lower test MSE than baseline
- **H2 (QuITE handles variable time gaps gracefully)**: with time-gap variance 0.1-0.5, +QuITE degrades less than baseline
- **H3 (QuITE is target-agnostic)**: works equally on smooth (sin), structured (regime switch), and random data with irregular gaps

## 4. Plan

### 4.1 Implementation (`lnn/core/quite_embedding.py` — NEW file)

Add 3 new functions:
- `QueryIrregularEmbedding(n_queries, d_model, n_heads=4, dropout=0.0)` — module that wraps a self-attention layer with learnable query tokens
- `apply_quite_embedding(observations, times, mask, query_module)` — main forward: handles irregular observations and returns latent tokens
- `quite_baseline_modes(observations, times, mask, mode='mean')` — baseline embeddings: 'mean', 'concat', 'add' for comparison

### 4.2 Tests (`tests/test_quite_embedding.py` — NEW file)

12 new tests:
1. `test_module_initialization` — n_queries, d_model
2. `test_output_shape_correct` — (B, n_queries, d_model)
3. `test_handles_variable_length` — different T per batch element
4. `test_handles_mask` — masked positions ignored
5. `test_handles_nan_values` — NaN treated as masked
6. `test_gradient_flows` — autograd check
7. `test_time_embedding_separate` — different times → different output
8. `test_baseline_mean_mode` — 'mean' aggregation baseline
9. `test_baseline_concat_mode` — 'concat' aggregation baseline
10. `test_baseline_add_mode` — 'add' aggregation baseline
11. `test_query_diversity` — different queries attend to different features
12. `test_exports` — verify exports

### 4.3 Bench (`scripts/bench_quite_irregular_ts.py` — NEW)

30 cells:
- 3 datasets: sin_irr (smooth), structured_irr (regime), random_irr (noisy)
- 5 conditions: CfC baseline, CfC+mean, CfC+concat, CfC+add, CfC+QuITE
- 2 seeds, 100 epochs
- 1 model: small CfC with hidden=16

For each cell measure:
- `task_loss` (MSE on irregular TS)
- `mask_recall` (how well does the model predict masked values)
- `time_gap_robustness` (Δ task loss as time gap variance increases)
- `latent_diversity` (variance across query tokens at output)

H1: QuITE lower task loss on irregular TS. H2: QuITE more robust to time-gap variance. H3: QuITE target-agnostic.

## 5. Expected outcomes

| dataset    | cond        | task_loss | mask_recall | latent_div |
|------------|-------------|-----------|-------------|------------|
| sin_irr    | baseline    | 0.15      | 0.50        | 1.0        |
| sin_irr    | mean        | 0.13      | 0.55        | 1.0        |
| sin_irr    | concat      | 0.12      | 0.60        | 1.0        |
| sin_irr    | add         | 0.12      | 0.60        | 1.0        |
| sin_irr    | **quite**   | **0.08**  | **0.75**    | **2.5**    |
| structured_irr | baseline | 0.50      | 0.50        | 1.0        |
| structured_irr | mean     | 0.45      | 0.55        | 1.0        |
| structured_irr | concat   | 0.42      | 0.58        | 1.0        |
| structured_irr | add      | 0.42      | 0.58        | 1.0        |
| structured_irr | **quite** | **0.32** | **0.70**   | **2.0**    |
| random_irr  | baseline    | 1.00      | 0.50        | 1.0        |
| random_irr  | mean        | 0.95      | 0.52        | 1.0        |
| random_irr  | concat      | 0.92      | 0.55        | 1.0        |
| random_irr  | add         | 0.92      | 0.55        | 1.0        |
| random_irr  | **quite**   | **0.85**  | **0.65**    | **1.5**    |

H1 ✓ if QuITE best on task_loss. H2 ✓ if QuITE most robust. H3 ✓ if QuITE best on all 3.

## 6. Why this matters

- **Fills PhysioNet gap**: most important untested domain in our backlog
- **Plug-and-play**: works with any backbone, including CfC
- **Time-aware**: handles irregular sampling, missing values, variable time gaps
- **Diagnostic value**: query diversity at output reveals what the model has learned to attend to

## 7. Files

- `docs/prds/2026-06-15-lnn-round-102-a-quite-embedding.md` (this file)
- `lnn/core/quite_embedding.py` (NEW) — 3 new functions
- `lnn/core/__init__.py` — export
- `tests/test_quite_embedding.py` (NEW) — 12 tests
- `scripts/bench_quite_irregular_ts.py` (NEW) — 30-cell bench
- `results/bench_quite_irregular_ts.json`
- `docs/research/2026-06-15_quite_irregular_ts_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v28.md`
- `README.md` — new section

## 8. Risk

Medium. The QuITE architecture is well-defined and ICML 2026 has the full reference implementation. The bench reuses round 95/97/100 infrastructure with irregular time gaps as a new dimension.

## 9. Backlog for round 103+

1. **QuITE++ hierarchical variant** — if round 102 works, try the 2-level version
2. **Real PhysioNet dataset** — wire to actual data loader
3. **QuITE + MoE** — combine with FAME for irregular-TS expert routing
4. **Compose 4-axis gates** in single FAMECfC stack (from round 99)
5. **Adaptive σ_min** — make round 99's σ_min learnable
6. **arXiv:2606.07500 SETA** — subspace-to-expert sharing for continual learning
7. **K=20, hidden=32, full recurrent training** — paper-scale
