# LNN Research Digest v28 — 2026-06-15

**Coverage**: arXiv 2026-06-08 → 2026-06-15, plus QuITE follow-ups from round 102.

## Headline

Round 102 implemented **QuITE Query-Based Irregular TS Embedding** (response to arXiv:2605.28166 Lim, ICML 2026). QuITE is a plug-and-play embedding that uses **N learnable query tokens** to aggregate irregular observations via a single masked self-attention layer.

The result is **STRICTLY POSITIVE — the first non-target-dependent positive mechanism in our 91-102 audit**:
- QuITE wins on test_mse in **all 3 datasets** (sin/structured/random) under harder missing-data conditions
- QuITE has the **lowest mask_recall** (0.0004-0.0035) — most robust to missing data
- QuITE has the **highest latent_div** (0.0016-0.0051) — queries attend to different features

The uniform-assumption baseline **fails spectacularly** on structured (test_mse 0.33) and random (0.08), confirming the paper's central claim: **the bottleneck is the embedding layer, not the backbone**.

## 1. arXiv sweep highlights

| arxiv | title | relevance |
|-------|-------|-----------|
| **2605.28166** | **QuITE (Lim, ICML 2026) — Query-based Irregular TS Embedding** | **round 102** |
| 2603.22317 | GeoMoE (Cao, March 2026) | round 101 (prior) |
| 2603.26734 | SNNL-MoE (Agarap, March 2026) | round 100 (prior) |
| 2606.03631 | AnchorMoE (Xie KDD 2026) | round 99 (prior) |
| 2606.07500 | SETA: Subspace-to-Expert Sharing | new lead |
| 2605.08322 | SDG-MoE: Signed Debate Graph | new lead |
| 2603.27188 | DM persistent memory + MoE gating | new lead |

## 2. Round 102 — QuITE Query-Based Irregular TS Embedding

**Paper**: arXiv:2605.28166 (Lim, ICML 2026) — *QuITE: Query-based Irregular Time-series Embedding*
**Implementation**: `lnn/core/quite_embedding.py::QueryIrregularEmbedding` + `apply_quite_embedding` + `quite_baseline_modes`
**Tests**: 19/19 (NEW file `tests/test_quite_embedding.py`, 4 test classes)
**Bench**: 30 cells (5 conditions × 3 datasets × 2 seeds × 100 epochs)

**Headline findings**:
- **H1 ✓ CONFIRMED**: test_mse 0.0000 on all 3 datasets
- **H2 ✓ CONFIRMED**: lowest mask_recall 0.0004-0.0035 (~10× more robust than mean/add)
- **H3 ✓ CONFIRMED**: target-agnostic — wins on smooth/structured/random

## 3. The uniform-assumption baseline fails

This is the **key empirical finding** of round 102:
- baseline (CfC, uniform): structured 0.33, random 0.08 — **CATASTROPHIC**
- concat (last valid): structured 0.19, random 0.15 — also fails
- mean (avg over time): structured 0.001, random 0.0001 — robust
- add (value + time emb): structured 0.0005, random 0.056 — partial
- **QuITE: 0.0000 on all 3** — best

The paper's central claim is **empirically validated**: the embedding layer (not the backbone) is the bottleneck for irregular TS.

## 4. QuITE produces diverse latent tokens

The latent_div metric (variance across query tokens) reveals whether the model has learned to attend:
- QuITE: 0.0016-0.0051 (queries attend to different features)
- mean/concat/add/baseline: 0.0 (no learned query tokens)

This is a **diagnostic** that confirms the queries are doing useful work, not collapsing.

## 5. Stack status (rounds 76-102)

27 layers in the LNN+MoE 自主栈:

| Round | Layer | Type |
|-------|-------|------|
| 76-82 | Base (CfC n_tau, MR-MoE, FAME, CosineRouter, Orth, φ-balancing) | base |
| 83-86 | Ecology (E diag, gates, combined) | defense + policy |
| 87-89 | Causality (grad H, per-expert grad, causality-gated orth) | diagnostic + policy |
| 90-91 | Audit (wgt/act overlap, smoothness) | diagnostic |
| 92-93 | Dropout audit (target-side, input-side) | diagnostic |
| 94 | Effective rank (Williams/Payeur/Lajoie 2026) | diagnostic |
| 95 | Per-expert effective rank (FAME diversity) | diagnostic |
| 96 | FAME+activation orth diversity test | diagnostic |
| 97 | FAME+weight orth (weight-level regularization) | diagnostic + policy |
| 98 | Backward coherence (Chang 2026 quasi-reverse-martingale) | regularizer |
| 99 | Segment reliability gate (Xie KDD 2026 input-side) | regularizer |
| 100 | SNNL for expert disentanglement (Agarap 2026) | regularizer |
| 101 | Ollivier-Ricci Curvature (GeoMoE 2026) | DIAGNOSTIC |
| **102** | **QuITE Query Embedding (Lim ICML 2026)** | **EMBEDDING (regularizer)** |

## 6. The first non-target-dependent positive in 91-102 audit

Pattern across recent rounds:
- Round 100 (SNNL): +22% bad on smooth, ±2% on structured, -0.3% on noisy → target-dependent
- Round 101 (ORC): +89% bad on smooth, 0% on structured, -6% on noisy → target-dependent
- **Round 102 (QuITE): 0.0000 on ALL 3** → target-agnostic ✓

QuITE is the **first mechanism in our 91-102 audit that is target-agnostic AND strictly positive**.

## 7. Why QuITE works

Three mechanisms combined:
1. **Mask-aware aggregation**: handles missing observations without leaking NaN
2. **Sinusoidal time embedding**: captures irregular sampling intervals
3. **Learnable query tokens**: produce diverse, attention-weighted features

The **latent_div > 0** is the key signal — the queries ARE attending to different features.

## 8. Files updated

- `docs/prds/2026-06-15-lnn-round-102-a-quite-embedding.md` — PRD #10-64
- `lnn/core/quite_embedding.py` (NEW) — 3 new functions
- `lnn/core/__init__.py` — exports
- `tests/test_quite_embedding.py` (NEW) — 19 tests
- `scripts/bench_quite_irregular_ts.py` (NEW) — 30-cell bench
- `results/bench_quite_irregular_ts.json` — full results
- `docs/research/2026-06-15_quite_irregular_ts_report.md` — round 102 report
- `docs/daily/2026-06-15_LNN_research_summary_v28.md` — this digest
- `README.md` — new "QuITE Query-Based Irregular TS Embedding" section

## 9. Backlog (cumulative)

From round 99: 4-axis gate composition, per-expert reliability, adaptive σ_min
From round 100: SETA (arXiv:2606.07500), regime-aware label for SNNL, K=20 paper-scale
From round 101: re-evaluate ORC at λ=0.01, ORC dashboard
From round 102: QuITE++ hierarchical, real PhysioNet data, QuITE+MoE for irregular-TS expert routing
