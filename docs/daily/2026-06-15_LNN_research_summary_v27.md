# LNN Research Digest v27 — 2026-06-15

**Coverage**: arXiv 2026-06-08 → 2026-06-15, plus curvature routing follow-ups from round 101.

## Headline

Round 101 implemented **Ollivier-Ricci Curvature (ORC)** routing signal (response to arXiv:2603.22317 GeoMoE by Cao et al., March 2026). The result is **HONEST NEGATIVE-WITH-NUANCE** — ORC is target-dependent, hurting task loss on smooth data (+89% on toy_sin) and not reliably increasing diversity. The mechanism is best classified as a **diagnostic** rather than a regularizer. The compound `orc+orth` on noisy data gives a +12% diversity boost at no task cost (the only positive compound effect).

ORC joins SNNL (round 100) as the second **target-dependent mechanism** in the 91-101 audit. The pattern is consistent: graph/feature-topology signals fight smooth learning.

## 1. arXiv sweep highlights

| arxiv | title | relevance |
|-------|-------|-----------|
| **2603.22317** | **GeoMoE (Cao, March 2026) — Ollivier-Ricci Curvature routing** | **round 101** |
| 2603.26734 | SNNL-MoE (Agarap, March 2026) | round 100 (prior) |
| 2606.03631 | AnchorMoE (Xie KDD 2026) | round 99 (prior) |
| 2606.08934 | Backward Coherence (Chang, June 2026) | round 98 (prior) |
| 2606.07500 | SETA: Subspace-to-Expert Sharing | new lead |
| 2605.08322 | SDG-MoE: Signed Debate Graph | new lead |
| 2603.27188 | DM persistent memory + MoE gating | new lead |

## 2. Round 101 — Ollivier-Ricci Curvature Routing

**Paper**: arXiv:2603.22317 (Cao et al., March 2026) — *Geometric Mixture-of-Experts with Curvature-Guided Adaptive Routing*
**Implementation**: `lnn/core/curvature.py::ollivier_ricci_curvature` + `mean_ollivier_ricci` + `curvature_routing_loss`
**Tests**: 17/17 (NEW file `tests/test_curvature.py`, 3 test classes)
**Bench**: 24 cells (FAMECfC K=4 × 3 datasets × 4 conditions × 2 seeds, 100 epochs)

**Headline findings**:
- **H1 ✗ REJECTED**: mean_orc only +11% on random, -6% on toy_sin
- **H2 ✗ REJECTED**: div_ratio DECREASES in all 3 datasets
- **H3 ✗ PARTIAL**: task loss -6% on random (helps), 0% on structured (neutral), **+89% on toy_sin (REGRESSION)**

## 3. ORC is target-dependent

| dataset    | task_loss Δ | mean_orc Δ | div_ratio Δ |
|------------|-------------|------------|--------------|
| toy_sin    | **+89%**    | -6%        | -2%          |
| structured | 0%          | 0%         | -2%          |
| random     | **-6%**     | +11%       | -3%          |

The mechanism only works when the **manifold is not already smooth**. On smooth periodic data, ORC fights against natural learning.

## 4. The compound-effect finding

On random data, combining ORC with orth gives the **highest diversity ratio** of any condition:
- random baseline: div_ratio 1.1418
- random + orth: 1.1149
- random + orc: 1.1070
- **random + orc+orth: 1.2767 (+12% over baseline)**

At no task cost (0.9657 vs 0.9703 baseline). This is a **special-case compound effect** that doesn't generalize to other datasets.

## 5. ORC vs other diversity mechanisms

| Round | Mechanism | Diversity Δ (best) | Task loss Δ (worst) |
|-------|-----------|---------------------|----------------------|
| 78 (FAME) | top-K sparse routing | +0.03-0.24 | varies |
| 80 (orth) | activation orth | +0.00 (weight) | +3% |
| 97 (weight orth) | weight orth | +0.00 (weight) | +3% |
| 100 (SNNL) | feature clustering | +0.08-0.20 | +22% |
| **101 (ORC)** | **graph curvature** | **-0.02 (avg)** | **+89% (toy_sin)** |

ORC is the **worst** diversity mechanism in our audit by task cost on smooth data. The +89% regression on toy_sin is severe.

## 6. Why ORC fails as a diversity regularizer

ORC measures **local geometry** (tree-like vs clustered neighborhoods) — a **topological** property that does not directly correspond to **weight/feature diversity** as measured by the FAME diversity ratio.

The two metrics measure different properties:
- mean_orc = "is the manifold stretched out?"
- div_ratio = "are the per-expert features in different regions?"

They are related but not identical. To boost div_ratio, we need orthogonality or SNNL, not ORC.

## 7. Stack status (rounds 76-101)

26 layers in the LNN+MoE 自主栈:

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
| **101** | **Ollivier-Ricci Curvature (GeoMoE 2026)** | **DIAGNOSTIC (re-classified from regularizer)** |

## 8. Why we re-classify ORC as a diagnostic

The honest finding: ORC doesn't reliably improve task loss or diversity in the toy regime. But it DOES capture a **unique topological property** of the expert manifold that's not measured by any other metric. So:

- **DO** use `mean_ollivier_ricci` as a **diagnostic** to characterize the expert manifold
- **DO NOT** use `curvature_routing_loss` as a default regularizer
- **CONSIDER** orc+orth combination on noisy data only

## 9. Target-dependent mechanisms in our stack

Two of our recent mechanisms are target-dependent:

| Round | Mechanism | smooth data | structured | noisy data |
|-------|-----------|-------------|------------|-------------|
| 100 | SNNL | +22% bad | ±2% safe | -0.3% good |
| **101** | **ORC** | **+89% bad** | **0% safe** | **-6% good** |

Pattern: **graph/feature-topology signals fight smooth learning**. The mechanism's gradient pushes experts apart in a way that disrupts the natural smooth fit.

## 10. Files updated

- `docs/prds/2026-06-15-lnn-round-101-a-curvature-routing.md` — PRD #10-63
- `lnn/core/curvature.py` (NEW) — 3 new functions
- `lnn/core/__init__.py` — exports
- `tests/test_curvature.py` (NEW) — 17 tests
- `scripts/bench_curvature_routing.py` (NEW) — 24-cell bench
- `results/bench_curvature_routing.json` — full results
- `docs/research/2026-06-15_curvature_routing_report.md` — round 101 report
- `docs/daily/2026-06-15_LNN_research_summary_v27.md` — this digest
- `README.md` — new "Ollivier-Ricci Curvature" section

## 11. Backlog (cumulative)

From round 99: 4-axis gate composition, per-expert reliability, adaptive σ_min
From round 100: SETA (arXiv:2606.07500), regime-aware label for SNNL, K=20 paper-scale, PhysioNet
From round 101: re-evaluate ORC at λ=0.01, test ORC on graph-structured data (per arXiv:2603.22317 use case), orc diagnostic dashboard
