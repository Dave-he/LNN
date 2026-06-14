# LNN Research Digest v26 — 2026-06-15

**Coverage**: arXiv 2026-06-08 → 2026-06-15, plus SNNL follow-ups from rounds 95-97.

## Headline

Round 100 implemented **Soft Nearest Neighbor Loss (SNNL)** for expert disentanglement (response to arXiv:2603.26734 Agarap & Azcarraga March 2026). SNNL is a **feature-space clustering loss** — it pulls same-regime features together, similar to a soft k-NN. The result is **strong diversity gains on multi-regime data (+17% on structured, +8% on random) but +22% task loss on smooth data**. SNNL is **target-dependent** — only useful when data has natural regime boundaries.

**SNNL is the largest diversity mechanism we've found** in our 91-100 audit — bigger than FAME top-K routing (round 78) and bigger than weight orthogonality (rounds 80, 97).

## 1. arXiv sweep highlights

| arxiv | title | relevance |
|-------|-------|-----------|
| **2603.26734** | **SNNL-MoE (Agarap, March 2026) — representation disentanglement** | **round 100** |
| 2606.03631 | AnchorMoE (Xie KDD 2026) | round 99 (prior) |
| 2606.08934 | Backward Coherence (Chang, June 2026) | round 98 (prior) |
| 2606.12240 | MR-MoE v2: Multi-Rate MoE | round 77 follow-up |
| 2606.07500 | SETA: Subspace-to-Expert Sharing | new lead |
| 2603.22317 | GeoMoE: Ollivier-Ricci Curvature | new lead |
| 2605.08322 | SDG-MoE: Signed Debate Graph | new lead |
| 2603.27188 | DM persistent memory + MoE gating | new lead |

## 2. Round 100 — SNNL for Expert Disentanglement

**Paper**: arXiv:2603.26734 (Agarap & Azcarraga, March 2026)
**Implementation**: `lnn/core/snnl.py::soft_nearest_neighbor_loss` + `expert_snnl_loss`
**Tests**: 15/15 (NEW file `tests/test_snnl.py`, 3 test classes)
**Bench**: 36 cells (FAMECfC K=4 × 3 datasets × 4 conditions × 3 seeds, 100 epochs)

**CRITICAL IMPLEMENTATION DETAIL**: with K=4 experts and top-K=1 routing, the natural label "expert index" gives 4 unique labels → no positive pairs → SNNL silently returns 0. The right interpretation: **the input's regime/class** is the label, not the expert. We use `t > 0.5` to bin each timestep into 2 classes.

**Headline findings**:
- **H1 ✓ PARTIAL**: div_ratio +17% on structured, +8% on random, +3% on toy_sin
- **H2 ✗ PARTIAL**: task loss safe on structured/random, +22% REGRESSION on toy_sin
- **H3 ✗ REJECTED**: snnl+orth combined is dominated by either alone (opposing forces)

## 3. The SNNL effect is target-dependent

SNNL only works when the **label assignment is meaningful**:
- **Structured** (regime switch at t=0.5): the `t > 0.5` label is MEANINGFUL — clustering by it is informative
- **Random** (no structure): the `t > 0.5` label is RANDOM — clustering by it is just regularization
- **Toy_sin** (smooth periodic): the `t > 0.5` label is ARTIFICIAL — forcing experts to cluster by it fights against smooth learning

**Recommendation**: enable SNNL on multi-regime/multi-task data; disable on smooth single-target data.

## 4. SNNL is the strongest diversity mechanism in the audit

| Round | Mechanism | Diversity Δ | Task loss Δ |
|-------|-----------|-------------|--------------|
| 78 (FAME) | top-K sparse routing | +0.03-0.24 | varies |
| 80 (orth) | activation orth | +0.00 (weight) | ±3% |
| 97 (weight orth) | weight orth | +0.00 (weight) | ±3% |
| **100 (SNNL)** | **feature clustering** | **+0.08 to +0.20** | **-0.3% to +22%** |

SNNL gives the **largest diversity improvement** of any mechanism tested.

## 5. The 4 distinct mechanism dimensions in our stack

SNNL adds a new axis to the 91-100 audit. We now have 4 distinct mechanism dimensions:

| Dimension | Mechanism | Round | Target |
|-----------|-----------|-------|--------|
| Weight organization | weight orthogonality | 80, 97 | W_i W_j^T |
| Routing balance | φ-balancing | 81 | g_k |
| Smoothness | backward coherence | 98 | dh/dt, dh_t |
| Input-side | reliability gate | 99 | sigma_local(x) |
| **Feature clustering** | **SNNL** | **100** | **exp(-||f_i - f_j||²/T)** |

These 5 mechanisms target **orthogonal properties** of the model. They can be combined in principle, but combining with conflicting forces (e.g. SNNL + orth) doesn't give the best of both.

## 6. Stack status (rounds 76-100)

24 layers in the LNN+MoE 自主栈:

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
| **100** | **SNNL for expert disentanglement (Agarap 2026)** | **regularizer** |

## 7. Why SNNL is a different kind of mechanism

SNNL is the **only** mechanism in our stack that operates on **feature-space distances** rather than weights, hidden state magnitudes, or input statistics:
- Round 80/97: weight matrices (W_i W_j^T)
- Round 81: routing probabilities (g_k)
- Round 91/98: hidden state derivatives (dh/dt, dh_t)
- Round 99: input statistics (sigma_local)
- **Round 100: feature distances (||f_i - f_j||²)**

This makes SNNL **compositional with weight/regularization mechanisms** but **competitive with weight orthogonality** at the per-timestep level (because both try to spread experts apart in some space).

## 8. Files updated

- `docs/prds/2026-06-15-lnn-round-100-a-snnl-expert-disentanglement.md` — PRD #10-62
- `lnn/core/snnl.py` (NEW) — 2 new functions
- `lnn/core/__init__.py` — export
- `tests/test_snnl.py` (NEW) — 15 tests
- `scripts/bench_snnl_expert_disentanglement.py` (NEW) — 36-cell bench
- `results/bench_snnl_expert_disentanglement.json` — full results
- `docs/research/2026-06-15_snnl_expert_disentanglement_report.md` — round 100 report
- `README.md` — new "Soft Nearest Neighbor Loss" section

## 9. Backlog (cumulative)

From round 97: "Both" orth mode as default for FAME
From round 98: QuITE (arXiv:2605.28166), per-expert coherence in FAME, MR-MoE v2
From round 99: 4-axis gate composition, per-expert reliability, adaptive σ_min
From round 100: SETA (arXiv:2606.07500), GeoMoE, regime-aware label for SNNL, K=20 paper-scale, PhysioNet
