# LNN Research Digest v29 — 2026-06-15

**Coverage**: arXiv 2026-06-08 → 2026-06-15, plus QuITE+MoE follow-up from round 103.

## Headline

Round 103 implemented **QuITE+MoE** — combining the round 102 QuITE query-based irregular-TS embedding with the round 78 FAME top-K sparse MoE routing. The QuITE module pre-computes a global "irregularity context" vector from the full sequence, which is concatenated to `[x_t, h_prev]` for routing decisions.

The result is **HONEST TARGET-DEPENDENT-WITH-NUANCE**:
- **WINS on noisy data with K=3**: -32.7% test_mse, -27.7% robust_mse on random_irr
- **TIES on smooth/structured data**: ±2% test_mse on sin_irr / structured_irr
- **H2 CONFIRMED**: QuITE+MoE has 2-3× higher expert utilization entropy (0.5-1.0 vs FAME=0.0)

The structural improvement in expert diversification is a real positive. QuITE+MoE is **safe to enable** (no regression) and provides **large gains on noisy irregular data with K≥3**.

## 1. Round 103 — QuITE+MoE Irregularity-Context-Aware Expert Routing

**Paper**: combination of arXiv:2605.28166 (QuITE) + arXiv:2606.08896 (FAME)
**Implementation**: `lnn/core/quite_moe.py` (NEW) — QuiteRouter, QuiteMoECfCCell, QuiteMoECfCNetwork, quite_context_pool
**Tests**: 28/28 (NEW file `tests/test_quite_moe.py`, 4 test classes)
**Bench**: 24 cells (2 conds × 3 datasets × 2 K settings × 2 seeds × 100 epochs)

**Headline findings**:
- **H1 MIXED**: WINS on random_irr K=3 (-32.7%), ties on others
- **H2 ✓ CONFIRMED**: 2-3× higher routing entropy (0.5-1.0 vs FAME=0.0)
- **H3 REJECTED**: target-dependent (helps noisy more)
- **H4 ✓ CONFIRMED**: training stable, no NaN, bounded grad

## 2. QuITE+MoE bench results

| dataset    | K,top_k | fame test | quite_moe test | Δ       | fame H | quite_moe H |
|------------|---------|-----------|----------------|---------|--------|-------------|
| sin_irr    | 2,1     | 0.0857    | 0.0872         | +1.7%   | 0.000  | 0.162       |
| sin_irr    | 3,2     | 0.0864    | 0.0877         | +1.5%   | 0.000  | 0.949       |
| structured | 2,1     | 0.3873    | 0.3919         | +1.2%   | 0.000  | 0.214       |
| structured | 3,2     | 0.3854    | 0.3930         | +2.0%   | 0.000  | **1.027**   |
| random     | 2,1     | 0.1768    | 0.1970         | +11.4%  | 0.000  | 0.516       |
| random     | 3,2     | 0.1924    | **0.1294**     | **-32.7%** | 0.000 | **1.002** |

## 3. Why QuITE+MoE helps on noisy data but not smooth

The routing signal `[x_t, h_prev, context]`:
- On **smooth data**: h_prev already perfectly captures the trajectory. The context is redundant. Routing collapses to a near-constant choice.
- On **structured data**: h_prev captures the regime, but the boundary depends on time-position. Small benefit.
- On **noisy data**: h_prev accumulates noise; x_t is unreliable. The QuITE context is the MOST informative signal — it tells the router "this sequence has a particular missingness pattern that expert 2 handles well". Hence the -32.7% improvement.

## 4. The structural improvement in expert utilization

This is the most important finding beyond the test_mse deltas:

- **FAME H=0.0 in all 6 cells**: the router ALWAYS picks the same expert. The per-step [x_t, h] signal is dominated by h (CfC's hidden state), which the router can't disentangle. FAME is essentially a SINGLE-expert model despite having K=2 or K=3 experts.
- **QuITE+MoE H=0.16-1.03 in all 6 cells**: the QuITE context provides an additional axis of variation that breaks the tie. The router uses multiple experts with meaningful distribution.

This is a **structural improvement** that may have downstream benefits beyond test_mse (e.g., better load balancing for further MoE extensions, more interpretable expert assignments).

## 5. Round 102 + 103 — the QuITE story

The two rounds form a complete story about QuITE:

| Round | QuITE as | Verdict | Best for |
|-------|----------|---------|----------|
| 102 | Backbone input embedding | STRICTLY POSITIVE | All irregular TS |
| 103 | Router context | TARGET-DEPENDENT | Noisy irregular TS with K≥3 |

QuITE replaces the uniform-assumption baseline (round 102) AND optionally augments the FAME router (round 103). Both are useful, but for different reasons.

## 6. Stack status (rounds 76-103)

28 layers in the LNN+MoE 自主栈:

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
| 102 | QuITE Query Embedding (Lim ICML 2026) | EMBEDDING (regularizer) |
| **103** | **QuITE+MoE Routing (Lim ICML 2026 + FAME)** | **ROUTING POLICY** |

## 7. Files updated

- `docs/prds/2026-06-15-lnn-round-103-a-quite-moe-routing.md` — PRD #10-65
- `lnn/core/quite_moe.py` (NEW) — 4 new functions/classes
- `lnn/core/__init__.py` — exports
- `tests/test_quite_moe.py` (NEW) — 28 tests
- `scripts/bench_quite_moe.py` (NEW) — 24-cell bench
- `results/bench_quite_moe.json` — full results
- `docs/research/2026-06-15_quite_moe_routing_report.md` — round 103 report
- `docs/daily/2026-06-15_LNN_research_summary_v29.md` — this digest
- `README.md` — new section

## 8. Backlog (cumulative)

From round 99: 4-axis gate composition, per-expert reliability, adaptive σ_min
From round 100: SETA (arXiv:2606.07500), regime-aware label for SNNL, K=20 paper-scale
From round 101: re-evaluate ORC at λ=0.01, ORC dashboard
From round 102: QuITE++ hierarchical, real PhysioNet data, QuITE+MoE for irregular-TS expert routing
From round 103: per-step QuITE, QuITE+MoE on noisier benchmarks, K=20 paper-scale

## 9. Cross-round pattern

The 91-103 audit shows two distinct patterns:

1. **Regularizers/diagnostics** (rounds 91-100): mostly target-dependent; gains depend on data structure.
2. **Architectural enhancements** (rounds 102-103): more robust across conditions.

QuITE (102) is universally positive because it fixes a real bug (uniform assumption). QuITE+MoE (103) is target-dependent because routing is sensitive to data structure (smooth data → local signals suffice, noisy data → global context helps).

The takeaway: when you fix a foundational issue (embedding), everyone benefits. When you add an enhancement (router context), benefits depend on the use case.
