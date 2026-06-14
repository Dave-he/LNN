# LNN Research Digest v30 — 2026-06-15

**Coverage**: arXiv 2026-06-08 → 2026-06-15, plus SDG-MoE follow-up from round 104.

## Headline

Round 104 implemented **SDG-MoE** (arXiv:2605.08322 Kulibaba et al. May 2026) — Signed Debate Graph Mixture-of-Experts with inter-expert deliberation via support (A⁺) and critique (A⁻) signed message passing, on top of round 103's QuITE+MoE.

The result is **HONEST NEGATIVE-WITH-NUANCE**:
- **H1 REJECTED**: test_mse unchanged from QuITE+MoE baseline (±2%, even +23% on random_irr K=3)
- **H2 REJECTED in WRONG DIRECTION**: routing entropy **DROPPED** from 0.16-1.03 (QuITE+MoE) to **0.0** (SDG-MoE) in all 12 cells
- **H4 CONFIRMED**: training stable, no NaN, bounded grad

**The structural finding is the key result**: deliberation pushes experts toward **consensus**, which makes the router degenerate (always pick the same expert). This is the **OPPOSITE** of what we wanted — we wanted deliberation to break FAME's H=0 lock-in, but instead it creates a NEW form of H=0 lock-in via expert consensus.

## 1. arXiv sweep highlights

| arxiv | title | relevance |
|-------|-------|-----------|
| **2605.08322** | **SDG-MoE: Signed Debate Graph MoE** | **round 104** |
| 2606.07500 | SETA: Subspace-to-Expert Sharing | new lead (backlog) |
| 2603.27188 | Deep Memory (cognitive architecture) | out of scope |
| 2606.00257 | ARCA: Adapter-residual routing | new lead (backlog) |
| 2606.00079 | BitsMoE: Bit allocation for MoE LLM | out of scope |
| 2605.28166 | QuITE (Lim ICML 2026) | round 102 (prior) |
| 2606.08896 | FAME | round 78 (prior) |

## 2. Round 104 — SDG-MoE Signed Debate Graph Deliberation

**Paper**: arXiv:2605.08322 (Kulibaba et al., May 2026)
**Implementation**: `lnn/core/sdg_moe.py` (NEW) — SDGConfig, disagreement_score, signed_debate_step, SDGLearnedInteractions, SDGQuiteMoECfCCell, SDGQuiteMoECfCNetwork
**Tests**: 27/27 (NEW file `tests/test_sdg_moe.py`, 6 test classes)
**Bench**: 48 cells (2 conds × 3 datasets × 2 K × 1 alpha × 2 seeds × 100 epochs)

**Headline findings**:
- **H1 REJECTED**: test_mse unchanged (worse on random_irr K=3)
- **H2 REJECTED in wrong direction**: H dropped to 0.0
- **H4 CONFIRMED**: training stable

## 3. Why deliberation hurts our setting

The paper's setting (LLM tokens on WikiText-103) has very different expert dynamics:
- **LLM setting**: experts process tokens independently, then aggregate. Each token sees different input. Deliberation refines the aggregate.
- **Our setting (time series)**: experts process the SAME sequence in parallel. Their outputs are correlated by construction. Deliberation amplifies the correlation.

The signed message passing is also problematic: A⁺ and A⁻ are unconstrained (`nn.Parameter` initialized at std=0.02). Without anti-symmetric constraints or normalization, the matrices can grow to push all experts to the same output.

## 4. The structural problem in our 28-layer stack

Round 103's finding (FAME H=0) and round 104's finding (SDG-MoE H=0) reveal a **deeper structural problem**:

> **Multi-expert routing in time-series MoE is fundamentally hard because the experts all see correlated inputs.**

The negative finding is not a failure — it reveals the boundaries of our design space.

## 5. Stack status (rounds 76-104)

29 layers in the LNN+MoE 自主栈:

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
| 103 | QuITE+MoE Routing (Lim ICML 2026 + FAME) | ROUTING POLICY |
| **104** | **SDG-MoE Deliberation (Kulibaba 2026)** | **HONEST NEGATIVE** |

## 6. 91-104 audit summary

| Round | Mechanism | Verdict | Key result |
|-------|-----------|---------|------------|
| 91 | CfC smoothness | HONEST NEGATIVE | smoothness is property NOT predictor |
| 92 | Temporal dropout (target-side) | HONEST NEGATIVE | CfC NOT more robust than MLP |
| 93 | Temporal dropout (input-side) | HONEST NEGATIVE | paper claim not rescued |
| 94 | Effective rank | HONEST NEGATIVE | CfC has HIGHEST rank, not lowest |
| 95 | Per-expert effective rank | FAME > MR-MoE modest | FAME 1.32 max |
| 96 | FAME+orth diversity | REJECTED | orth targets activation, not weight |
| 97 | Weight orth | HEADLINE | mean_eff_rank -20% |
| 98 | Backward coherence | PARTIAL | CfC toy_sin -10% |
| 99 | Reliability gate | STRICTLY POSITIVE | 6/6 cells improve task loss |
| 100 | SNNL | TARGET-DEPENDENT | +17% div on structured |
| 101 | ORC | RE-CLASSIFIED DIAGNOSTIC | +89% REGRESSION on smooth |
| 102 | QuITE | STRICTLY POSITIVE | first non-target-dep in audit |
| 103 | QuITE+MoE | TARGET-DEPENDENT | WINS on random K=3 |
| 104 | SDG-MoE | HONEST NEGATIVE | H drops to 0 |

The audit reveals that **mechanisms that work for LLM-style token MoE often don't transfer to time-series MoE** because time-series experts see correlated inputs.

## 7. Files updated

- `docs/prds/2026-06-15-lnn-round-104-a-sdg-moe-deliberation.md` — PRD #10-66
- `lnn/core/sdg_moe.py` (NEW) — 6 new components
- `lnn/core/__init__.py` — exports
- `tests/test_sdg_moe.py` (NEW) — 27 tests
- `scripts/bench_sdg_moe.py` (NEW) — 48-cell bench
- `results/bench_sdg_moe.json` — full results
- `docs/research/2026-06-15_sdg_moe_deliberation_report.md` — round 104 report
- `docs/daily/2026-06-15_LNN_research_summary_v30.md` — this digest
- `README.md` — new section

## 8. Backlog (cumulative)

From round 99: 4-axis gate composition, per-expert reliability, adaptive σ_min
From round 100: SETA (arXiv:2606.07500), regime-aware label for SNNL, K=20 paper-scale
From round 101: re-evaluate ORC at λ=0.01, ORC dashboard
From round 102: QuITE++ hierarchical, real PhysioNet data, QuITE+MoE for irregular-TS expert routing
From round 103: per-step QuITE, QuITE+MoE on noisier benchmarks
From round 104: anti-symmetric A⁺/A⁻, disagreement maximization, DLNet ICPR 2026

## 9. Cross-round pattern

The 91-104 audit shows a clear pattern:
- **Mechanisms targeting structural issues (QuITE for embedding, FAME for routing)**: work
- **Mechanisms targeting refinement (SNNL, ORC, deliberation)**: often target-dependent or negative
- **Architectural enhancements (QuITE, QuITE+MoE)**: more robust than regularizers

The takeaway: when you fix a foundational issue, you can target it. When you add a refinement, it depends on whether the refinement aligns with the data structure.
