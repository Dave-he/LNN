# PRD #10-66 — SDG-MoE: Signed Debate Graph Inter-Expert Deliberation (Round 104)

**Date**: 2026-06-15
**Round**: 104
**Status**: Drafted.

## 1. Why round 104

Round 103 (QuITE+MoE) revealed a **critical structural finding**: FAME has routing entropy H=0.0 in all 6 cells. The FAME router always picks the same expert regardless of input. This is a **single-expert lock-in** caused by the per-step `[x_t, h]` signal being dominated by `h` (CfC's hidden state), which the FAME router cannot disentangle.

The question for round 104 is:

> *Can inter-expert deliberation break the single-expert lock-in?*

arXiv:2605.08322 (Kulibaba et al., May 2026) — **SDG-MoE: Signed Debate Graph Mixture-of-Experts** — proposes exactly this. After top-K routing, the active experts engage in **deliberation** before their outputs are aggregated:

1. **Two learned interaction matrices**: A⁺ (support graph, reinforcing) and A⁻ (critique graph, corrective)
2. **Signed message-passing**: each active expert updates its representation based on weighted support and critique from the other active experts
3. **Disagreement-gated anchoring** (Friedkin-Johnsen style): controls deliberation strength without losing expert specialization

Reported gains: **+19.8% validation perplexity** over the strongest baseline (unsigned graph + vanilla MoE) on WikiText-103, C4, Paloma.

In our setting, the deliberation idea is most powerful for the cases where FAME is degenerate (H=0) and experts would benefit from cross-pollination. Combined with the round 103 QuITE+MoE (which gives different experts to different contexts), deliberation should make the multi-expert collaboration meaningful.

## 2. Architecture

```
Standard FAME/QuITE-MoE routing:
  x_t, h, context → router → top-K g → h_new = Σ_k g_k · expert_k(x_t, h)
                                                              ↑
                                                       (no interaction)

SDG-MoE routing:
  x_t, h, context → router → top-K g
                                       ↓
                       Active experts: E_active = [expert_k for k in top_idx]
                                       ↓
                       Signed message passing:
                         e_k ← e_k + α · A⁺ · e_active  (support update)
                         e_k ← e_k + β · A⁻ · e_active  (critique update)
                         α, β = f(disagreement)  (Friedkin-Johnsen anchoring)
                                       ↓
                       Aggregated: h_new = Σ_k g_k · e_k
```

Key design choices:
- **Signed message passing** (A⁺ for support, A⁻ for critique): allows experts to reinforce OR correct each other
- **Disagreement-gated anchoring**: deliberation strength scales with inter-expert disagreement (more disagreement → more deliberation, but bounded)
- **Per-step A⁺/A⁻**: simple learnable matrices shared across all steps; can also be made input-dependent later
- **Plays well with both FAME and QuITE+MoE**: deliberation is post-routing, so it's compatible with any router

## 3. Hypotheses

- **H1 (SDG-MoE improves test_mse over baseline on irregular TS)**: deliberation lets experts share information → -10% to -20% test_mse expected on noisy data
- **H2 (SDG-MoE expert utilization is more uniform than QuITE+MoE alone)**: deliberation creates positive feedback for under-used experts → H > QuITE+MoE alone
- **H3 (SDG-MoE target-agnostic)**: works on smooth/structured/random irregular data
- **H4 (SDG-MoE training stable)**: Friedkin-Johnsen anchoring prevents expert collapse / divergence

## 4. Plan

### 4.1 Implementation (`lnn/core/sdg_moe.py` — NEW file)

Add 4 new components:
- `signed_debate_step(expert_outs, A_pos, A_neg, disagreement_gate)` — one round of signed message passing
- `disagreement_score(expert_outs)` — pairwise cosine disagreement
- `SDGConfig` — dataclass for hyperparams (α_max, β_max, n_steps, use_anchoring)
- `SDGQuiteMoECfCCell(input_size, hidden_size, n_experts, top_k, n_tau_per_expert, tau_scales, d_context, sdg_config)` — wraps QuiteMoECfCCell with deliberation
- `SDGQuiteMoECfCNetwork(...)` — full network wrapper

Key implementation details:
- **expert_outs shape**: (B, K_active, H) — only top-K active experts
- **A_pos, A_neg shape**: (K_active, K_active) — sparse signed interaction matrices (or dense, learnable)
- **Disagreement**: pairwise cosine similarity between expert outputs, inverted
- **Anchoring**: Friedkin-Johnsen style — `e_k ← (1-λ_d) · e_k + λ_d · e_k_updated`, where λ_d = f(disagreement)

### 4.2 Tests (`tests/test_sdg_moe.py` — NEW file)

12 new tests in 4 classes:
1. `TestSignedDebateStep` (3 tests):
   - output shape
   - sign of A_pos vs A_neg produces different outputs
   - zero debate (A_pos=A_neg=0) gives identity
2. `TestDisagreementScore` (3 tests):
   - identical experts → disagreement=0
   - orthogonal experts → disagreement=1
   - shape is (B,) for (B, K, H) input
3. `TestSDGConfig` (2 tests):
   - default values
   - invalid α_max raises
4. `TestSDGQuiteMoECfCCell` (4 tests):
   - initialization
   - forward shape
   - zero debate = vanilla routing
   - positive debate improves expert differentiation (H goes up)

### 4.3 Bench (`scripts/bench_sdg_moe.py` — NEW)

30 cells:
- 2 conditions: quite_moe (baseline), quite_moe+sdg
- 3 datasets: sin_irr, structured_irr, random_irr
- 2 K settings: K=2,top_k=1; K=3,top_k=2
- 2 deliberation strengths: 0.1, 0.5
- 2 seeds, 100 epochs

For each cell measure:
- `test_mse`, `test_robust_mse`
- `expert_utilization` (entropy)
- `dead_experts`
- `grad_norm`, `training_stable`

H1: SDG-MoE lower test_mse. H2: SDG-MoE higher H. H3: SDG-MoE wins on all 3. H4: SDG-MoE training stable.

### 4.4 Decision rule

SDG-MoE is **STRICTLY POSITIVE** if H1, H2, H3 all pass. Otherwise it is **HONEST TARGET-DEPENDENT** or **HONEST NEGATIVE**.

## 5. Why this matters

- **Breaks FAME's single-expert lock-in** (round 103's H=0 finding)
- **Amplifies QuITE+MoE's diversification** (round 103's H=0.16-1.03)
- **Brings inter-expert collaboration** to our 28-layer LNN+MoE stack
- **Targets the FAME H=0 problem identified in round 103** — closes the loop

## 6. Files

- `docs/prds/2026-06-15-lnn-round-104-a-sdg-moe-deliberation.md` (this file)
- `lnn/core/sdg_moe.py` (NEW) — 5 new components
- `lnn/core/__init__.py` — export
- `tests/test_sdg_moe.py` (NEW) — 12 tests
- `scripts/bench_sdg_moe.py` (NEW) — 30-cell bench
- `results/bench_sdg_moe.json`
- `docs/research/2026-06-15_sdg_moe_deliberation_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v30.md`
- `README.md` — new section

## 7. Risk

Medium. The signed message passing is a small modification of standard MoE. The disagreement-gated anchoring is well-known from Friedkin-Johnsen 1990. The bench reuses round 103's infrastructure with a new deliberation layer.

## 8. Backlog for round 105+

1. **QuITE++ hierarchical** — combine with round 102 hierarchical variant
2. **Real PhysioNet dataset** — wire to actual data loader
3. **Per-step QuITE** — re-compute context at every step
4. **Input-dependent A⁺/A⁻** — make interaction matrices context-dependent
5. **K=20, hidden=32, paper-scale settings**
6. **DLNet (ICPR 2026) edge-battery LNN** — replication/extension
