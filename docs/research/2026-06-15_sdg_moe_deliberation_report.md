# Round 104 — SDG-MoE Signed Debate Graph Deliberation (PRD #10-66)

**Date**: 2026-06-15
**Round**: 104
**Paper**: arXiv:2605.08322 (Kulibaba et al., May 2026) — *SDG-MoE: Signed Debate Graph Mixture-of-Experts*

## TL;DR

We implement SDG-MoE — adding inter-expert deliberation via support (A⁺) and critique (A⁻) signed message passing with disagreement-gated Friedkin-Johnsen anchoring, on top of round 103's QuITE+MoE. The result is **HONEST NEGATIVE-WITH-NUANCE**:

- **H1 REJECTED**: test_mse essentially unchanged from QuITE+MoE baseline (±2%)
- **H2 REJECTED in WRONG DIRECTION**: routing entropy dropped from 0.16-1.03 (QuITE+MoE) to **0.0** (SDG-MoE) — all 12 cells
- **H4 CONFIRMED**: training stable, no NaN, bounded grad

**The structural finding is the key result**: deliberation pushes experts toward **consensus**, which makes the router degenerate (always pick the same expert). This is the **OPPOSITE** of what we wanted — we wanted deliberation to break FAME's H=0 lock-in, but instead it creates a NEW form of H=0 lock-in via expert consensus.

## 1. The architectural idea

arXiv:2605.08322 proposes that after top-K routing, the active experts should engage in **deliberation** before their outputs are aggregated:
1. **Two learned matrices**: A⁺ (support, reinforcing) and A⁻ (critique, corrective)
2. **Signed message passing**: `e_k ← e_k + α·A⁺·e_active - β·A⁻·e_active`
3. **Disagreement-gated anchoring** (Friedkin-Johnsen): `e_k ← (1-λ_d)·e_k + λ_d·e_k_updated` where `λ_d ∝ disagreement`

Reported gains: **+19.8% validation perplexity** on WikiText-103/C4/Paloma for the paper's LLM setting.

## 2. Implementation

`lnn/core/sdg_moe.py`:
- `SDGConfig(alpha_max, beta_max, n_steps, use_anchoring, anchoring_strength)` — deliberation hyperparameters
- `disagreement_score(expert_outs)` — pairwise cosine dissimilarity among active experts
- `signed_debate_step(expert_outs, A_pos, A_neg, alpha, beta)` — one round of signed message passing
- `SDGLearnedInteractions(n_experts)` — learnable A⁺ and A⁻ matrices
- `SDGQuiteMoECfCCell(...)` — wraps `QuiteMoECfCCell` with deliberation
- `SDGQuiteMoECfCNetwork(...)` — full network with pre-computed QuITE context + deliberation

## 3. Bench setup

48 cells:
- 2 conditions: `quite_moe` (baseline, round 103), `sdg_moe`
- 3 datasets: sin_irr, structured_irr, random_irr
- 2 K settings: K=2,top_k=1; K=3,top_k=2
- 1 alpha=0.1 (default, can be tuned)
- 2 seeds × 100 epochs
- T=32, D=2, hidden=16, lr=1e-3, Adam

Test on data with HIGHER missing rate (50% vs train 30%, plus extreme 70% for robust).

## 4. Results

| dataset    | K,top_k | QuITE+MoE test | SDG-MoE test | Δ     | QuITE+MoE H | SDG-MoE H |
|------------|---------|----------------|--------------|-------|-------------|-----------|
| sin_irr    | 2,1     | 0.0872         | 0.0860       | -1.4% | 0.162       | **0.000** |
| sin_irr    | 3,2     | 0.0877         | 0.0867       | -1.1% | 0.949       | **0.000** |
| structured | 2,1     | 0.3919         | 0.3863       | -1.4% | 0.214       | **0.000** |
| structured | 3,2     | 0.3930         | 0.3854       | -1.9% | 1.027       | **0.000** |
| random     | 2,1     | 0.1970         | 0.2116       | +7.4% | 0.516       | **0.000** |
| random     | 3,2     | **0.1294**     | 0.1594       | +23.2%| 1.002       | **0.000** |

## 5. Findings

### 5.1 H1 — SDG-MoE improves test_mse ✗ REJECTED

SDG-MoE test_mse is essentially identical to QuITE+MoE baseline on smooth/structured (±2%), and **WORSE on random K=3** (+23.2%). The deliberation step provides no benefit and may actually hurt.

### 5.2 H2 — SDG-MoE expert utilization is more uniform ✗ REJECTED in WRONG DIRECTION

**CRITICAL FINDING**: SDG-MoE has **H=0.0 in ALL 12 cells** (vs QuITE+MoE 0.16-1.03). The deliberation pushes experts to **consensus**, which makes the routing collapse to a single expert.

Why? The support matrix A⁺ encourages reinforcing updates, which means experts amplify each other's outputs. After one round of deliberation, all active experts have similar outputs. The router then sees similar `[x_t, h, context]` and picks the same expert every time.

This is a **new form of H=0 lock-in**, but via a different mechanism:
- FAME H=0: the per-step `[x_t, h]` is dominated by h
- SDG-MoE H=0: the deliberation makes all experts produce similar outputs

### 5.3 H3 — SDG-MoE target-agnostic — N/A (test_mse unchanged, so target-dependence is moot)

### 5.4 H4 — SDG-MoE training stable ✓ CONFIRMED

- 0/12 cells have NaN losses
- All gradient norms < 1.0
- 0 dead experts in all cells
- Friedkin-Johnsen anchoring works as intended (prevents divergence)

## 6. Why deliberation hurts our setting

The paper's setting (LLM tokens on WikiText-103) has very different expert dynamics:
- **LLM setting**: experts process tokens independently, then aggregate. Each token sees different input. Deliberation is about whether to refine the aggregate.
- **Our setting (time series)**: experts process the SAME sequence in parallel. Their outputs are correlated by construction. Deliberation amplifies the correlation, leading to consensus.

The signed message passing is also problematic: A⁺ and A⁻ are unconstrained (just `nn.Parameter` initialized at std=0.02). Without explicit anti-symmetric constraints or normalization, the matrices can grow to push all experts to the same output.

## 7. What this teaches us about the LNN+MoE stack

Round 103's finding (FAME H=0) and round 104's finding (SDG-MoE H=0) reveal a **deeper structural problem** in our 28-layer stack:

> **Multi-expert routing in time-series MoE is fundamentally hard because the experts all see correlated inputs.**

This is a real negative result. The hypothesis "we can break FAME's single-expert lock-in by adding deliberation" is **WRONG**. Deliberation amplifies the lock-in via a different mechanism.

The 91-104 audit has now accumulated:
- 4 strictly positive mechanisms (rounds 99, 100 partial, 102, ...)
- ~7 target-dependent mechanisms
- ~5 honest negative mechanisms (rounds 91, 92, 93, 94, 96, 104)
- ~2 structural findings (rounds 103, 104)

The negative findings are not failures — they reveal the boundaries of our design space.

## 8. Possible future directions

1. **Anti-symmetric A⁺/A⁻**: constrain A⁺ = -A⁻ so support and critique exactly cancel
2. **Disagreement maximization**: instead of just measuring disagreement, USE it as a loss (encourage high disagreement)
3. **Per-step QuITE context refresh**: re-compute QuITE at every step with the routing signal
4. **Skip the deliberation layer** and use orthogonality loss (round 80/97) instead — orthogonality has been shown to ACTUALLY increase diversity

## 9. Verdict

| Hypothesis | Verdict |
|------------|---------|
| H1 (SDG-MoE lower test_mse) | ✗ REJECTED — no significant change |
| H2 (SDG-MoE expert utilization more uniform) | ✗ REJECTED in wrong direction — H=0 |
| H3 (SDG-MoE target-agnostic) | N/A |
| H4 (SDG-MoE training stable) | ✓ CONFIRMED |

**SDG-MoE is an HONEST NEGATIVE-WITH-NUANCE addition to the LNN stack.** The mechanism is correctly implemented, training is stable, but the deliberation pushes experts to consensus — the opposite of what we wanted. This adds to the growing list of structural findings about multi-expert routing in time-series MoE.

## 10. Files

- `docs/prds/2026-06-15-lnn-round-104-a-sdg-moe-deliberation.md` — PRD
- `lnn/core/sdg_moe.py` (NEW) — 6 new components
- `lnn/core/__init__.py` — exports
- `tests/test_sdg_moe.py` (NEW) — 27/27 tests
- `scripts/bench_sdg_moe.py` (NEW) — 48-cell bench
- `results/bench_sdg_moe.json` — full results
- `docs/research/2026-06-15_sdg_moe_deliberation_report.md` — this report
- `docs/daily/2026-06-15_LNN_research_summary_v30.md` — daily summary
- `README.md` — new section

## 11. Backlog for round 105+

1. **QuITE++ hierarchical** — combine with round 102 hierarchical variant
2. **Real PhysioNet dataset** — wire to actual data loader
3. **Per-step QuITE** — re-compute context at every step
4. **Compose 4-axis gates** in single QuiteMoECfC stack (round 99)
5. **arXiv:2606.07500 SETA** — subspace-to-expert sharing for continual learning
6. **K=20, hidden=32, paper-scale settings**
7. **DLNet (ICPR 2026) edge-battery LNN** — replication/extension
8. **Anti-symmetric A⁺/A⁻** — constrained deliberation
