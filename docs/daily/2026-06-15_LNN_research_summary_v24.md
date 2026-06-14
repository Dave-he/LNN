# LNN Research Digest v24 — 2026-06-15

**Coverage**: arXiv 2026-06-08 → 2026-06-15, plus 3 follow-up round-97 + round-98 papers.

## Headline

Round 98 implemented a **backward-coherence regularizer** (response to arXiv:2606.08934, Chang June 2026). It is a safe additive loss at λ=0.1 with **2/9 cells showing a real positive effect** (CfC toy_sin -10% task loss, GRU structured -19% bwd_std) but **3/9 cells showing bwd_std going UP** (LSTM/GRU random, toy_sin). The paper's strong claims do not reproduce in our 1D toy regime. Honest negative-with-nuance: safe to enable, but not a magic bullet.

## 1. arXiv sweep (12 fresh papers this week)

| arxiv | title | relevance |
|-------|-------|-----------|
| 2606.08934 | Backward Coherence and Hidden-State Stability in RNNs (Chang) | **round 98** |
| 2606.12240 | MR-MoE: Multi-Rate MoE for Time Series (follow-up) | round 77 follow-up |
| 2606.10703 | Causal Audit of MoE Routing (Zhang) | rounds 87-89 follow-up |
| 2605.28166 | QuITE: Query-based Irregular TS Embedding | **new lead** for round 99+ |
| 2606.00243 | Effective Rank in Neural Networks (Williams/Payeur/Lajoie ICML) | round 94 source |
| 2606.07670 | Temporal Smoothness in Continuous-Time RNNs (Li/Pal/Tan) | round 91 source |
| 2605.27467 | Temporal Dropout in RNNs (Thu/Oo/Supnithi) | rounds 92, 93 source |
| 2606.08896 | FAME: Forecastability-Aware MoE | round 78 source |
| 2606.12240 | MR-MoE v2: K-experts + multi-rate | round 77 follow-up |
| 2605.15403 | φ-Balancing EMA mirror-descent | round 81 source |
| 2605.06415 | MoE Ecology Number E = T·H/(O+B) | round 83 source |
| 2601.00457 | Weight-Space vs Activation-Space Orth (Kim 2026) | round 90 source |

## 2. Round 98 — Backward Coherence

**Paper**: arXiv:2606.08934 (Chang, June 2026)
**Implementation**: `lnn/core/smoothness_metrics.py::backward_coherence_loss(states, λ)`
**Tests**: 21/21 (was 14, +7 new in `TestBackwardCoherence`)
**Bench**: 72 cells (4 models × 3 datasets × 2 conditions × 3 seeds, 100 epochs)
**Lambda**: 0.1 (chosen via manual sweep; PRD's λ=0.001 is too small)

**Headline findings**:
- H1 PARTIAL — bwd_std drops in 2/9 cells, rises in 3/9
- H2 ✓ — task loss within ±5% in 8/9 cells, CfC toy_sin IMPROVES 10%
- H3 ✗ — max_grad unchanged (coherence ≠ smoothness)

**Notable cells**:
- CfC toy_sin: task 0.1498 → 0.1352 (**-10%**), bwd 0.1145 → 0.1133 (-1%)
- GRU structured: bwd 0.0515 → 0.0419 (**-19%**), task unchanged
- LSTM/GRU random: bwd goes UP 4-6% (anti-correlated with noise target)

## 3. Round 97 follow-up (in backlog)

Round 97's "both" mode (activation + weight orthogonality) is the **cleanest combination**:
- diversity_ratio preserved
- mean_eff_rank reduced 20%
- task loss within ±3%
- act_cos reduced on structured/random

Recommendation: **"both" as default for FAME stacks**.

## 4. New leads for round 99+

1. **arXiv:2605.28166 QuITE** — query-based irregular TS embedding. Different from COGENT/MR-MoE in that it uses an *attention over irregular timestamps* rather than multi-rate processing. Worth a 1-cell pilot to see if it composes with our existing irregular-TS path.
2. **arXiv:2606.08934 follow-up** — backward coherence in FAME context. Apply backward coherence to FAMECfC's per-expert hidden states (not just the concatenated output). Could provide per-expert coherence rather than whole-model coherence.
3. **arXiv:2606.12240 MR-MoE v2** — new MR-MoE variant with hierarchical rate. Round 77 implemented K-experts + softmax router; v2 adds temporal hierarchy.

## 5. Stack status (rounds 76-98)

22 layers in the LNN+MoE 自主栈:

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
| **98** | **Backward coherence (Chang 2026 quasi-reverse-martingale)** | **regularizer** |

## 6. Backward coherence vs smoothness: a comparison

| Property | Smoothness (round 91) | Backward coherence (round 98) |
|----------|------------------------|-------------------------------|
| Loss | `mean((dh/dt)²)` | `mean(||h_{t+1} - h_t||²)` |
| Target | First derivative magnitude | Discrete step size |
| CfC advantage | ✓ 2× lower max_grad | partial 1% bwd drop |
| Task impact | ✗ CfC worse (rounds 92-94) | ✓ CfC toy_sin -10% |
| λ safety | 0.001 | 0.1 |
| Computational cost | cheap | cheap |

**Key distinction**: smoothness targets the *continuous* derivative, backward coherence targets the *discrete* step. For a smooth target, the two are equivalent. For a noisy target, coherence penalizes useful high-frequency tracking.

## 7. Files updated

- `docs/prds/2026-06-15-lnn-round-98-a-backward-coherence.md` — PRD #10-60
- `lnn/core/smoothness_metrics.py` — 1 new function
- `lnn/core/__init__.py` — export
- `tests/test_smoothness_metrics.py` — 21/21 (was 14)
- `scripts/bench_cfc_backward_coherence.py` (NEW) — 72-cell bench
- `results/bench_cfc_backward_coherence.json` — full results
- `docs/research/2026-06-15_cfc_backward_coherence_report.md` — round 98 report
- `README.md` — new "Backward Coherence" section

## 8. Backlog for round 99+

1. **QuITE (arXiv:2605.28166)** — query-based irregular TS embedding (1-cell pilot)
2. **Backward coherence in FAME** — per-expert coherence on FAMECfC hidden states
3. **MR-MoE v2 (arXiv:2606.12240)** — hierarchical rate variant
4. **"Both" orth mode as default for FAME** — from round 97 follow-up
5. **K=20, hidden=32, full recurrent training** — paper-scale settings
6. **Regime-labeled task** — does FAME router pick the right expert?
7. **PhysioNet-style irregular time-series** — most important untested domain
8. **Paper-style note** combining rounds 91-98 — 8-round audit complete
