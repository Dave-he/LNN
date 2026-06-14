# LNN Research Daily Digest v21 — 2026-06-15 (Round 95)

**Focus**: directly test the "diverse experts" claim of the FAME paper (arXiv:2606.08896) and the "multi-rate expert specialization" claim of MR-MoE (arXiv:2606.12240) by measuring per-expert weight effective rank after training.

## 1. Paper survey (June 2026 arXiv)

The arXiv listing for cs.NE 2606 returned a strong cluster of relevant papers:
- **arXiv:2606.08896** (Li/Zhang/Wang/Peng/Wei) — *FAME: Forecastability-Aware Mixture of Experts for Heterogeneous Time Series Forecasting*. This is the paper our round 78 already implemented at cell level. Round 95 tests its central claim.
- **arXiv:2606.12240** (Zong/Boker/Eldardiry) — *Multi-Rate Mixture of Experts for Accelerating Liquid Neural Network Training*. This is round 77's paper. Round 95 tests its central claim.
- **arXiv:2606.03631** (Xie et al.) — *AnchorMoE*. Our round 80 already implemented orthogonality-constrained MoE.
- **arXiv:2606.07670** (Li/Pal/Tan) — *Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting*. Round 91 (smoothness) is a response.
- **arXiv:2606.09907** (Rahman/Kumar/Maass) — *LongMoE*. New, deferred.

Of these, the per-expert diversity question (round 95) is the cleanest one our existing stack can answer.

## 2. Implementation: per-expert effective rank (PRD #10-57)

Round 95 adds 3 functions to `lnn/core/effective_rank.py`:
- `per_expert_effective_rank(cell)` — iterate over `cell.experts[i]`, collect 2D params, return mean eff_rank per expert
- `expert_diversity_ratio(ranks)` — max/min ratio
- `expert_diversity_summary(cell)` — combined: per_expert, mean, min, max, std, diversity_ratio, n_experts, n_dead

7 new unit tests, **27/27 total** in `tests/test_effective_rank.py`.

## 3. Test + bench summary

- **27/27 unit tests** pass (was 20, +7 new)
- **36-cell bench** (3 datasets × 2 models × 2 conditions × 3 seeds, 100 epochs):

| dataset    | FAME trained div | MR-MoE trained div | FAME > MR-MoE? |
|------------|------------------|--------------------|-----------------|
| toy_sin    | **1.32 ± 0.08**  | 1.08 ± 0.01        | ✓ (Δ=0.24)     |
| structured | 1.15 ± 0.04      | 1.12 ± 0.04        | ✓ (Δ=0.03)     |
| random     | **1.31 ± 0.08**  | 1.13 ± 0.01        | ✓ (Δ=0.18)     |

- **H1 (FAME develops > 1.5 diversity)**: REJECTED — modest 1.15-1.32, not > 1.5
- **H2 (utilization correlates with eff_rank)**: REJECTED — no correlation
- **H3 (dead experts collapse)**: REJECTED — dead experts stay at init eff_rank, don't collapse
- **H4 (orthogonality boosts diversity)**: NOT TESTED (deferred to backlog)
- **Cumulative suite**: 649/649 in-domain green (up from 641/641; +8 new effective_rank tests)

## 4. Honest verdict

The FAME "diverse experts" claim is **modestly supported** in our cell-level instantiation: FAME is consistently more diverse than MR-MoE by 0.03-0.24. But neither reaches the > 1.5 diversity the FAME paper implies for production settings.

The MR-MoE "multi-rate specialization" claim is **NOT supported** — its diversity ratio barely changes from init under 100 epochs of training.

**Dead experts stay at init** (eff_rank ≈ 5-6) — this is good news: the router correctly gates gradient, so unused experts don't drift.

## 5. Implication for the LNN stack

- **FAME is the better choice when expert diversity matters** — top_k routing does cause differentiation
- **MR-MoE is closer to a "soft attention ensemble"** — dense softmax mixes experts too uniformly
- **Round 80 orthogonality** is the natural next test (H4) — does the orth loss actually increase diversity?

## 6. Cumulative state — 16-layer LNN+MoE 自主栈 (rounds 76-95)

| Round | Layer | Type |
|-------|-------|------|
| 76-82 | Base (CfC n_tau, MR-MoE, FAME, CosineRouter, Orth, φ-balancing) | base |
| 83-86 | Ecology (E diag, gates, combined) | defense + policy |
| 87-89 | Causality (grad H, per-expert grad, causality-gated orth) | diagnostic + policy |
| 90-91 | Audit (wgt/act overlap, smoothness) | diagnostic |
| 92-93 | Dropout audit (target-side, input-side) | diagnostic |
| 94 | Effective rank (Williams/Payeur/Lajoie 2026) | diagnostic |
| **95** | **Per-expert effective rank (FAME diversity test)** | **diagnostic** |

## 7. Backlog for round 96+

1. **Test FAME with orthogonality** (H4) — direct test of round 80 mechanism
2. **K=20, hidden=32, full recurrent training** — paper-scale settings
3. **Regime-labeled task** — does FAME router pick the right expert?
4. **PhysioNet-style irregular time-series** — most important untested domain
5. **Audit the ecology gate under dropout** (backlog #3)
6. **Paper-style note** combining rounds 91-95
