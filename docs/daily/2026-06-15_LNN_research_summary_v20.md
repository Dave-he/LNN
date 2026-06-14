# LNN Research Daily Digest v20 — 2026-06-15 (Round 94)

**Focus**: test the prediction from arXiv:2606.00243 (Williams/Payeur/Lajoie, ICML 2026) that locality-constrained learning rules find low-rank solutions. Does CfC's smoothness prior (round 91) translate to lower effective rank?

## 1. Paper survey

This round focused on a single fresh lead from the arXiv sweep:
- **arXiv:2606.00243** (Williams, Payeur, Lajoie, ICML 2026) — *Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks*. Theoretical analysis showing RFLO solutions are restricted to low-rank perturbations of initial parameters. Connection: CfC's smoothness (round 91) might be functionally a locality constraint, predicting lower effective rank for trained CfC solutions.

No other fresh LNN-specific arXiv hits from June 2026 in cs.NE or cs.LG.

## 2. Implementation: effective rank helper (PRD #10-56)

Round 94 adds 4 functions in `lnn/core/effective_rank.py`:
- `effective_rank(W)` — (Σ σᵢ)² / (Σ σᵢ²) on a 2D matrix
- `mean_effective_rank(weights)` — mean across a list of 2D tensors
- `effective_rank_trajectory(states)` — eff_rank of a (T, d) hidden-state matrix
- `rank_summary(weights, states)` — combines weight + hidden eff_rank

20/20 unit tests pass.

## 3. Test + bench summary

- **20/20 unit tests** pass
- **12-cell bench** (4 models × 3 seeds, 100 epochs, 1D f(t) fitting):

| model | mse   | weight_eff_rank | hidden_eff_rank |
|-------|-------|------------------|------------------|
| MLP   | 0.1721 | **3.61** (lowest) | 1.55 |
| CfC   | 0.2591 | **8.36** (HIGHEST) | 1.93 |
| LSTM  | 0.3366 | 4.73 | 1.73 |
| GRU   | 0.2982 | 3.85 | 2.07 |

- **H1 ✗ (paper prediction)**: CfC has the HIGHEST weight_eff_rank, not lowest. Smoothness is NOT a low-rank bias.
- **H2 ✗ (correlation with smoothness)**: The rank and smoothness rankings are **inverted**. Smoothest model (CfC) has highest rank.
- **H3 PARTIAL**: CfC's hidden_eff_rank = 1.93 < 4 ✓, but GRU/LSTM/MLP are in the same range.
- **H4 ✓ (no collapse)**: All models have eff_rank > 1.5.
- **Cumulative suite**: 641/641 in-domain green (up from 621/621; +20 new)

## 4. Honest-negative: 4-round smoothness audit

| Round | Hypothesis | Verdict |
|-------|-----------|---------|
| 91 (smoothness) | CfC has lower max_grad than MLP | ✓ (2.03 vs 3.66) |
| 92 (target-side dropout) | Smoother → more robust | ✗ (LSTM wins) |
| 93 (input-side dropout) | Smoother → more robust | ✗ (MLP wins) |
| 94 (effective rank) | Smoother → lower rank | ✗ (CfC highest rank) |

**Smoothness is a property of the function class CfC learns, NOT a generic advantage.** The CfC stack should be chosen for tasks where smooth interpolation matters (3DGS, irregular time-series with smooth priors), not for tasks where robustness or parameter efficiency are the primary metrics.

## 5. Verdict on arXiv:2606.00243

| Claim | Status in our stack |
|---|---|
| Locality-restricted learning finds low-rank solutions in linear RNNs | Confirmed in their setting |
| Smoothness is a kind of locality constraint | **REJECTED** in our setting |
| Theory generalizes to continuous-time cells | **REJECTED** — needs RFLO/tBPTT specifically |

## 6. Implications for the LNN stack

- **CfC's representational capacity is NOT underutilized** — its high weight_eff_rank means it's not a "smoothness shortcut"
- **For parameter efficiency** (smallest model that fits), **MLP is the winner** (3.61 rank × 321 params)
- **For 3DGS-style smooth tasks**, CfC's smoothness + high rank is a feature
- **For tasks needing locality in function space**, look at RFLO-style training, not CfC

## 7. Cumulative state — 15-layer LNN+MoE 自主栈 (rounds 76-94)

| Round | Layer | Type |
|-------|-------|------|
| 76-82 | Base (CfC n_tau, MR-MoE, FAME, CosineRouter, Orth, φ-balancing) | base |
| 83-86 | Ecology (E diag, gates, combined) | defense + policy |
| 87-89 | Causality (grad H, per-expert grad, causality-gated orth) | diagnostic + policy |
| 90-91 | Audit (wgt/act overlap, smoothness) | diagnostic |
| 92-93 | Dropout audit (target-side, input-side) | diagnostic |
| **94** | **Effective rank (Williams/Payeur/Lajoie 2026)** | **diagnostic** |

## 8. Backlog for round 95+

1. Real irregular time-series (PhysioNet-style) — the most important untested domain
2. Combined smoothness + state — add gating to CfC for seq-model-style robustness
3. Audit other layer types (FAME top-K under dropout? ecology gate under dropout?)
4. Per-expert effective rank (FAME/MR-MoE experts) — direct test of "experts are diverse" claim
5. Paper-style note combining rounds 91-94 — the 4-round smoothness audit has 3 rejections + 1 confirmation
6. Pivot to a new problem domain — the stack is well-audited
