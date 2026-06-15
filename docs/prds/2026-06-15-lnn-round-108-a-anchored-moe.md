# PRD #10-70 — Round 108: Anchored MoE with Structural Prior (response to arXiv:2605.25166)

**Date**: 2026-06-15
**Round**: 108
**Paper**: arXiv:2605.25166 — *AME-TS: Anchored Mixture-of-Experts for Time Series Forecasting* (Wang, Xue, Razi, Song, Marlowe — May 2026)
**Status**: To implement

## Motivation

Our 91-107 audit pattern is **structural > routing-only**:
- 3 strictly positive winners (99 Reliability Gate, 102 QuITE, 105 SETA, 107 Soft MoE) are all structural
- 6 routing-only refinements (100 SNNL, 101 ORC, 103 QuITE+MoE, 104 SDG-MoE, 106 AuxLF) all fail to improve test_mse

**AME-TS proposes a structural anchoring of routing to interpretable temporal features**, exactly the kind of structural fix our audit predicts will work. It also directly addresses a problem we have: **expert specialization is weakly identified and unstable in time-series MoE** (FAME H=0 in round 78, ORC and SNNL failed to fix it).

## What AME-TS does (in 60 seconds)

Three-stage pipeline:
1. **Regime Predictor**: lightweight module that estimates per-series descriptors:
   - `forecastability` (how predictable is this series?)
   - `seasonality` (what's the dominant period?)
   - `trend` (is the series trending?)
   - `sparsity` (how much missing/non-zero data?)
2. **Structural Prior**: maps descriptors to a soft prior `p_k(descriptors) ∈ Δ^K` over K experts
3. **Routing Anchoring**: token-level routing is anchored to the structural prior, so experts specialize on **interpretable axes** rather than emergent-learned patterns

The anchoring can be done via:
- KL divergence: `loss += λ · KL(p_router || p_prior)` — pulls routing toward the prior
- Direct bias: `logits += log(p_prior)` — additive anchoring
- Mixture: `p_final = α · p_router + (1-α) · p_prior`

## Hypotheses

**H1 — Anchored routing improves stability**: routing should be more stable across training (low variance in expert assignments per series over time).

**H2 — Interpretable experts**: each expert should specialize on a different combination of (forecastability, seasonality, trend, sparsity), so we can NAME what each expert does.

**H3 — test_mse preserved or improved on synthetic**: structural anchoring should not hurt and may help in datasets with clear temporal structure (sin_irr, structured_irr).

**H4 — test_mse preserved on random_irr**: anchoring should be neutral on data without clear structure (random_irr).

## Why this should help (per audit)

- It's a **structural change** to the routing pipeline (adds a regime predictor + structural prior)
- It addresses the **causal problem** (routing decisions should depend on input properties, not noise)
- It's **interpretable** (we can read off what each expert does)
- It can be **combined with SETA** (round 105) — regime predictor on input → structural prior over K unique experts

## Architecture

```
input: x (B, T, D)
  │
  ├── Regime Predictor f_reg(x): (B, T, D) → (B, T, 4)  [forecastability, seasonality, trend, sparsity]
  │   - small 2-layer MLP per timestep (or 1D conv)
  │   - NaN-safe: torch.nan_to_num(x, nan=0) before MLP
  │
  ├── Pool descriptors: (B, T, 4) → (B, 4) [mean over time]
  │
  ├── Structural Prior: p_k = softmax(MLP(descriptors))  (B, K)  [soft prior over K experts]
  │
  ├── Router: logit = Router_MLP([x_t, h, ctx])  (B, K)  [learned routing logits]
  │
  ├── Anchored routing: logit_anchored = logit + log(p_prior + ε)  (B, K)
  │
  └── Top-K: top_k of logit_anchored → soft assignment
```

## Test plan

- Regime predictor returns (B, 4) descriptors in [0, 1] after sigmoid
- Structural prior is a valid probability (sums to 1, non-negative)
- Anchored routing is differentiable (gradients flow)
- Anchored routing reduces variance in expert assignment across seeds
- Each expert specializes on a different combination of descriptors
- NaN-aware (descriptors stay in [0,1] even with missing input)

## Bench plan

12 cells:
- 4 conditions: `baseline` (no anchoring), `anchor_kl` (KL divergence), `anchor_logit` (additive), `anchor_mix` (mixture)
- 3 datasets: sin_irr, structured_irr, random_irr
- 2 seeds × 100 epochs

Measure: test_mse, expert routing entropy, expert specialization (variance in [forecast, season, trend, spars] per expert), routing stability (std of routing assignment across epochs).

## Files to create

- `lnn/core/anchored_moe.py` (NEW, ~300 lines)
- `tests/test_anchored_moe.py` (NEW, 15+ tests)
- `scripts/bench_anchored_moe.py` (NEW, 12-cell bench)
- `docs/research/2026-06-15_anchored_moe_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v34.md`
- `README.md` (new section)
- `lnn-round-108-anchored-moe.md` (memory)

## Risks

1. **Regime predictor may not work** on 1D synthetic data (single feature, no real heterogeneity to detect)
2. **Anchoring may be too strong** — if `λ` is high, routing is dominated by prior and loses flexibility
3. **NaN propagation** — descriptors must be NaN-safe
4. **Interpretability may be illusory** — even with anchoring, the prior could be a meaningless function

## References

- arXiv:2605.25166 — Wang, Xue, Razi, Song, Marlowe (May 2026) *AME-TS: Anchored Mixture-of-Experts for Time Series Forecasting*
- arXiv:2606.08896 — round 78 (FAME, forecastability-aware)
- arXiv:2606.12240 — round 77 (MR-MoE for LNN, baseline we're improving on)
- arXiv:2606.07500 — round 105 (SETA, complementary)
- arXiv:2308.00951 — round 107 (Soft MoE, complementary)
