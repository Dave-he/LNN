# PRD #10-69 — Round 107: Soft MoE Routing for Time-Series (response to arXiv:2308.00951)

**Date**: 2026-06-15
**Round**: 107
**Paper**: arXiv:2308.00951 (Puigcerver et al. ICLR 2023) — *From Sparse to Soft Mixtures of Experts*
**Status**: To implement

## Motivation

The 91-106 audit revealed:
- **H=0 lock-in**: FAME (round 103), SDG-MoE (round 104) both collapse to single-expert routing
- **Structural > routing-only**: QuITE (102), SETA (105) succeeded; routing-only mechanisms (100 SNNL, 101 ORC, 103 QuITE+MoE, 104 SDG-MoE, 106 AuxLF) all failed
- **What we need**: a structural fix that replaces the discrete token-expert assignment with a differentiable alternative

**Soft MoE** (Puigcerver et al. 2023) is exactly this: instead of hard token→expert assignment (with top-K), it computes **learned weighted averages of all tokens** for each expert. Every token contributes to every expert via soft weights, making the entire pipeline differentiable and **inherently avoiding dead experts**.

The mechanism: for a sequence of T tokens and K experts:
1. Compute per-(token, expert) scores: `s_ij = softmax(φ(x_i) · ψ(e_j))` (T × K)
2. **Dispatch**: for each expert j, compute `dispatch_j = Σ_i s_ij · x_i` (weighted average of all tokens)
3. **Process**: `y_j = expert_j(dispatch_j)`
4. **Combine**: `output_i = Σ_j s_ij · y_j` (weighted sum back to tokens)

This is a **structural change to the routing operation itself** (not a refinement), matching the audit pattern.

## Goal

Implement a `SoftMoERouter` that replaces the hard top-K routing in our MoE cells with the Soft MoE dispatch/combine mechanism. Test whether:
- H=0 lock-in is structurally avoided (no expert can become dead)
- test_mse is preserved or improves in our time-series setting
- it composes with SETA's shared+unique decomposition (round 105)

## Hypotheses

**H1 — H=0 lock-in structurally impossible**: routing entropy should be high by construction (every expert sees all tokens via soft weights).

**H2 — test_mse preserved or improved**: Soft MoE is a structural change matching the audit pattern, so should at least not regress.

**H3 — Compositional with SETA**: replace the unique-routing in SETA with Soft MoE → expected to retain shared+unique benefit.

**H4 — Training stable**: full differentiability means no H=0 collapse, no routing collapse.

## Architecture

```
input: x (B, T, D), context (B, d_context)
│
├── Slot embedding ψ(e) ∈ R^(K × d_slot)
├── φ(x) = Linear(D → d_slot) for each token
├── scores = softmax(φ(x) · ψ(e)^T, dim=tokens)  # (B, T, K)
├── dispatch = scores.transpose(-1,-2) @ x  # (B, K, D) — weighted avg per expert
├── expert_k(dispatch_k) for k in 1..K
├── combine = scores @ expert_outputs  # (B, T, D)
└── output = combine
```

## Test plan

- Test dispatch shape (B, T, D) → (B, K, D) with weights summing to 1 over tokens
- Test combine shape (B, K, D) → (B, T, D) with weights summing to 1 over experts
- Test that ALL experts receive non-zero input (no dead experts)
- Test gradient flows through everything
- Test Soft MoE is permutation-invariant in expert order
- Test that adding NaN to a token doesn't crash (NaN-safe)
- Test SETA + Soft MoE: shared experts still always-active, unique uses soft routing

## Bench plan

24 cells:
- 4 conditions: `baseline` (single expert), `fame_h1` (round 78 baseline), `seta` (round 105), `seta_soft`
- 3 datasets: sin_irr, structured_irr, random_irr
- 2 seeds × 100 epochs
- T=32, D=2, hidden=16, K=5 (S=2 shared + U=3 unique), top_k=2
- Measure: test_mse, shared_H, unique_H, expert_util_std (across K experts)

## Expected outcomes (per audit pattern)

- **H1 CONFIRMED**: H=0 lock-in structurally impossible
- **H2 NEUTRAL/POSITIVE**: test_mse likely preserved (architectural change)
- **H3 CONFIRMED**: composes with SETA
- **H4 CONFIRMED**: training stable

## Verdict

If H1 + H3 + H4 confirmed, this adds Soft MoE as a **safer** alternative to top-K routing in our stack. It may not improve test_mse dramatically, but it eliminates a whole class of failure modes (H=0 collapse).

## Why this might still help in 1D time-series

Unlike LLM tokens (long, sparse, irregular), time-series tokens are **dense and structured**. This may actually **favor** Soft MoE because:
- Every expert can learn from the entire sequence context (not just a subset)
- The weighted average is similar to a learned attention pattern
- In 1D, the soft dispatch acts as a learned smoothing operation

## Files to create

- `lnn/core/soft_moe.py` (NEW, ~250 lines)
- `tests/test_soft_moe.py` (NEW, 15+ tests)
- `scripts/bench_soft_moe.py` (NEW, 24-cell bench)
- `docs/research/2026-06-15_soft_moe_routing_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v33.md`
- `README.md` (new section)

## Risks

1. **Computational cost**: dispatch and combine are O(T·K) — but for small T (32) and K (5) this is trivial
2. **Permutation invariance**: expert ordering is meaningless in Soft MoE — need to be careful with init
3. **May not improve over SETA in test_mse**: if both are structural, they may be equivalent

## References

- arXiv:2308.00951 (Puigcerver, Riquelme, Mustafa, Hutter, ICLR 2023) — Soft MoE
- arXiv:2406.18219 (Lo et al. 2024) — Closer look at MoE
- arXiv:2509.11348 (Tran et al. 2025) — Linear Mode Connectivity of MoE
- arXiv:2408.15664 (round 106) — AuxLF
- arXiv:2606.07500 (round 105) — SETA (complementary)
