# PRD #10-67 — Round 105: SETA Sparse Shared + Unique Experts (response to arXiv:2606.07500)

**Date**: 2026-06-15
**Round**: 105
**Paper**: arXiv:2606.07500 (Siddika, Hossen, Mallick, Jannesari, June 2026) — *SETA: Sparse Subspace-to-Expert Sharing for Task-Agnostic Continual Learning*
**Status**: To implement

## Motivation

Rounds 103-104 revealed a **deep structural problem** in the LNN+MoE stack: **multi-expert routing in time-series MoE collapses to a single expert (H=0)**, because all experts see correlated inputs.

FAME router (round 103): `[x_t, h]` dominated by h → single-expert lock-in.
SDG-MoE deliberation (round 104): deliberation pushes experts to consensus → single-expert lock-in.

**What we need**: an architectural change that *forces* a baseline of multi-expert utilization while still allowing experts to specialize on different regimes.

**SETA's insight (re-interpreted for time-series)**: decompose the K experts into two disjoint groups:
- **S = shared experts** (always active, no routing, output averaged) → provides a baseline of multi-expert utilization
- **U = unique experts** (top-k routed among themselves) → specializes on different regimes

This is **architecturally** different from the previous 28 layers of the stack (which all use a single routed group). The shared group is **guaranteed to produce a multi-expert signal** because all S shared experts are always computed and averaged, regardless of routing decisions.

## Goal

Implement `SETAMoECfCCell` and `SETAMoECfCNetwork` that:
1. Decompose experts into shared (always-active) and unique (top-k routed) groups
2. Output: `out = mean(shared_experts) + Σ top-k(unique) g_i · e_i`
3. Compute SETA's two regularizers:
   - **Elastic anchoring**: penalize shared expert weight drift from EMA anchors
   - **Routing regularization**: keep unique router entropy above a target (anti-H=0)
4. Wrap with QuITE context (round 102) for fair comparison with round 103

## Hypotheses

**H1 — SETA breaks the H=0 lock-in**: shared experts are always active, so the routing entropy of the FULL system (Σ |g_i| over shared + unique) should be at minimum `log(S) > 0` for any input.

**H2 — SETA does not hurt test_mse**: the shared experts add capacity without harming the routing decision.

**H3 — Elastic anchoring stabilizes training**: the shared experts don't drift, so the multi-expert baseline is stable across training.

**H4 — Routing regularization keeps unique experts active**: with target entropy > 0, the unique router doesn't collapse to a single expert.

## Architecture

```
input: x_t (B, D), h (B, H), context (B, d_context)
│
├── Shared branch (S experts, always active)
│   ├── expert_0(x_t, h) ────┐
│   ├── expert_1(x_t, h) ────┤
│   └── ...                  ├─ mean ── shared_out (B, H)
│                            ┘
├── Unique branch (U experts, top-k routed)
│   ├── expert_S(x_t, h) ──┐
│   ├── expert_S+1(x_t, h) ─┤
│   ├── ...                ├─ top-k via router + softmax ── unique_out (B, H)
│   └── ...                ┘
│
└── output = shared_out + unique_out
```

## Loss terms

```python
def elastic_anchoring_loss(shared_experts, anchor_state_dict, lambda_val):
    """L2 between current shared expert weights and EMA-snapshotted anchors.
    Returns lambda_val * ||theta_shared - theta_anchor||^2.
    """
    loss = 0.0
    for expert in shared_experts:
        for name, p in expert.named_parameters():
            anchor_p = anchor_state_dict[expert_name + "." + name]
            loss = loss + ((p - anchor_p) ** 2).sum()
    return lambda_val * loss

def routing_regularization(router, target_entropy, lambda_val):
    """Penalize deviation of unique-router entropy from target_entropy.
    Returns lambda_val * (H_actual - target_entropy)^2.
    """
    g = router.last_g  # (B, top_k) — already softmaxed
    p_safe = g.clamp(min=1e-8)
    H_actual = -(p_safe * p_safe.log()).sum(dim=-1).mean()
    return lambda_val * (H_actual - target_entropy) ** 2
```

## Test plan

- Test shared experts always produce output (no routing mask)
- Test unique experts are top-k routed
- Test output is sum of shared-mean + unique-weighted
- Test elastic_anchoring_loss = 0 when weights = anchors
- Test elastic_anchoring_loss > 0 when weights diverge
- Test routing_regularization = 0 when H = target
- Test routing_regularization > 0 when H != target
- Test gradient flows through shared AND unique paths
- Test SETAMoECfCNetwork forward with QuITE context
- Test EMA anchor update

## Bench plan

24 cells:
- 3 conditions: `quite_moe` (baseline round 103), `seta_only_shared`, `seta_full`
- 2 datasets: sin_irr, random_irr (focus on regime-changing data)
- 2 K settings: S=2+U=3 (K=5), S=1+U=4 (K=5)
- 2 seeds × 100 epochs
- T=32, D=2, hidden=16, lr=1e-3, Adam
- Test on data with HIGHER missing rate 50% vs train 30%
- Measure: test_mse, full system routing entropy, shared utilization, unique utilization, training stability

## Expected outcomes

- **H1 CONFIRMED**: full system entropy > 0 (because shared experts always active)
- **H2 PARTIAL**: test_mse same or slightly better (shared experts add capacity)
- **H3 PARTIAL**: shared expert weights are stable across epochs (low weight delta)
- **H4 PARTIAL**: unique router entropy above target (not collapsed to single)

## Verdict

Will be classified as: positive, target-dependent, or negative based on results.

## Why SETA might succeed where FAME/SDG-MoE failed

Both FAME and SDG-MoE attempted to fix the H=0 problem by **routing changes alone** (different router, different gating signal). SETA attempts a **structural** fix:
- FAME: x_t → router → top-k → all experts see same h, experts collapse
- SDG-MoE: x_t → top-k → deliberation → experts reach consensus
- **SETA: x_t → S shared (always-on) + U unique (top-k) → shared provides non-zero baseline utilization by construction**

## Files to create

- `lnn/core/seta_moe.py` (NEW, ~400 lines)
- `tests/test_seta_moe.py` (NEW, 25+ tests)
- `scripts/bench_seta_moe.py` (NEW, 24-cell bench)
- `docs/research/2026-06-15_seta_sparse_shared_experts_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v31.md`
- `README.md` (new section)

## Risks

1. **Always-on shared experts add compute** — S shared experts cost O(S·H·D) per step. For S=2, K=5: 2/5 = 40% extra. Acceptable.
2. **Shared experts may interfere with unique experts** — the shared mean is added to the unique contribution, so the total signal is `shared_mean + unique_weighted`. The shared experts might learn to output a "residual" that the unique experts then refine. Could be net positive.
3. **EMA anchor may be too stable** — if EMA is too slow, anchoring provides no benefit (anchors always close to current weights). If too fast, anchors drift with weights, providing no regularization.

## References

- arXiv:2606.07500 (Siddika et al. June 2026) — SETA
- arXiv:2605.08322 (round 104) — SDG-MoE (predecessor attempt)
- arXiv:2606.08896 (round 78) — FAME
- arXiv:2605.28166 (round 102) — QuITE
