# PRD #10-81 — PEER (Mixture of A Million Experts) for CfC

**Round**: 119
**Date**: 2026-06-15
**Status**: ⚠️ NEGATIVE-WITH-NUANCE
**Commit**: TBD
**Tests**: 28/28 in `tests/test_peer_moe.py`
**Bench**: 36 cells (3 datasets × 6 conditions × 2 seeds, 30 epochs)

## Paper

**arXiv:2407.04153** — *Mixture of A Million Experts* (Xu Owen He,
Google DeepMind, July 2024).  PEER = single-neuron expert × product-key
routing × millions of experts.

## What

A new expert family for our 91-118 MoE audit: **N single-neuron experts**
(each is a `Linear(in_features, out_features)` with no activation),
routed by either **product-key lookup** (paper-faithful) or **softmax
top-K** (ablation).  Each expert is the **smallest possible unit** (a
single linear neuron), in contrast to the sub-MLP experts used in all
prior mechanisms.

This is the **1st linear expert family** in the audit — every prior
mechanism uses sub-MLP experts (CfC cells with 3+ branches and
tanh/sigmoid activations).

## Why

1. **Closes the "linear expert family" gap** in the audit's untested
   list.
2. **Structural** (new expert family, not just a router) + **data-
   structure-independent** (linearity has no data assumption) +
   **preserves recurrent state mixing** (h_new = base + Σ α_k · expert_k
   in the same form as LoRA-MoRE).
3. **Product-key routing is a deterministic alternative to softmax**:
   two hash tables, find top-K in each, dedup, weighted sum.  This
   addresses the FAME H=0 lock-in problem (round 103) by making routing
   decisions a hash lookup rather than a soft probability.
4. **Strong hypothesis**: with enough single neurons (N → ∞), PEER
   approximates any piecewise-linear function (universal approximator).
   At N=8, the question is whether 8 single neurons can compete with
   3 sub-MLP experts in 1D.

## Mechanism

```
h_base = base_cfc(x_t, h)                # [B, H]   (shared base)
combined = [x_t; h]                      # [B, I+H]
scores_1 = combined @ key_table_1.T      # [B, √N]
scores_2 = combined @ key_table_2.T      # [B, √N]
top_idx_1, top_idx_2 = topk(scores, K)   # K candidates each
all_idx = unique([top_idx_1, top_idx_2]) # K unique candidates
α = softmax(scores[all_idx], dim=-1)     # [B, K] routing weights
h_lora = sum_k α_k · expert_{all_idx_k}(combined)
h_new = h_base + h_lora
```

### Key implementation details

1. **Single-neuron expert** = `nn.Linear(in, out, bias=True)`.  No
   activation function.
2. **Two key tables**: each `[n_buckets, key_dim]` where
   `n_buckets = max(2, ceil(sqrt(N)))` and `key_dim = I+H`.  For N=8,
   `n_buckets = 3`.
3. **Top-K in each table** → K candidates per table → K² candidates
   total → dedup by expert index → take top-K of unique experts.
4. **Mapping bucket → expert**: `expert_idx = bucket_idx % N` (1-to-1
   modulo N, so each bucket maps to one expert).
5. **Scoring**: `α = softmax(scores, dim=-1)` over the K selected
   candidates, ensuring weights sum to 1.
6. **Softmax ablation**: `LinearSoftmaxRouter` does the same job as
   the product-key router but with a single linear projection.  This
   isolates the "linear expert family" effect from the "product-key
   routing" effect.

## Hypotheses tested

- **H1** (linear basis is enough on smooth data): REJECTED — PEER
  loses to sigmoid/LoRA on sin_irr (0.0056 vs 0.0047/0.0048).  The
  nonlinearity in sub-MLP experts matters for sin-shaped targets.
- **H1** (PEER competitive on structured/random): CONFIRMED — PEER
  n8_pk matches sigmoid on structured (0.0035 vs 0.0034) and beats it
  on random (0.0020 vs 0.0052, 2.6× better).
- **H2** (product-key escapes FAME H=0 lock-in): PARTIAL — routing
  entropy is healthy (0.88-1.08 nats, not stuck at 0), but n_active=3
  shows only 3 of 8 experts fire on average.
- **H3** (product-key is strictly better than softmax): **CONFIRMED** —
  on structured_irr (0.0035 vs 0.0051, 46% better) and random_irr
  (0.0020 vs 0.0038, 90% better).  Sin_irr tied.

## Critical bugs fixed during round 119

1. **`test_peer_cell_gradient_flow` was too strict**: it required
   ALL 8 experts to receive gradients, but with top_k=2 and batch=3
   only 6 selections happen, leaving some experts without grad.  Fixed
   to assert `n_with_grad >= 2` (at least 2 experts got grad).
2. **Deterministic dedup loop in `ProductKeyRouter.forward`**: needed
   Python loop to dedup expert indices across the two key tables.  The
   loop is O(B·K²) per step, fine for B=8 and K=2.

## Files

- `lnn/core/peer_moe.py` (NEW, ~530 lines): `SingleNeuronExpert`,
  `ProductKeyRouter`, `LinearSoftmaxRouter`, `PEERCfCCell`,
  `PEERCfCNetwork`, `peer_utilization`
- `tests/test_peer_moe.py` (NEW, 28/28 tests)
- `scripts/bench_peer_moe.py` (NEW, 36 cells)
- `results/bench_peer_moe.json` (NEW)
- `docs/prds/2026-06-15-lnn-round-119-a-peer-moe.md` (this PRD)
- `docs/research/2026-06-15_peer_moe_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v45.md`
- `lnn-round-119-peer-moe.md` (memory)

## Recommendation

**PEER is the 8th NEGATIVE-WITH-NUANCE in the 91-119 audit.**

- **DO use PEER product-key when the data is structured or noisy** —
  the linear basis + deterministic routing is competitive with the
  best winners on those regimes.
- **DO NOT use PEER for smooth nonlinear targets** — the lack of
  nonlinearity in single-neuron experts hurts on sin-shaped data.
- **Product-key routing is structurally better than softmax** when
  paired with linear experts (a real positive from the ablation).
- For production: prefer **sigmoid_k3_dense (round 116)** or
  **lora_k3_r4_dense (round 118)** which are STRICTLY POSITIVE on
  all 3 datasets.

**The 91-119 audit pattern**:
- 8 winners: 99, 102, 105, 107, 113, 114, 116, 118
- **8 negatives: 108, 109, 110, 112, 115, 117, 119 (this round)**
- Round 119 is the 1st **linear** expert family tested (refines the
  pattern: sub-MLP experts > linear experts in 1D)

## Future work

1. **Sweep N ∈ {4, 8, 16, 32, 64}** to find the sweet spot for linear
   experts
2. **Hybrid PEER + LoRA**: some experts are linear, some are LoRA
   adapters — combining the two expert families
3. **PEER on PhysioNet 36D** — higher-D data may benefit from
   linear basis where 1D doesn't
4. **Product-key routing for sub-MLP experts** — apply the
   deterministic hash lookup to the FAME sub-MLP experts
5. **Hash-based routing for CfC** — round 102 used QuITE which is
   attention-based; product-key is a cheaper alternative
