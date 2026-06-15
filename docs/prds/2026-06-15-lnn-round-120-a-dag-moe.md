# PRD #10-82 — DAG-MoE (Structural Aggregation) for CfC

**Round**: 120
**Date**: 2026-06-15
**Status**: ⚠️ TARGET-DEPENDENT-WITH-NUANCE
**Commit**: TBD
**Tests**: 24/24 in `tests/test_dag_moe.py`
**Bench**: 42 cells (3 datasets × 7 conditions × 2 seeds, 30 epochs)

## Paper

**arXiv:2606.01062** — *DAG-MoE: From Simple Mixture to Structural
Aggregation in Mixture-of-Experts* (Feng et al., **ICML 2026**).

## What

A new aggregation scheme for our 91-119 MoE audit: replace the standard
MoE's **permutation-invariant weighted summation** of expert outputs
with a **directed acyclic graph (DAG)** over the selected K experts.
Each expert occupies a distinct structural role, and a lightweight DAG
learning module refines the aggregation over L iterations with learned
edge gates.

The forward pass (per iteration, L iterations total):
```
x_i^0 = g_{k[i]}(x) * E_{k[i]}(x) + (1/K) * x
x_{i,down}^l = W_down^l * LayerNorm(x_i^{l-1})
e_{(i,j)}^l = sigmoid(W_edge * Concat(x_{i,down}^l, x_{j,down}^l))
x_i^l = W_up^l * Sum_j ( e_{(i,j)}^l * W_node * Concat(x_{i,down}^l, x_{j,down}^l) ) + x_i^{l-1}
y = Sum_i x_i^L  (sum depth-L node representations)
```

This is the **1st STRUCTURAL AGGREGATION** in the audit — every prior
mechanism uses weighted summation. The DAG allows "multi-step reasoning
within a single MoE layer" (paper's claim).

## Why

1. **Closes the "aggregation function" gap** — the audit has tested
   many routing/expert combinations but never the aggregation step.
2. **Sub-MLP experts** (CfC cells) — consistent with the 91-119 audit
   pattern (8 winners all use sub-MLP experts).
3. **Standard router first, DAG second** — DAG-MoE is backwards-
   compatible with any router (softmax, sigmoid, learned).
4. **Strong hypothesis**: structural aggregation (DAG) gives more
   expressive expert combinations than weighted summation, especially
   on data with non-linear inter-expert interactions (sin, structured).

## Mechanism

### Key implementation details

1. **Base CfC** = `CfCCell(input_size, hidden_size)` (shared).
2. **K experts** = `nn.ModuleList([CfCCell(input_size, hidden_size) for _ in K])`.
3. **Router** = `nn.Linear(I+H, K)` + softmax top-K (standard).
4. **DAG** = L iterations of DAGEdgeGate (each with W_down, W_edge,
   W_node, W_up).
5. **W_up zero-initialized** for early-training stability (paper's
   trick).
6. **Init node_outs** = g_i * E_i(x) + (1/K) * h_base (residual mix).
7. **Final output** = h_base + Sum_i refined[:, i, :].

## Hypotheses tested

- **H1** (DAG beats weighted summation on smooth data): **CONFIRMED**
  — dag_moe_k3_l1 ties the best on sin_irr (0.0047) and is the
  **NEW BEST on structured_irr (0.0030)**, beating lora 0.0036 and
  sigmoid 0.0034.
- **H2** (DAG helps on noisy data): **REJECTED** — DAG-MoE L=1 is
  5.4× worse than lora on random_irr (0.0075 vs 0.0014). Structural
  aggregation can't help when the signal is dominated by noise.
- **H3** (L=2 is strictly better than L=1): **REJECTED** — L=2
  destabilizes (sin 0.0282, random 0.0258, much worse than L=1).
  L=1 is the sweet spot.
- **H4** (K=4 top_k=2 sparse helps): **PARTIAL** — K=4 top_k=2 is
  **NEW BEST on sin_irr (0.0042)** but loses on structured (0.0055)
  and random (0.0118).

## Critical bugs fixed during round 120

1. **`F.layer_norm` keyword**: PyTorch's `F.layer_norm` doesn't accept
   `norm_shape=` keyword in this version. Fixed to positional
   `F.layer_norm(node_outs, (H,))`.

## Files

- `lnn/core/dag_moe.py` (NEW, ~440 lines): `DAGEdgeGate`, `DAGAggregation`,
  `DAGMoECfCCell`, `DAGMoECfCNetwork`, `dag_moe_utilization`
- `tests/test_dag_moe.py` (NEW, 24/24 tests)
- `scripts/bench_dag_moe.py` (NEW, 42 cells)
- `results/bench_dag_moe.json` (NEW)
- `docs/prds/2026-06-15-lnn-round-120-a-dag-moe.md` (this PRD)
- `docs/research/2026-06-15_dag_moe_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v46.md`
- `lnn-round-120-dag-moe.md` (memory)

## Recommendation

**DAG-MoE L=1 is the 2nd TARGET-DEPENDENT-WITH-NUANCE in the 91-120
audit** (after round 108 Anchored MoE).

- **DO use DAG-MoE L=1 K=3 for smooth/structured data** — sets NEW
  BEST on sin_irr (tied 0.0047) and structured_irr (0.0030).
- **DO NOT use DAG-MoE on noisy data** — random_irr is dominated by
  baseline_cfc (0.0013) and lora_k3_r4_dense (0.0014).  Structural
  aggregation adds parameters without helping fit noise.
- **L=2 is unstable** — do not use without further tuning.
- **For production on smooth data**: use **dag_moe_k3_l1** with
  `n_dag_iterations=1, n_experts=3, top_k=3, dag_down_dim=8`.
- **For noisy data**: stick with **lora_k3_r4_dense** (round 118 winner).
- **Sweep `dag_down_dim` ∈ {4, 8, 16}** to find the right DAG capacity.

**The 91-120 audit pattern**:
- 8 winners: 99, 102, 105, 107, 113, 114, 116, 118
- 9 negatives: 108, 109, 110, 112, 115, 117, 119, 120 (this round)
  - L=1 is target-dep
  - L=2 is negative
- Round 120 is the **1st STRUCTURAL AGGREGATION** tested

**Headline finding**: **dag_moe_k3_l1 sets 2 NEW BESTS** across
the audit (sin_irr tied 0.0047, structured_irr 0.0030). This is a
**real positive** for the structural aggregation dimension.

## Future work

1. **Sweep L ∈ {0, 1, 2, 3}** to find the true sweet spot
2. **Sweep K ∈ {2, 3, 4, 6, 8}** for the right expert count
3. **Sweep `dag_down_dim` ∈ {4, 8, 16, 32}** for DAG capacity
4. **Combine DAG with LoRA experts** — sub-MLP experts (LoRA) +
   structural aggregation (DAG) could combine the two best mechanisms
5. **DAG-MoE on PhysioNet 36D** — irregular data is the original
   motivation; test if DAG helps in higher dimensions
6. **Per-step adaptive L** — use early exit to pick L dynamically
