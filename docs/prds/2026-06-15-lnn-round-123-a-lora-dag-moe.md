# PRD #10-85 — Round 123 LoRA-DAG-MoE (Hybrid)

**Date**: 2026-06-15
**Round**: 123
**Status**: Implemented, tested, benched
**Verdict**: **STRICTLY POSITIVE** — 1 NEW BEST on structured_irr

## Goal

Test whether combining **LoRA-MoRE (round 118) expert family** with
**DAG-MoE (round 120) aggregation** produces multiplicative gains.
Round 122 (ProbLoRA) failed because it combined routing × expert
family (coupled in 1D). This round combines expert family ×
aggregation (orthogonal mechanism dimensions).

## Reference papers

- **LoRA-MoRE**: arXiv:2505.22694 (Zhang et al., ACL 2025 Findings)
- **DAG-MoE**: arXiv:2606.01062 (Feng et al., ICML 2026)
- **LoRA**: arXiv:2106.09685 (Hu et al., 2021)

## Design

**LoRA-DAG-MoE** cell:
- Shared base CfC (round 118 pattern)
- K low-rank LoRA adapters (rank r, B-init-zero warm start)
- FAME router (top-K sparse) or Sigmoid router (dense)
- DAG aggregation: L iterations of DAGEdgeGate over K selected
  LoRA deltas

**Forward pass**:
1. `h_base = base_cfc(x_t, h, dt)`
2. `combined = [x_t; h]`
3. `all_deltas = stack([(alpha/r) * B_i(combined @ A_i) for i in K])` (rank-r LoRA)
4. `g_full, top_idx = router(x_t, h)` (top-K sparse with FAME, dense with sigmoid)
5. `selected = gather(all_deltas, top_idx)` (top-K deltas)
6. `top_g = gather(g_full, top_idx)` (top-K mixture weights)
7. `node_outs = top_g * selected + (1/k) * h_base` (DAG init)
8. `refined = dag(node_outs)` (L iterations of DAGEdgeGate)
9. `h_lora = sum_i refined[:, i, :]`
10. `h_new = h_base + h_lora`

## Files

- `lnn/core/lora_dag_moe.py` (NEW, ~360 lines)
- `tests/test_lora_dag_moe.py` (NEW, 26/26 tests)
- `scripts/bench_lora_dag_moe.py` (NEW, 54 cells)
- `results/bench_lora_dag_moe.json` (NEW)
- `docs/research/2026-06-15_lora_dag_moe_report.md`
- `lnn/core/__init__.py` (added LoRA-DAG exports)
- `README.md` (new LoRA-DAG-MoE section)

## Bench (54 cells, 30 epochs, 2 seeds)

| Condition | sin_irr | structured_irr | random_irr |
|-----------|---------|----------------|------------|
| baseline_cfc            | 0.0094 | 0.0053 | **0.0013** |
| lora_k3_r4_dense        | 0.0047 | 0.0036 | 0.0014 |
| dag_moe_k3_l1           | 0.0047 | 0.0030 | 0.0075 |
| **lora_dag_k3_r1_l1**   | 0.0040 | **0.0021** | 0.0038 |
| lora_dag_k3_r4_l1       | 0.0037 | 0.0029 | 0.0070 |
| lora_dag_k3_r4_l2       | 0.0070 | 0.0096 | 0.0143 |
| lora_dag_k3_r4_l3       | 0.2200 | 0.1178 | 0.0275 |
| lora_dag_k3_r8_l2       | 0.1347 | 0.0056 | 0.0092 |

## Verdict

**STRICTLY POSITIVE** — `lora_dag_k3_r1_l1` achieves 0.0021 on
structured_irr (NEW BEST, 30% improvement over prior 0.0030 from
`dag_moe_k3_l1`) at 2.7× smaller parameter cost (4341 vs 11679).

## Why it works (NEW INSIGHT)

The first STRICTLY POSITIVE hybrid in the 91-123 audit. Round 122
(ProbLoRA = routing × expert family) failed because those
dimensions are coupled in 1D. Round 123 (LoRA × DAG = expert ×
aggregation) succeeds because they're orthogonal. The LoRA
experts provide diverse low-rank input signals that the DAG can
compose structurally, vs identical sub-CfC experts that the DAG
struggles to differentiate.

## Future work

1. Sweep alpha ∈ {0.5, 1.0, 2.0, 4.0}
2. Test with sigmoid router for dense mode
3. Test on PhysioNet 36D
4. Add shared expert (round 113) on top
5. Hybrid LoRA-DAG + QuITE for irregular data
6. Larger K (K=4, 6, 8)
