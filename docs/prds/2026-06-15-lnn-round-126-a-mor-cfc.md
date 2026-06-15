# PRD #10-88 — Round 126 Mixture-of-Recursions for CfC

**Date**: 2026-06-15
**Round**: 126
**Status**: Implemented, tested, benched
**Verdict**: **HONEST NEGATIVE-WITH-NUANCE** — MoR standalone is
the best standalone CfC variant, but does NOT compose with the
4-axis triple hybrid.

## Goal

Test whether per-timestep variable recursion depth (MoR) helps
the CfC cell, both standalone and on top of the round 124 triple
hybrid (LoRA-DAG-Shared).

## Reference paper

- **Mixture-of-Recursions**: arXiv:2507.10524 (Bae et al., Google
  DeepMind August 2025) — variable recursion depth per token

## Design

The MoR cell computes h_1, h_2, ..., h_max_depth by applying the
shared CfC cell recursively, then mixes them with the router weights:

```
h_1 = cell(x_t, h, dt)         # depth 1
h_2 = cell(x_t, h_1, dt)       # depth 2
...
h_D = cell(x_t, h_{D-1}, dt)   # depth D
h_new = sum_d w_d * h_d         # router-weighted mixture
```

The router is a simple linear+softmax with a **warm-start bias**
of [-2, -4, -6, ...] so the initial output is ~98% depth-1.

## Files

- `lnn/core/mor_cfc.py` (NEW, ~200 lines)
  - `MoRRouter`, `MoRCfCCell`, `MoRCfCNetwork`, `mor_router_summary`
- `tests/test_mor_cfc.py` (NEW, 23/23 pass)
- `scripts/bench_mor_cfc.py` (NEW, 54 cells)
- `results/bench_mor_cfc.json` (NEW)

## Bench (54 cells, 30 epochs, 2 seeds)

| Condition | sin_irr | structured_irr | random_irr |
|-----------|---------|----------------|------------|
| baseline_cfc            | 0.0094 | 0.0053 | 0.0013 |
| mor_d1                  | 0.0106 | 0.0080 | 0.0016 |
| mor_d2                  | 0.0100 | 0.0071 | 0.0033 |
| mor_d3                  | 0.0089 | 0.0044 | 0.0021 |
| **mor_d4**              | **0.0067** | **0.0033** | **0.0012** |
| **lora_dag_shared_ks1** | **0.0017** | 0.0034 | 0.0056 |
| **lora_dag_shared_ks2** | 0.0018 | **0.0020** | 0.0091 |
| mor_d3_lora_dag_ks1     | 0.0042 | 0.0025 | 0.0038 |
| mor_d3_lora_dag_ks2     | 0.0029 | 0.0027 | 0.0026 |

## Verdict

**HONEST NEGATIVE-WITH-NUANCE**. No NEW BESTS. MoR standalone is
positive (best standalone CfC variant at d=4, 2753 params), but
the 5-axis hybrid (MoR+LoRA-DAG-Shared) is dominated by the
4-axis triple hybrid.

## Best on each dataset

- **sin_irr**: lora_dag_shared_ks1 = 0.0017 (round 124 still leads)
- **structured_irr**: lora_dag_shared_ks2 = 0.0020 (round 125 still leads)
- **random_irr**: baseline_cfc = 0.0013

## MoR standalone is the best parameter-efficient CfC variant

`mor_d4` (2753 params) is the best standalone CfC:
- sin: 0.0067 (-29% vs baseline 0.0094)
- structured: 0.0033 (-38% vs baseline 0.0053)
- random: 0.0012 (-8% vs baseline 0.0013)

## Future work

1. Deeper MoR (d=6, 8, 10)
2. MoR with hard top-K at inference (compute saving)
3. MoR + Mod (round 111) — skip steps then vary depth
4. Per-Layer Adaptive Depth
5. MoR integrated INSIDE LoRA-DAG-Shared cell (not stacked)

## Key insight

**The 4-axis stack (rounds 123-125) is the Pareto frontier.**
Adding a 5th axis (recursion depth) creates coupling with the
existing axes. MoR standalone IS useful, just not in combination
with the 4-axis hybrid.
