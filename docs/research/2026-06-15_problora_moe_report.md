# Round 122 — ProbLoRA-MoE (Probabilistic Routing + LoRA-rank-r) — Research Report

**Date**: 2026-06-15
**PRD**: #10-84
**Commit**: TBD
**Verdict**: NEGATIVE-WITH-NUANCE (no NEW BEST, but 2nd place on sin_irr with r=4)

## Summary

Implemented **ProbLoRA-MoE** (ProbMoE routing + LoRA-rank-r deltas,
hybrid of round 121 + round 118 winners) and tested on 3 datasets
× 9 conditions × 2 seeds = 54 cells.  The hybrid doesn't beat the
best of its components — ProbMoE alone (round 121) wins on sin_irr
(0.0026 vs 0.0029 for ProbLoRA r=4) and LoRA alone (round 118) wins
on structured_irr (0.0036 vs 0.0044 for ProbLoRA r=2).  ProbLoRA
is the **smallest MoE variant** at 3193 params (r=2), but the
parameter efficiency doesn't translate to test_mse gains.

## 1. Architecture

```python
class ProbLoRACfCCell:
    base_cfc : shared base CfC (CfCCell)
    experts  : K low-rank LoRA adapters (rank r, B-init-zero)
    router   : ProbMoERouter (3 modes: exact_k, sample, dynamic_k)

    forward(x_t, h, dt):
        h_base = base_cfc(x_t, h, dt)              # shared
        g, top_idx, probs = router(x_t, h, mode)   # K marginal probs
        combined = [x_t; h]                        # [B, I+H]
        all_deltas = stack([(alpha/r) * B_i(combined @ A_i) for i in K])
        selected = all_deltas.gather(top_idx)
        h_lora = sum_i g_i * selected_i             # [B, H]
        h_new = h_base + h_lora
```

The hybrid combines:
- **ProbMoE router (round 121)**: per-expert probability p_i is the
  marginal probability of expert i being in the selected subset.
  Gradient flows through marginals, no STE.
- **LoRA-rank-r experts (round 118)**: low-rank adapters
  `(alpha/r) * B(combined @ A)` with B-init-zero for warm start.

## 2. Implementation

### Files
- `lnn/core/problora_moe.py` (NEW, ~280 lines)
  - `ProbLoRAExpert(in, out, rank, alpha, dropout)` — LoRA adapter
  - `ProbLoRACfCCell(input, hidden, n_experts, top_k, rank, alpha, temperature, mode)` — cell
  - `ProbLoRACfCNetwork(...)` — stacked network
  - `problora_moe_utilization(cell)` — diagnostic
- `tests/test_problora_moe.py` (NEW, 25/25 tests)
  - Expert: init, B-zero-at-init, forward shape, forward zero at init, with dropout
  - Cell: init in 3 modes, forward shape in 3 modes, forward_with_aux, gradient flow (exact_k + sample), diag metadata, smoke sin (3 modes)
  - Network: forward, last_step, NaN, learns
  - Bench-style: mini-bench sin, parameter count, parameter efficiency vs ProbMoE

## 3. Bench results (54 cells, 30 epochs, 2 seeds)

| Condition | sin_irr | structured_irr | random_irr |
|-----------|---------|----------------|------------|
| baseline_cfc | 0.0094±0.0019 | 0.0053±0.0010 | **0.0013**±0.0004 |
| sigmoid_k3_dense | 0.0048±0.0010 | 0.0034±0.0009 | 0.0052±0.0024 |
| lora_k3_r4_dense | 0.0047±0.0003 | **0.0036**±0.0000 | 0.0014±0.0008 |
| lora_k3_r4_top2 | 0.0049±0.0005 | 0.0037±0.0002 | 0.0031±0.0003 |
| **prob_moe_k3_exactk** | **0.0026**±0.0004 | 0.0036±0.0004 | 0.0048±0.0037 |
| problora_k3_r2_exactk | 0.0050±0.0006 | 0.0044±0.0021 | 0.0035±0.0024 |
| problora_k3_r4_exactk | 0.0029±0.0001 | 0.0045±0.0015 | 0.0029±0.0019 |
| problora_k3_r2_sample | 0.0050±0.0008 | 0.0044±0.0016 | 0.0033±0.0022 |
| problora_k3_r4_dynamick | 0.0029±0.0001 | 0.0045±0.0015 | 0.0029±0.0019 |

### Parameter counts

| Condition | n_params |
|-----------|----------|
| baseline_cfc | 2545 |
| sigmoid_k3_dense | 7763 |
| lora_k3_r4_dense | 3691 |
| lora_k3_r4_top2 | 3685 |
| prob_moe_k3_exactk | 10285 |
| **problora_k3_r2_exactk** | **3193** |
| problora_k3_r4_exactk | 3685 |

## 4. Analysis

### H1 ✗ ProbLoRA does NOT beat ProbMoE on sin_irr

ProbMoE exact_k: 0.0026 (round 121 winner)
ProbLoRA r=2: 0.0050 (worse, -92%)
ProbLoRA r=4: 0.0029 (close, +12%)

The low-rank bottleneck hurts at small r (r=2 too restrictive) but
matches at r=4.  ProbLoRA never beats ProbMoE.

### H2 ✗ ProbLoRA does NOT beat LoRA on structured_irr

LoRA r=4 dense: 0.0036 (round 118 winner)
ProbLoRA r=2: 0.0044 (worse, +22%)
ProbLoRA r=4: 0.0045 (worse, +25%)

Replacing the FAME top-K router with ProbMoE hurts on structured
data.  The FAME router has a specific "forecastability" bias that
ProbMoE doesn't replicate.

### H3 ✓ ProbLoRA is the smallest MoE variant (3193 params)

ProbLoRA r=2 with K=3 experts is 3193 params — smaller than
lora_k3_r4_dense (3691) and far smaller than sigmoid_k3_dense
(7763) or prob_moe_k3_exactk (10285).  This is the most
parameter-efficient MoE in the audit.

### H4 ✗ Hybrid doesn't beat best of components

The fundamental finding: **the hybrid (ProbLoRA) doesn't beat the
best of its components** (ProbMoE alone wins on sin, LoRA alone
wins on struct).  The two mechanisms are not orthogonal — they
address different aspects (routing vs expert capacity) but
combining them doesn't give multiplicative gains in 1D.

## 5. Audit pattern (91-122)

**19 structural mechanisms tested. 8 winners (STRICTLY POSITIVE)**: 99, 102, 105, 107, 113, 114, 116, 118. **11 negatives/target-dep**: 108, 109, 110, 112, 115, 117, 119, 120, 121, **122**.

**Why:** The 8 winners use sub-MLP experts or low-rank LoRA
deltas with various routers.  ProbLoRA combines the best routing
(round 121) with the best expert family (round 118) but the
combination doesn't exceed either alone.  In 1D, the routing
choice and expert family are not orthogonal — switching the
router changes the optimal expert family.

**How to apply:** Use **prob_moe_k3_exactk** (round 121) for
smooth data (sin). Use **lora_k3_r4_dense** (round 118) for
structured data. Don't use both — the hybrid is strictly worse
than the best component.

## 6. Critical implementation details

1. **B-init-zero for warm start** — same as round 118 LoRA-MoRE.
   At init, the model is identical to the base CfC, so training
   starts from a known-good baseline.
2. **scale = alpha / rank** — same as round 118.  Default
   alpha=1.0, rank=4 → scale=0.25.
3. **No STE** — gradient flows through marginal probabilities,
   same as round 121.
4. **Parameter efficiency** — K=3 rank=2 → 60 LoRA params vs
   3*(I+H)*H = 480 for full sub-MLP.  8× smaller.

## 7. Future work

1. **Sweep alpha ∈ {0.5, 1.0, 2.0, 4.0}** to test the LoRA scaling
2. **Sweep rank r ∈ {1, 2, 4, 8, 16}** more carefully — r=4 vs r=8
3. **Test with shared expert** (combine with round 113 DeepSeek
   Shared Expert Isolation)
4. **Test with sigmoid** (round 116) instead of probabilistic
5. **Test on PhysioNet 36D** — irregular data, may favor LoRA
6. **Hybrid LoRA + DAG-MoE** (combine round 118 + round 120) —
   different combination, same family
