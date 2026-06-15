# PRD #10-84 — Round 122: ProbLoRA-MoE (Probabilistic Routing + LoRA-rank-r deltas)

**Date**: 2026-06-15
**Status**: NEGATIVE-WITH-NUANCE (no NEW BEST)
**Session**: /loop 1h session #47
**Combines**: Round 121 ProbMoE (routing) + Round 118 LoRA-MoRE (experts)

## Motivation

The 91-121 audit identified **8 STRICTLY POSITIVE winners**, two of
which are particularly relevant:
- **ProbMoE (round 121)**: probabilistic routing with marginal
  probability as the per-expert gating signal.  No straight-through
  estimator.
- **LoRA-MoRE (round 118)**: low-rank expert deltas (rank r) added
  to a shared base CfC, with B initialized to zero for warm start.

Round 121's future work suggested combining these two winners. This
PRD implements the hybrid: **ProbLoRA-MoE** keeps the
parameter-efficient LoRA-rank-r experts and swaps the FAME top-K
router for a ProbMoE-style probabilistic router.  This isolates
the question: **does the routing mechanism (probabilistic
marginals vs softmax top-K) matter when the expert family is
already the parameter-efficient LoRA adapter?**

## Architecture

```python
class ProbLoRACfCCell:
    base_cfc : shared base CfC (CfCCell)
    experts  : K low-rank LoRA adapters (rank r, B-init-zero)
    router   : ProbMoERouter (3 modes: exact_k, sample, dynamic_k)

    forward(x_t, h, dt):
        h_base = base_cfc(x_t, h, dt)              # shared
        g, top_idx, probs = router(x_t, h, mode)   # K marginal probs
        combined = [x_t; h]                        # [B, I+H]
        # K low-rank deltas
        all_deltas = stack([(alpha/r) * B_i(combined @ A_i) for i in K])
        # gather top-K deltas
        selected = all_deltas.gather(top_idx)
        h_lora = sum_i g_i * selected_i             # [B, H]
        h_new = h_base + h_lora
```

## Key design choices

1. **LoRA-rank-r expert deltas** — consistent with the 8 winners
   in 91-121 audit (all use sub-MLP experts or low-rank variants).
2. **Probabilistic router** (3 modes) — same ProbMoERouter API as
   round 121, no STE.
3. **Shared base CfC** — same as round 121 (ProbMoE) and round 119
   (PEER).  B-initialized-to-zero adapters.
4. **Parameter efficiency**: rank=2 → 3*2*(I+H) = 60 LoRA params,
   vs full sub-MLP 3*(I+H)*H = 480.  8× smaller per expert.

## Bench results (54 cells)

| Condition | sin_irr | structured_irr | random_irr |
|-----------|---------|----------------|------------|
| baseline_cfc | 0.0094±0.0019 | 0.0053±0.0010 | **0.0013**±0.0004 |
| sigmoid_k3_dense | 0.0048±0.0010 | 0.0034±0.0009 | 0.0052±0.0024 |
| lora_k3_r4_dense | 0.0047±0.0003 | 0.0036±0.0000 | 0.0014±0.0008 |
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
| **problora_k3_r2_exactk** | **3193** (smallest MoE variant!) |
| problora_k3_r4_exactk | 3685 |

## Analysis

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

## Verdict: NEGATIVE-WITH-NUANCE

ProbLoRA is **competitive** (close 2nd on sin with r=4) but
doesn't win.  The 8 STRICTLY POSITIVE winners (99, 102, 105, 107,
113, 114, 116, 118) remain the safe default.  This is the **19th
STRUCTURAL mechanism** in the 91-122 audit and the **11th neg/
target-dep**.

## Audit pattern (91-122)

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

## Critical implementation details

1. **B-init-zero for warm start** — same as round 118 LoRA-MoRE.
   At init, the model is identical to the base CfC, so training
   starts from a known-good baseline.
2. **scale = alpha / rank** — same as round 118.  Default
   alpha=1.0, rank=4 → scale=0.25.
3. **No STE** — gradient flows through marginal probabilities,
   same as round 121.
4. **Parameter efficiency** — K=3 rank=2 → 60 LoRA params vs
   3*(I+H)*H = 480 for full sub-MLP.  8× smaller.

## Files

- `lnn/core/problora_moe.py` (NEW, ~280 lines)
- `tests/test_problora_moe.py` (NEW, 25/25 tests)
- `scripts/bench_problora_moe.py` (NEW, 54 cells)
- `results/bench_problora_moe.json` (NEW)
- `docs/prds/2026-06-15-lnn-round-122-a-problora-moe.md` (this PRD)
- `docs/research/2026-06-15_problora_moe_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v48.md` (digest v48)

## Future work

1. **Sweep alpha ∈ {0.5, 1.0, 2.0, 4.0}** to test the LoRA scaling
2. **Sweep rank r ∈ {1, 2, 4, 8, 16}** more carefully — r=4 vs r=8
3. **Test with shared expert** (combine with round 113 DeepSeek
   Shared Expert Isolation)
4. **Test with sigmoid** (round 116) instead of probabilistic
5. **Test on PhysioNet 36D** — irregular data, may favor LoRA
6. **Hybrid LoRA + DAG-MoE** (combine round 118 + round 120) —
   different combination, same family
