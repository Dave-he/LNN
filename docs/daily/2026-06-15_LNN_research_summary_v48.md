# LNN Research Digest v48 — 2026-06-15

**Coverage**: ProbLoRA-MoE (Probabilistic Routing + LoRA-rank-r deltas) for CfC — hybrid of round 121 ProbMoE + round 118 LoRA-MoRE (PRD #10-84). 91-122 audit update (NEGATIVE-WITH-NUANCE, 19th structural mechanism, hybrid doesn't beat best of components).

## Headline

Round 122 implemented **ProbLoRA-MoE**, the natural hybrid of two of the 8 STRICTLY POSITIVE winners in the 91-121 audit: **ProbMoE (round 121) routing** + **LoRA-MoRE (round 118) experts**. The hybrid combines parameter-efficient LoRA-rank-r expert deltas (B-init-zero, scale=alpha/r) with probabilistic marginal routing (no STE).

**The result is NEGATIVE-WITH-NUANCE** — ProbLoRA does NOT beat the best of its components. ProbMoE alone wins on sin_irr (0.0026 vs 0.0029 for ProbLoRA r=4) and LoRA alone wins on structured_irr (0.0036 vs 0.0044 for ProbLoRA r=2). ProbLoRA is the **smallest MoE variant** (3193 params, r=2) but the parameter efficiency doesn't translate to test_mse gains.

Bench at 30 epochs (54 cells, 2 seeds):

| Condition | sin_irr | structured_irr | random_irr | n_params |
|-----------|---------|----------------|------------|----------|
| baseline_cfc            | 0.0094±0.0019 | 0.0053±0.0010 | **0.0013**±0.0004 | 2545 |
| sigmoid_k3_dense        | 0.0048±0.0010 | 0.0034±0.0009 | 0.0052±0.0024 | 7763 |
| **lora_k3_r4_dense**    | 0.0047±0.0003 | **0.0036**±0.0000 | 0.0014±0.0008 | 3691 |
| lora_k3_r4_top2         | 0.0049±0.0005 | 0.0037±0.0002 | 0.0031±0.0003 | 3685 |
| **prob_moe_k3_exactk**  | **0.0026**±0.0004 | 0.0036±0.0004 | 0.0048±0.0037 | 10285 |
| problora_k3_r2_exactk   | 0.0050±0.0006 | 0.0044±0.0021 | 0.0035±0.0024 | **3193** |
| problora_k3_r4_exactk   | 0.0029±0.0001 | 0.0045±0.0015 | 0.0029±0.0019 | 3685 |
| problora_k3_r2_sample   | 0.0050±0.0008 | 0.0044±0.0016 | 0.0033±0.0022 | 3193 |
| problora_k3_r4_dynamick | 0.0029±0.0001 | 0.0045±0.0015 | 0.0029±0.0019 | 3685 |

**Best on each dataset (no NEW BESTS)**:
- **sin_irr**: prob_moe_k3_exactk = 0.0026 (round 121 still leads, ProbLoRA r=4 close 2nd at 0.0029)
- **structured_irr**: dag_moe_k3_l1 = 0.0030 (round 120 still leads), lora_k3_r4_dense = 0.0036 (round 118 still leads ProbLoRA)
- **random_irr**: baseline_cfc = 0.0013 (no MoE needed)

Key findings:
- **Hybrid doesn't beat best of components** — ProbMoE wins sin (0.0026 vs 0.0029), LoRA wins struct (0.0036 vs 0.0044)
- **ProbLoRA r=2 is the smallest MoE variant (3193 params)** — 8× smaller per expert than full sub-MLP
- **Mode choice doesn't matter for ProbLoRA** — exact_k and dynamic_k produce identical results (threshold = 1/n = 0.33 < all 3 probs), sample is slightly noisier
- **Routing and expert family are not orthogonal** — switching the router changes the optimal expert family

## 1. ProbLoRA-MoE in 60 seconds

ProbMoE (round 121): per-expert probability p_i is the marginal probability of expert i being in the selected subset. Gradient flows through marginals, no straight-through estimator.

LoRA-MoRE (round 118): low-rank expert deltas `(alpha/r) * B(combined @ A)` added to a shared base CfC, with B initialized to zero for warm start.

ProbLoRA-MoE = ProbMoE router + LoRA-rank-r experts. The hybrid is parameter-efficient (r=2 → 60 LoRA params vs 480 for full sub-MLP, 8× smaller).

## 2. Why ProbLoRA is NEGATIVE-WITH-NUANCE

### H1 ✗ ProbLoRA does NOT beat ProbMoE on sin_irr

ProbMoE exact_k: 0.0026 (round 121 winner)
ProbLoRA r=2: 0.0050 (worse, -92%)
ProbLoRA r=4: 0.0029 (close 2nd, +12%)

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

## 3. 91-122 audit pattern update

**Pattern (91-122)**: 19 structural mechanisms tested. **8 winners (STRICTLY POSITIVE)**: 99, 102, 105, 107, 113, 114, 116, 118. **11 negatives/target-dep**: 108, 109, 110, 112, 115, 117, 119, 120, 121, **122**.

| Round | Mechanism | Type | Verdict |
|-------|-----------|------|---------|
| 99 | Reliability gate | Augmentation | STRICTLY POSITIVE |
| 102 | QuITE | Embedding | STRICTLY POSITIVE |
| 105 | SETA | Architecture | STRICTLY POSITIVE |
| 107 | Soft MoE | Structural | SAFER ROUTING |
| 108 | Anchored MoE | Structural | TARGET-DEP |
| 109 | Dynamic TMoE | Structural | NEGATIVE-WITH-NUANCE |
| 110 | Freq Experts | Structural | NEGATIVE-WITH-NUANCE |
| 112 | Expert Choice | Structural | NEGATIVE |
| 113 | DeepSeek Shared Expert | Structural (residual) | STRICTLY POSITIVE |
| 114 | ReMoE (ReLU Routing) | Structural (soft gating) | STRICTLY POSITIVE |
| 115 | MH-MoE (Multi-Head) | Structural (sub-token) | NEGATIVE |
| 116 | Sigmoid Routing | Structural (no normalization) | STRICTLY POSITIVE |
| 117 | Gumbel-Softmax | Structural (stochastic) | NEGATIVE-WITH-NUANCE |
| 118 | LoRA-MoRE | Structural (rank-r delta) | STRICTLY POSITIVE |
| 119 | PEER (Single Neurons) | Structural (linear expert) | NEGATIVE-WITH-NUANCE |
| 120 | DAG-MoE (DAG aggregation) | Structural (DAG) | TARGET-DEP-WITH-NUANCE |
| 121 | ProbMoE (Probabilistic) | Structural (probabilistic) | TARGET-DEP-WITH-NUANCE |
| **122** | **ProbLoRA-MoE (Hybrid)** | **Structural (probabilistic + LoRA)** | **NEGATIVE-WITH-NUANCE** |

**NEW INSIGHT (round 122)**: **Hybrid of 2 winners doesn't beat best of components**. Routing (ProbMoE) and expert family (LoRA) are not orthogonal — switching the router changes the optimal expert family. The 8 winners represent Pareto frontier, not a foundation for multiplicative combination.

This refines the audit pattern: **in 1D, the routing choice and expert family are coupled**. Future combinations should consider orthogonal mechanisms (e.g., ProbLoRA + DAG-MoE) rather than routing + expert.

## 4. Implementation details

- **Core**: `lnn/core/problora_moe.py` (NEW, ~280 lines)
  - `ProbLoRAExpert(in_features, out_features, rank, alpha, dropout, small_init)` — LoRA adapter
  - `ProbLoRACfCCell(input_size, hidden_size, n_experts, top_k, rank, alpha, temperature, mode)` — cell
  - `ProbLoRACfCNetwork(...)` — stacked ProbLoRA CfC
  - `problora_moe_utilization(cell)` — diagnostic
- **Tests**: `tests/test_problora_moe.py` (NEW, 25/25 pass)
  - Expert: init, B-zero-at-init, forward shape, forward zero at init, with dropout (5 tests)
  - Cell: init in 3 modes, forward shape in 3 modes, forward_with_aux, gradient flow (exact_k + sample), diag metadata, smoke sin (3 modes) (12 tests)
  - Network: forward, last_step, NaN, learns (4 tests)
  - Bench-style: mini-bench sin, parameter count, parameter efficiency vs ProbMoE (3 tests)
- **Bench**: `scripts/bench_problora_moe.py` (NEW, 54 cells, 30 epochs)
  - 3 datasets × 9 conditions × 2 seeds
  - Conditions: baseline_cfc, sigmoid_k3_dense, lora_k3_r4_dense, lora_k3_r4_top2, prob_moe_k3_exactk, problora_k3_r2_exactk, problora_k3_r4_exactk, problora_k3_r2_sample, problora_k3_r4_dynamick
- **PRD**: `docs/prds/2026-06-15-lnn-round-122-a-problora-moe.md` (PRD #10-84)
- **Report**: `docs/research/2026-06-15_problora_moe_report.md`
- **Memory**: `lnn-round-122-problora-moe.md`
- **Exports**: `lnn/core/__init__.py` adds `ProbLoRAExpert, ProbLoRACfCCell, ProbLoRACfCNetwork, problora_moe_utilization`

## 5. Critical implementation details

1. **B-init-zero for warm start** — same as round 118 LoRA-MoRE. At init, the model is identical to the base CfC.
2. **scale = alpha / rank** — same as round 118. Default alpha=1.0, rank=4 → scale=0.25.
3. **No STE** — gradient flows through marginal probabilities, same as round 121.
4. **Parameter efficiency** — K=3 rank=2 → 60 LoRA params vs 3*(I+H)*H = 480 for full sub-MLP. 8× smaller.

## 6. Future work

1. **Sweep alpha ∈ {0.5, 1.0, 2.0, 4.0}** to test the LoRA scaling
2. **Sweep rank r ∈ {1, 2, 4, 8, 16}** more carefully — r=4 vs r=8
3. **Test with shared expert** (combine with round 113 DeepSeek Shared Expert Isolation)
4. **Test with sigmoid** (round 116) instead of probabilistic
5. **Test on PhysioNet 36D** — irregular data, may favor LoRA
6. **Hybrid LoRA + DAG-MoE** (combine round 118 + round 120) — different combination, same family

## 7. Recommendation

**ProbLoRA-MoE is the 11th NEGATIVE/TARGET-DEP in the 91-122 audit** (after rounds 108, 109, 110, 112, 115, 117, 119, 120, 121).

- **DO use prob_moe_k3_exactk** (round 121) for smooth data (sin_irr).
- **DO use lora_k3_r4_dense** (round 118) for structured data.
- **DO NOT use ProbLoRA-MoE** — the hybrid doesn't beat the best of its components in 1D.
- **For production on smooth data**: use **prob_moe_k3_exactk** with `n_experts=3, top_k=2, mode="exact_k"`.
- **For production on structured data**: use **lora_k3_r4_dense** with `n_experts=3, top_k=0, rank=4, alpha=1.0, router_type="sigmoid"`.
- **For production on noisy data**: use **baseline_cfc** (no MoE).
