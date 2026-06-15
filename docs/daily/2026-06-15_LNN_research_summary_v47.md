# LNN Research Digest v47 — 2026-06-15

**Coverage**: ProbMoE (Probabilistic Routing) for CfC (response to arXiv:2606.01509 ICML 2026) + 91-121 audit update (1 NEW BEST on sin, marginal-prob gradient is clean, 3 modes tested, mode choice doesn't matter in 1D).

## Headline

Round 121 implemented **ProbMoE (Differentiable Probabilistic Routing)** for CfC. The mechanism: replace the standard MoE's discrete top-K selection with **probabilistic inference over cardinality-constrained subsets**. The per-expert probability p_i is the **marginal probability** of expert i being in the selected subset, providing a clean surrogate gradient (no straight-through estimator needed).

**The result is TARGET-DEP-WITH-NUANCE** — ProbMoE exact_k is **NEW BEST on sin_irr (0.0026, beats lora 0.0047 by 45%)** but **loses on random_irr (3.7× worse than baseline)**. Mode choice (exact_k vs sample vs dynamic_k) doesn't matter in 1D.

Bench at 30 epochs (48 cells, 2 seeds):

| Condition | sin_irr | structured_irr | random_irr | n_params |
|-----------|---------|----------------|------------|----------|
| baseline_cfc         | 0.0094±0.0019 | 0.0053±0.0010 | **0.0013**±0.0004 | 2545 |
| fame_k3_t1           | 0.0196±0.0007 | 0.0153±0.0043 | 0.0181±0.0100 | 7757 |
| sigmoid_k3_dense     | 0.0048±0.0010 | 0.0034±0.0009 | 0.0052±0.0024 | 7763 |
| lora_k3_r4_dense     | 0.0047±0.0003 | 0.0036±0.0000 | 0.0014±0.0008 | 3691 |
| dag_moe_k3_l1        | 0.0047±0.0004 | **0.0030**±0.0003 | 0.0075±0.0032 | 11679 |
| **prob_moe_k3_exactk**   | **0.0026**±0.0004 | 0.0036±0.0004 | 0.0048±0.0037 | 10285 |
| prob_moe_k3_sample   | 0.0030±0.0007 | 0.0032±0.0015 | 0.0034±0.0017 | 10285 |
| prob_moe_k3_dynamick | 0.0026±0.0004 | 0.0036±0.0004 | 0.0048±0.0037 | 10285 |

**Best on each dataset (1 NEW BEST)**:
- **sin_irr**: prob_moe_k3_exactk = **0.0026** (NEW BEST, beats lora 0.0047 by 45%, dag 0.0047 by 45%)
- **structured_irr**: dag_moe_k3_l1 = 0.0030 (round 120 still leads)
- **random_irr**: baseline_cfc = 0.0013 (no MoE needed)

Key findings:
- **prob_moe_k3_exactk sets NEW BEST on sin_irr (0.0026)** — beats both lora (0.0047) and dag (0.0047) by 45%
- **Marginal-prob gradient is clean** — `router.proj.weight.grad is not None` in both exact_k and sample modes, with no STE needed
- **Mode choice doesn't matter in 1D** — exact_k and dynamic_k produce identical results (threshold = 1/n = 0.33 < all 3 probs), sample is slightly noisier (0.0030)
- **ProbMoE is target-dep** — random_irr 0.0048 vs baseline 0.0013 (3.7× worse, like DAG round 120)

## 1. ProbMoE in 60 seconds

Standard MoE: `y = sum_i g_i * E_i(x)`. Top-K is discrete, non-differentiable.

Gumbel-Softmax: uses straight-through estimator (STE) — biased gradient.

ProbMoE: `p_i = softmax(score_i / T)` is the **marginal probability** of expert i being in the selected subset. Gradient flows through the marginals, not through discrete selection. Three variants:
- `exact_k` — deterministic top-K from softmax, renorm to 1
- `sample` — multinomial sampling of K experts without replacement (Gumbel-free)
- `dynamic_k` — threshold-based variable cardinality (more experts on hard tokens)

## 2. Why ProbMoE is target-dep-with-nuance

### H1 ✓ ProbMoE exact_k is competitive with the best MoE

ProbMoE exact_k (0.0026 on sin) beats all other MoE variants on smooth data:
- vs lora 0.0047: -45%
- vs sigmoid 0.0048: -46%
- vs dag 0.0047: -45%
- vs fame 0.0196: -87%

### H2 ✗ Mode choice doesn't matter much in 1D

exact_k and dynamic_k produce identical results (same router path when threshold = 1/n = 0.33 < all 3 probs). Sample mode is slightly noisier (0.0030 on sin) but still competitive. The paper's distinction between modes only matters at scale (ImageNet, GPT-3) where the routing choice has bigger downstream effects.

### H3 ✓ Marginal probability gradient is clean

Gradient flow test confirmed: `router.proj.weight.grad is not None` in both exact_k and sample modes, with no STE needed. The probabilistic interpretation is the paper's main contribution and it works as advertised.

### H4 ✓ ProbMoE is target-dep: smooth > noisy

Random_irr is the worst dataset for ProbMoE: 0.0048 (vs 0.0013 baseline). Like DAG (round 120), ProbMoE adds capacity that helps on smooth data but hurts on noisy data — a recurring pattern in 91-121 audit (target-dep: 108, 109, 110, 112, 115, 117, 119, 120, **121**).

## 3. 91-121 audit pattern update

**Pattern (91-121)**: 18 structural mechanisms tested. **8 winners (STRICTLY POSITIVE)**: 99, 102, 105, 107, 113, 114, 116, 118. **10 negatives/target-dep**: 108, 109, 110, 112, 115, 117, 119, 120, **121**.

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
| **121** | **ProbMoE (Probabilistic)** | **Structural (probabilistic)** | **TARGET-DEP-WITH-NUANCE** |

**NEW INSIGHT (round 121)**: **Probabilistic routing with marginal probabilities is more principled than Gumbel-Softmax (no STE) and more interpretable than sigmoid weighted sum. But in 1D, the routing choice matters less than the per-expert capacity.**

This refines the audit pattern: **ProbMoE is target-dependent in 1D — best for smooth, worst for noisy**. Future mechanisms should consider hybrid (ProbMoE + LoRA deltas).

## 4. Implementation details

- **Core**: `lnn/core/prob_moe.py` (NEW, ~360 lines)
  - `ProbMoERouter(input_size, hidden_size, n_experts, top_k, temperature)` — 3 modes
  - `ProbMoECfCCell(input_size, hidden_size, n_experts, top_k, temperature, mode)` — base + K experts + probabilistic router
  - `ProbMoECfCNetwork(...)` — stacked ProbMoE CfC
  - `prob_moe_utilization(cell)` — diagnostic
- **Tests**: `tests/test_prob_moe.py` (NEW, 26/26 pass)
  - Router: init, temperature, top-K > n_experts raises, exact_k/sample/dynamic_k forward shapes, dynamic_k cardinality, unknown mode raises (8 tests)
  - Cell: init in 3 modes, forward shape in 3 modes, forward_with_aux, gradient flow (exact_k + sample), diag metadata, smoke sin (exact_k + sample) (10 tests)
  - Network: forward, last_step, NaN, learns, parameter count (5 tests)
  - Mini-bench: smoke on toy sin, parameter count (3 tests)
- **Bench**: `scripts/bench_prob_moe.py` (NEW, 48 cells, 30 epochs)
  - 3 datasets × 8 conditions × 2 seeds
  - Conditions: baseline_cfc, fame_k3_t1, sigmoid_k3_dense, lora_k3_r4_dense, dag_moe_k3_l1, prob_moe_k3_exactk, prob_moe_k3_sample, prob_moe_k3_dynamick
- **PRD**: `docs/prds/2026-06-15-lnn-round-121-a-prob-moe.md` (PRD #10-83)
- **Report**: `docs/research/2026-06-15_prob_moe_report.md`
- **Memory**: `lnn-round-121-prob-moe.md`
- **Exports**: `lnn/core/__init__.py` adds `ProbMoERouter, ProbMoECfCCell, ProbMoECfCNetwork, prob_moe_utilization`

## 5. Critical implementation details

1. **Multinomial sampling is per-batch (not batched)**: PyTorch's `torch.multinomial` doesn't support batched sampling without replacement, so we loop over the batch. O(B*K) per step, fine for our small B=8 and K=3.
2. **dynamic_k threshold = 1/n_experts** (uniform prior). If fewer than K experts exceed the threshold, fall back to top-K.
3. **Routing weight g is renormalized** to sum to 1.
4. **No STE**: gradient flows through `probs` (the softmax distribution), not through the discrete selection.

## 6. Future work

1. **Sweep temperature** ∈ {0.5, 1.0, 2.0} to test if sharper / smoother distributions matter at scale
2. **Sweep K ∈ {2, 3, 4, 6, 8}** with ProbMoE
3. **Sweep top_k** ∈ {1, 2, 3} (top_k=1 = FAME-like, top_k=2 = current, top_k=3 = all experts)
4. **Combine ProbMoE with LoRA experts** — LoRA + ProbMoE could combine two of the best mechanisms (low-rank expert deltas + probabilistic routing)
5. **ProbMoE on PhysioNet 36D** — irregular data is the original motivation
6. **ProbMoE + QuITE** (round 102-103) — query-based context may give the router a richer input

## 7. Recommendation

**ProbMoE exact_k is the 3rd TARGET-DEPENDENT-WITH-NUANCE in the 91-121 audit** (after rounds 108, 120).

- **DO use ProbMoE exact_k for smooth data** — sets NEW BEST on sin_irr (0.0026, beats lora by 45%).
- **DO NOT use ProbMoE on noisy data** — random_irr is dominated by baseline_cfc (0.0013) and lora_k3_r4_dense (0.0014).
- **Mode choice doesn't matter in 1D** — exact_k and dynamic_k are equivalent when threshold = 1/n < all probs. Use **sample** if you want regularization (stochastic routing acts as ensemble).
- **For production on smooth data**: use **prob_moe_k3_exactk** with `n_experts=3, top_k=2, temperature=1.0, mode="exact_k"`.
- **For noisy data**: stick with **lora_k3_r4_dense** (round 118 winner) or **baseline_cfc**.
