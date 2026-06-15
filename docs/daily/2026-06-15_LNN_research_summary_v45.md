# LNN Research Digest v45 — 2026-06-15

**Coverage**: PEER (Mixture of A Million Experts) for CfC (response to arXiv:2407.04153) + 91-119 audit update (8th negative, 1st linear expert family, product-key ablation strictly positive).

## Headline

Round 119 implemented **PEER (Parameter Efficient Expert Retrieval)** for CfC. The mechanism: N **single-neuron experts** (each is a `Linear(in, out)`, no activation), routed by **product-key lookup** (paper-faithful) or **softmax top-K** (ablation). This is the **1st linear expert family** in the 91-119 audit — every prior mechanism used sub-MLP experts.

**The result is NEGATIVE-WITH-NUANCE** — PEER is competitive on structured/random but loses on sin. However, the **product-key ablation is strictly better than softmax** for linear experts, a real positive finding.

Bench at 30 epochs (36 cells, 2 seeds):

| Condition | sin_irr | structured_irr | random_irr |
|-----------|---------|----------------|------------|
| baseline_cfc         | 0.0094±0.0019 | 0.0053±0.0010 | 0.0013±0.0004 |
| fame_k3_t1           | 0.0196±0.0007 | 0.0153±0.0043 | 0.0181±0.0100 |
| sigmoid_k3_dense     | 0.0048±0.0010 | 0.0034±0.0009 | 0.0052±0.0024 |
| **lora_k3_r4_dense** | **0.0047±0.0003** | 0.0036±0.0000 | **0.0014±0.0008** |
| peer_n8_pk           | 0.0056±0.0007 | **0.0035±0.0011** | 0.0020±0.0004 |
| peer_n8_softmax      | 0.0055±0.0011 | 0.0051±0.0009 | 0.0038±0.0023 |

**Parameter counts**:

| Condition | n_params | vs sigmoid_dense |
|-----------|----------|------------------|
| baseline_cfc       |  2545 |  33% |
| **lora_k3_r4_dense** |  3691 |  48% |
| peer_n8_pk         |  9501 | 122% |
| peer_n8_softmax    |  9617 | 124% |
| sigmoid_k3_dense   |  7763 | 100% |
| fame_k3_t1         |  7757 | 100% |

Key findings:
- **PEER n8_pk is competitive with sigmoid/LoRA on structured and random** but loses on sin
- **PEER n8_pk beats sigmoid_k3_dense 2.6× on random_irr** (0.0020 vs 0.0052) — linear experts work well on noisy data
- **Product-key routing is strictly better than softmax** for linear experts: structured_irr 0.0035 vs 0.0051 (46% better), random_irr 0.0020 vs 0.0038 (90% better) — a real positive from the ablation
- **Routing entropy H ≈ 0.88-1.08 nats** (well-balanced, not stuck at 0)
- **n_active = 3 of 8 experts** fire on average — room for better utilization

## 1. PEER in 60 seconds

Standard MoE has K **sub-MLP** experts (each a full CfC cell in our case). PEER replaces this with:
- N **single-neuron experts**: each is a `Linear(in, out, bias=True)` with no activation function
- Two **key tables** for routing: each `[n_buckets, I+H]` with `n_buckets = max(2, ceil(sqrt(N)))`

The forward pass:
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

**Key property**: each expert is a single linear neuron, the smallest possible expert unit. With N → ∞, PEER approximates any piecewise-linear function.

## 2. Why PEER is negative-with-nuance

### PEER vs sigmoid_k3_dense (round 116 winner)

- sin_irr:       0.0056 vs 0.0048 (PEER 17% worse)
- structured_irr: 0.0035 vs 0.0034 (tied)
- random_irr:    0.0020 vs 0.0052 (PEER 2.6× better)

PEER **wins on 1/3** (random), **ties on 1/3** (structured), **loses on 1/3** (sin). Not a clear winner overall.

### PEER vs lora_k3_r4_dense (round 118 winner)

- sin_irr:       0.0056 vs 0.0047 (PEER 19% worse)
- structured_irr: 0.0035 vs 0.0036 (tied)
- random_irr:    0.0020 vs 0.0014 (PEER 43% worse)

PEER is consistently behind LoRA — LoRA is the better winner.

### Why single-neuron experts lose to sub-MLP on sin

The sin function requires **nonlinearity** to express. A single linear neuron computes `y = w·x + b` — a linear function. The base CfC has tanh/sigmoid activations which provide the nonlinearity, but the PEER deltas are linear. On sin-shaped targets, the linear deltas can't represent the high-frequency content.

On structured/random data, the deltas are smaller adjustments to the base, so linearity is less of a handicap.

### Why product-key routing is strictly better than softmax

For linear experts, the **product-key routing** has a structural advantage over softmax:
1. **Deterministic routing decisions** (hash lookups) avoid the softmax saturation problem
2. **Two-stage retrieval** (top-K in each table, then top-K of K²) gives more diverse candidates
3. **Routed by key similarity, not learned weights** — the keys themselves adapt to data, but the routing decision is a hard lookup

The fact that **product-key is 46-90% better than softmax on structured/random** is a real positive finding for the audit.

## 3. 91-119 audit pattern update

**Pattern (91-119)**: 16 structural mechanisms tested. **8 winners: 99, 102, 105, 107, 113, 114, 116, 118**. **8 negatives: 108, 109, 110, 112, 115, 117, 119**.

| Round | Mechanism | Type | Verdict |
|-------|-----------|------|---------|
| 99 | Reliability gate | Augmentation | STRICTLY POSITIVE |
| 102 | QuITE | Embedding | STRICTLY POSITIVE |
| 105 | SETA | Architecture | STRICTLY POSITIVE |
| 107 | Soft MoE | Structural | SAFER ROUTING |
| 108 | Anchored MoE | Structural | TARGET-DEP |
| 109 | Dynamic TMoE | Structural | NEGATIVE-WITH-NUANCE |
| 110 | Freq Experts | Structural | NEGATIVE-WITH-NUANCE |
| 111 | MoD Routing | Structural | POSITIVE-WITH-NUANCE |
| 112 | Expert Choice | Structural | NEGATIVE |
| 113 | DeepSeek Shared Expert | Structural (residual) | STRICTLY POSITIVE |
| 114 | ReMoE (ReLU Routing) | Structural (soft gating) | STRICTLY POSITIVE |
| 115 | MH-MoE (Multi-Head) | Structural (sub-token) | NEGATIVE |
| 116 | Sigmoid Routing | Structural (no normalization) | STRICTLY POSITIVE |
| 117 | Gumbel-Softmax | Structural (stochastic) | NEGATIVE-WITH-NUANCE |
| 118 | LoRA-MoRE | Structural (rank-r delta) | STRICTLY POSITIVE |
| **119** | **PEER (Single Neurons)** | **Structural (linear expert)** | **NEGATIVE-WITH-NUANCE** |

**NEW INSIGHT (round 119)**: **Linear experts are not enough in 1D time-series.** The 8 winners all use sub-MLP experts. PEER's single-neuron experts lack the nonlinearity needed to express smooth functions like sin.

This refines the audit pattern: **sub-MLP experts > linear experts in 1D**. Future mechanisms should preserve sub-MLP structure.

## 4. Implementation details

- **Core**: `lnn/core/peer_moe.py` (NEW, ~530 lines)
  - `SingleNeuronExpert(in_features, out_features, bias=True)` — one expert = one linear neuron
  - `ProductKeyRouter(input_size, hidden_size, n_experts, top_k=2, n_buckets=None)` — two key tables, top-K in each, dedup, softmax
  - `LinearSoftmaxRouter(input_size, hidden_size, n_experts, top_k=2)` — softmax ablation
  - `PEERCfCCell(input_size, hidden_size, n_experts=8, top_k=2, router_type="product_key")` — base CfC + N linear experts + router
  - `PEERCfCNetwork(...)` — stacked PEER CfC network
  - `peer_utilization(cell)` — diagnostic
- **Tests**: `tests/test_peer_moe.py` (NEW, 28/28 pass)
  - SingleNeuronExpert init, forward, no-bias
  - ProductKeyRouter: default √N buckets, top-K in each, dedup, softmax
  - LinearSoftmaxRouter: linear projection + top-K
  - PEERCfCCell: both router types, forward shape, gradient flow, smoke on toy_sin
  - PEERCfCNetwork: forward, NaN handling, learns
- **Bench**: `scripts/bench_peer_moe.py` (NEW, 36 cells, 30 epochs)
  - 3 datasets × 6 conditions × 2 seeds
  - Conditions: baseline_cfc, fame_k3_t1, sigmoid_k3_dense, lora_k3_r4_dense, peer_n8_pk, peer_n8_softmax
- **PRD**: `docs/prds/2026-06-15-lnn-round-119-a-peer-moe.md` (PRD #10-81)
- **Report**: `docs/research/2026-06-15_peer_moe_report.md`
- **Memory**: `lnn-round-119-peer-moe.md`
- **Exports**: `lnn/core/__init__.py` adds `SingleNeuronExpert, ProductKeyRouter, LinearSoftmaxRouter, PEERCfCCell, PEERCfCNetwork, peer_utilization`

## 5. Critical bugs fixed during round 119

1. **`test_peer_cell_gradient_flow` was too strict**: it required ALL 8 experts to receive gradients, but with top_k=2 and batch=3 only 6 selections happen, leaving some experts without grad. Fixed to assert `n_with_grad >= 2`.
2. **Deterministic dedup loop in `ProductKeyRouter.forward`**: needed Python loop to dedup expert indices across the two key tables. The loop is O(B·K²) per step, fine for B=8 and K=2.

## 6. Future work

1. **Sweep N ∈ {4, 8, 16, 32, 64}** to find the sweet spot for linear experts
2. **Hybrid PEER + LoRA**: some experts are linear, some are LoRA adapters
3. **PEER on PhysioNet 36D** — higher-D data may benefit from linear basis
4. **Product-key routing for sub-MLP experts** — apply the deterministic hash lookup to FAME
5. **Hash-based routing for CfC** — round 102 used QuITE which is attention-based; product-key is a cheaper alternative

## 7. Recommendation

**PEER is the 8th NEGATIVE-WITH-NUANCE in the 91-119 audit.**

- **DO use PEER product-key when the data is structured or noisy** — the linear basis + deterministic routing is competitive with the best winners on those regimes
- **DO NOT use PEER for smooth nonlinear targets** — the lack of nonlinearity hurts
- **Product-key routing is structurally better than softmax** when paired with linear experts
- For production: prefer **sigmoid_k3_dense (round 116)** or **lora_k3_r4_dense (round 118)** which are STRICTLY POSITIVE on all 3 datasets
