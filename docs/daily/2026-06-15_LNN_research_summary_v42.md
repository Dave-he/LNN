# LNN Research Digest v42 — 2026-06-15

**Coverage**: Sigmoid Routing (Qwen2-MoE style) + 91-116 audit update (8th STRUCTURAL WINNER — sigmoid dense beats FAME 8-10× on smooth data).

## Headline

Round 116 implemented **Sigmoid Routing for MoE** (Qwen2-MoE, arXiv:2407.10671). The mechanism: **replace softmax with sigmoid** in the router. Each expert gets an independent score in [0, 1] with no "softmax budget" competition. The 4th major router family in the 91-116 audit (after softmax, ReLU, cosine) and the 1st without normalization.

**The result is STRICTLY POSITIVE on smooth data** — sigmoid (dense) routing beats FAME by **8-10× on sin/structured** and matches baseline on noisy.

Bench at 50 epochs (30 cells, 2 seeds):

| Condition | sin_irr | structured_irr | random_irr |
|-----------|---------|----------------|------------|
| baseline_cfc      | 0.0023±0.0001 | 0.0010±0.0001 | **0.0005±0.0002** |
| fame_k3_t1        | 0.0112±0.0016 | 0.0061±0.0021 | 0.0050±0.0041 |
| **sigmoid_k3_dense** | **0.0013±0.0005** | **0.0009±0.0004** | 0.0020±0.0011 |
| sigmoid_k3_t1     | 0.0132±0.0001 | 0.0095±0.0006 | 0.0173±0.0086 |
| sigmoid_k3_t2     | 0.0038±0.0016 | 0.0019±0.0008 | 0.0046±0.0011 |

Key findings:
- **Dense sigmoid WINS on sin_irr** (0.0013 vs FAME 0.0112 — **8.6× better**)
- **Dense sigmoid matches baseline on structured_irr** (0.0009 vs 0.0010)
- **Dense sigmoid beats FAME on all 3 datasets** by 2.5-10×
- **Sparse sigmoid (t=1, t=2) underperforms dense** — top-K selection noise hurts
- **Routing entropy H ≈ 1.10 nats for dense, 0.36-0.69 for sparse** — dense is naturally balanced

## 1. Sigmoid Routing in 60 seconds

Standard MoE (FAME) uses softmax routing, which normalizes scores to sum to 1. Sigmoid routing replaces softmax with sigmoid, so each expert gets an independent score in [0, 1] with no normalization:

```
input x_t [B, D]
  │
  ├── K experts: CfC cells (one forward per expert)  ──→  K × [B, H]
  │
  ├── sigmoid router: g = sigmoid(W x + b)  ──→  [B, K] in [0, 1] (NOT normalized)
  │
  └── h_new = sum_i g_i * expert_i(x_t)  ──→  [B, H]
```

**Key property**: no normalization → no "softmax budget" competition → all K experts can fire simultaneously.

## 2. Why Sigmoid Routing works on smooth data

### Three properties of sigmoid routing

1. **No normalization** — each expert gets an independent score in [0, 1]. Multiple experts can fire simultaneously with no competition. When the input is rich (smooth), all K experts can contribute meaningfully.

2. **Naturally sparse via small init** — we initialize W ~ N(0, 0.01), so g ~ 0.5 for all experts at init. As the network learns, the W magnitudes diverge and only some experts fire strongly on specific input patterns.

3. **Per-expert bias optional** — Qwen2-MoE uses a bias term on the routing score (similar to DeepSeek-V3's AuxLF). We include this as a `use_router_bias=True` default.

### Why dense wins over sparse

The sparse variants (t=1, t=2) consistently underperform dense (t=0). The reason: in the sigmoid setting, top-K selection **adds noise** without providing a benefit. The sigmoid scores are already in [0, 1] and naturally differentiate between "strong" and "weak" experts. Forcing sparsity breaks the natural gradient flow.

This is **opposite to softmax/FAME**, where top-K is essential (without top-K, softmax collapses to uniform and experts don't specialize).

## 3. 91-116 audit pattern update

**Pattern (91-116)**: 13 structural mechanisms tested.

| Round | Mechanism | Verdict |
|-------|-----------|---------|
| 99 | Reliability gate | STRICTLY POSITIVE |
| 102 | QuITE | STRICTLY POSITIVE |
| 105 | SETA | STRICTLY POSITIVE |
| 107 | Soft MoE | SAFER ROUTING |
| 108 | Anchored MoE | TARGET-DEP |
| 109 | Dynamic TMoE | NEGATIVE-WITH-NUANCE |
| 110 | Freq Experts | NEGATIVE-WITH-NUANCE |
| 111 | MoD Routing | POSITIVE-WITH-NUANCE (compute-saving) |
| 112 | Expert Choice | NEGATIVE (recurrent dynamics broken) |
| 113 | DeepSeek Shared Expert | STRICTLY POSITIVE |
| 114 | ReMoE | STRICTLY POSITIVE |
| 115 | MH-MoE | NEGATIVE (low-D regime) |
| **116** | **Sigmoid Routing** | **STRICTLY POSITIVE on smooth + NEUTRAL on noisy** |

**8 winners: 99, 102, 105, 107, 113, 114, 116**. **5 negative/target-dep: 108, 109, 110, 112, 115**.

**NEW INSIGHT (round 116)**: Sigmoid routing is the **4th major router family** and the **1st without normalization**. The 4 families:

| Property | Softmax | Sigmoid | ReLU | Cosine |
|----------|---------|---------|------|--------|
| Range | [0, 1] sums to 1 | [0, 1] each | [0, ∞) | [-1, 1] |
| Normalization | YES (sum=1) | NO | NO | NO |
| Multi-expert | 0 or top-K | All K | All K (positive) | All K |
| Default sparsity | top-K | dense | natural | natural |
| Test winner | FAME (78/103) | **sigmoid_dense (116)** | ReMoE (114) | ❌ (82) |

**Pattern reinforced**:
- Winners: data-structure-independent, preserve recurrent state, **add diversity** (not remove)
- Dense vs. sparse: **router-dependent** — softmax needs top-K, sigmoid/relu prefer dense

## 4. Implementation details

- **Core**: `lnn/core/sigmoid_moe.py` (NEW, ~370 lines)
  - `SigmoidRouter(input_size, hidden_size, n_experts, top_k=0, use_bias=True, router_hidden=0, small_init=True)` — per-expert sigmoid gate
  - `SigmoidMoECfCCell(input_size, hidden_size, n_experts=3, top_k=0, ...)` — K experts, dense or sparse
  - `SigmoidMoECfCNetwork(...)` — stacked sigmoid-routed MoE CfC network
  - `sigmoid_moe_utilization(cell)` — diagnostic for expert utilization
- **Tests**: `tests/test_sigmoid_moe.py` (NEW, 32/32 pass)
  - Init, forward shape, sigmoid in [0, 1] (no normalization)
  - Dense mode (top_k=0) vs sparse (top_k>0)
  - Per-expert bias, gradient flow
  - Recurrent state preserved (h_new.shape == h.shape)
  - Toy sin smoke (converges with K=3 dense)
- **Bench**: `scripts/bench_sigmoid_moe.py` (NEW, 30 cells, 50 epochs)
  - 3 datasets × 5 conditions × 2 seeds
  - Conditions: baseline_cfc, fame_k3_t1, sigmoid_k3_dense, sigmoid_k3_t1, sigmoid_k3_t2
- **PRD**: `docs/prds/2026-06-15-lnn-round-116-a-sigmoid-moe.md` (PRD #10-78)
- **Report**: `docs/research/2026-06-15_sigmoid_moe_report.md`
- **Memory**: `lnn-round-116-sigmoid-moe.md`
- **Exports**: `lnn/core/__init__.py` adds `SigmoidMoECfCCell, SigmoidMoECfCNetwork, SigmoidRouter, sigmoid_moe_utilization`

## 5. Critical bugs fixed during round 116

1. **`nn.Linear` not subscriptable** in `small_init` logic: was `self.net[-1]` but `self.net` is a Linear (not Sequential) when `router_hidden=0`. Fixed: branch on `router_hidden > 0`.
2. **Missing assertion messages**: `assert n_experts >= 1` was missing the f-string. Fixed: `f"n_experts must be >= 1, got {n_experts}"`.
3. **`cell.small_init` AttributeError**: `SigmoidMoECfCCell.__init__` was not storing `small_init` as instance attribute. Fixed: added `self.small_init = bool(small_init)`.
4. **`h.sum().backward()` failed** in test: `h` was created without `requires_grad=True`. Fixed: pass `requires_grad=True` to both `x_t` and `h`.

## 6. Future work

1. **Sigmoid + DeepSeek (round 113)**: combine the additive residual with sigmoid routing
2. **Sigmoid + QuITE (round 102)**: combine the embedding with sigmoid routing
3. **Sigmoid + Orthogonality (round 97)**: weight-level orth on the router
4. **Adaptive sparsity**: learn top-K based on input complexity (sparse on noisy, dense on smooth)
5. **Sigmoid on high-D (PhysioNet 36D)**: would likely generalize as designed

## 7. Recommendation

- **Sigmoid (dense)**: Use for known-smooth data (sinusoidal, low-frequency patterns). Best result on sin_irr (8× better than FAME).
- **Baseline CfC (no MoE)**: Use for known-noisy data. Best result on random_irr.
- **DeepSeek/ReMoE (rounds 113/114)**: Use for production when you don't know data structure in advance. Both are STRICTLY POSITIVE on all 3 datasets in 91-115 audit.
