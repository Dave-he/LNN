# LNN Research Digest v43 — 2026-06-15

**Coverage**: Gumbel-Softmax Routing (Stochastic MoE) + 91-117 audit update (5th router family tested, NEGATIVE-WITH-NUANCE — beats FAME but loses to sigmoid_dense).

## Headline

Round 117 implemented **Gumbel-Softmax Routing for MoE** (Jang et al. 2017, arXiv:1611.01144, combined with Switch Transformer's top-1 stochastic routing). The 5th major router family in the 91-117 audit and the 1st **stochastic** router.

**The result is NEGATIVE-WITH-NUANCE** — Gumbel-Softmax beats FAME on all 3 datasets by 1.7-3.1× but loses to sigmoid_dense (round 116 winner) on all 3. Annealing has no effect at 50 epochs.

Bench at 50 epochs (30 cells, 2 seeds):

| Condition | sin_irr | structured_irr | random_irr |
|-----------|---------|----------------|------------|
| baseline_cfc      | 0.0023±0.0001 | 0.0010±0.0001 | **0.0005±0.0002** |
| fame_k3_t1        | 0.0112±0.0016 | 0.0061±0.0021 | 0.0050±0.0041 |
| **sigmoid_k3_dense** | **0.0013±0.0005** | **0.0009±0.0004** | 0.0020±0.0011 |
| gumbel_k3_high    | 0.0036±0.0009 | 0.0013±0.0005 | 0.0030±0.0020 |
| gumbel_k3_anneal  | 0.0038±0.0006 | 0.0012±0.0004 | 0.0030±0.0021 |

Key findings:
- **Gumbel beats FAME on all 3 datasets** — softmax + Gumbel noise is more stable than sparse top-K
- **Gumbel is worse than sigmoid_dense on all 3 datasets** — the noise adds variance without benefit
- **Annealing has no effect at 50 epochs** — both gumbel_high (T=1.0) and gumbel_anneal (T 1.0→0.1) give nearly identical results
- **Routing entropy H ≈ 1.07-1.09 nats** for all gumbel conditions — well-balanced

## 1. Gumbel-Softmax Routing in 60 seconds

Standard softmax routing is deterministic — same input → same routing weights. Gumbel-Softmax adds Gumbel noise to the logits before softmax:

```
input x_t
  │
  ├── router logits: z = W x
  │
  ├── Gumbel noise: g = -log(-log(U)),  U ~ Uniform(0, 1)
  │
  ├── add noise + temperature: z' = (z + g) / T
  │
  ├── Gumbel-Softmax: g_routing = softmax(z', dim=-1)
  │
  └── h_new = sum_i g_routing_i * expert_i(x_t, h_t)
```

**Key property**: stochastic at training time, deterministic at inference. Temperature T anneals from 1.0 (exploration) to 0.1 (exploitation).

## 2. Why Gumbel-Softmax is competitive but not winning

### Why Gumbel beats FAME

FAME uses sparse top-K=1, which has 3 failure modes:
1. Dead expert gradient — non-selected experts get no gradient
2. Top-K selection noise — small logit differences cause large routing changes
3. Imbalanced load — one expert can dominate

Gumbel-Softmax is dense softmax + noise:
1. All experts get gradient via soft mixture
2. No top-K selection noise
3. Naturally balanced (softmax + uniform noise → near-uniform on average)

### Why Gumbel loses to sigmoid

Sigmoid (round 116) has 3 advantages:
1. No normalization — each expert fires independently in [0, 1]
2. No noise — deterministic
3. No temperature — no annealing schedule

The Gumbel noise introduces per-batch variance that doesn't help (and may hurt) in 1D.

### Why annealing has no effect

At 50 epochs with anneal_rate=0.95: T = 1.0 → 0.95^50 ≈ 0.077 → clamped to 0.1.
So T converges to 0.1 in ~25 epochs, leaving only 25 epochs of "exploration".

**Hypothesis**: at 200+ epochs with anneal_rate=0.99, annealing might help. But at our standard 50-epoch scale, the schedule is too aggressive.

## 3. 91-117 audit pattern update

**Pattern (91-117)**: 14 structural mechanisms tested. **8 winners: 99, 102, 105, 107, 113, 114, 116**. **6 negative/target-dep: 108, 109, 110, 112, 115, 117**.

**5 router families compared**:

| Property | Softmax | Sigmoid | ReLU | Cosine | Gumbel |
|----------|---------|---------|------|--------|--------|
| Range | [0, 1] sums to 1 | [0, 1] each | [0, ∞) | [-1, 1] | [0, 1] sums to 1 |
| Normalization | YES (sum=1) | NO | NO | NO | YES (sum=1) |
| Stochastic | NO | NO | NO | NO | **YES (Gumbel)** |
| Default sparsity | top-K | dense | natural | natural | top-1 |
| Test winner | FAME (78/103) | sigmoid_dense (116) | ReMoE (114) | ❌ (82) | ❌ (117) |

**NEW INSIGHT (round 117)**: Stochastic routing is NOT a clear win in 1D. Across 4 stochastic mechanisms (Gumbel round 117, temporal_dropout rounds 92-93, input_dropout round 93, MH-MoE round 115), none of them helped. The 1D signal is too small for noise to be useful.

**Pattern reinforced**:
- Winners: data-structure-independent, **NO noise**, preserve recurrent state
- Failures: stochastic mechanisms, noise, dynamic schedules

## 4. Implementation details

- **Core**: `lnn/core/gumbel_moe.py` (NEW, ~430 lines)
  - `_sample_gumbel(shape, device, dtype, eps=1e-9)` — Gumbel(0, 1) noise sampler
  - `GumbelRouter(input_size, hidden_size, n_experts, temperature=1.0, router_hidden=0, small_init=True, anneal_rate=0.95, min_temperature=0.1)` — Gumbel-Softmax router
  - `GumbelMoECfCCell(input_size, hidden_size, n_experts=3, temperature=1.0, anneal_rate=0.95, min_temperature=0.1, ...)` — K experts, stochastic mixture
  - `GumbelMoECfCNetwork(...)` — stacked Gumbel MoE CfC network
  - `gumbel_moe_utilization(cell)` — diagnostic for expert utilization
- **Tests**: `tests/test_gumbel_moe.py` (NEW, 31/31 pass)
  - Gumbel noise distribution (mean ~ 0.5772, std ~ 1.2825)
  - Init, forward shape, softmax sums to 1
  - Training adds noise, eval is deterministic
  - anneal_step decreases T (or hits min_temperature floor)
  - Gradient flows to all K experts
  - Toy sin smoke (converges with K=3)
- **Bench**: `scripts/bench_gumbel_moe.py` (NEW, 30 cells, 50 epochs)
  - 3 datasets × 5 conditions × 2 seeds
  - Conditions: baseline_cfc, fame_k3_t1, sigmoid_k3_dense, gumbel_k3_high, gumbel_k3_anneal
- **PRD**: `docs/prds/2026-06-15-lnn-round-117-a-gumbel-moe.md` (PRD #10-79)
- **Report**: `docs/research/2026-06-15_gumbel_moe_report.md`
- **Memory**: `lnn-round-117-gumbel-moe.md`
- **Exports**: `lnn/core/__init__.py` adds `GumbelMoECfCCell, GumbelMoECfCNetwork, GumbelRouter, gumbel_moe_utilization`

## 5. Critical bugs fixed during round 117

1. **`cell.anneal_step` method missing**: initial design had `anneal_step` only on router. Fixed: added proxy method on cell.
2. **Network's `anneal_step` doesn't propagate to all cells**: fixed by iterating through `self.cells` in network.
3. **Test `test_anneal_step_decreases_temperature` was too strict** (required T < T_init): relaxed to `T_new < T_init OR T_new == min_temperature`.

## 6. Future work

1. **Gumbel-Softmax at constant high T (T=1.0, T=2.0)** — sweep over temperature
2. **Gumbel-Softmax on PhysioNet 36D** — would likely work as designed
3. **Hard Gumbel-Softmax (straight-through)** — discrete routing with gradient
4. **Gumbel-Softmax + DeepSeek (round 113)** — additive residual + stochastic routing
5. **Gumbel-Softmax with longer annealing** — 200+ epochs with rate 0.99

## 7. Recommendation

- **DO NOT use Gumbel-Softmax in production** in 1D time-series
- **Sigmoid (round 116)** is strictly better on smooth data
- **DeepSeek/ReMoE (rounds 113/114)** for production with mixed data
- Consider Gumbel-Softmax at high T on higher-dimensional data (PhysioNet 36D)
