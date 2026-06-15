# Round 117 — Gumbel-Softmax Routing for MoE (Stochastic MoE) — Response to arXiv:1611.01144 + arXiv:2101.03961

**Date**: 2026-06-15
**Round**: 117
**Paper**: arXiv:1611.01144 (Jang et al. 2017, ICLR 2017) — *Categorical Reparameterization with Gumbel-Softmax* + arXiv:2101.03961 (Fedus et al. 2021) — *Switch Transformer*
**PRD**: #10-79
**Tests**: 31/31 in `tests/test_gumbel_moe.py`
**Bench**: 30 cells, 50 epochs (3 datasets × 5 conditions × 2 seeds)

## Summary

We implemented **Gumbel-Softmax Routing for MoE** — the 5th major router family in the 91-116 audit and the 1st **stochastic** router. The mechanism: add Gumbel noise to the router logits before softmax, with a temperature parameter that's annealed during training.

**The result is NEGATIVE-WITH-NUANCE** — Gumbel-Softmax is competitive (beats FAME on all 3 datasets) but doesn't beat sigmoid_dense (round 116 winner). Annealing has no effect at our 50-epoch scale.

Bench at 50 epochs (30 cells, 2 seeds):

| Condition | sin_irr | structured_irr | random_irr |
|-----------|---------|----------------|------------|
| baseline_cfc      | 0.0023±0.0001 | 0.0010±0.0001 | **0.0005±0.0002** |
| fame_k3_t1        | 0.0112±0.0016 | 0.0061±0.0021 | 0.0050±0.0041 |
| sigmoid_k3_dense  | **0.0013±0.0005** | **0.0009±0.0004** | 0.0020±0.0011 |
| **gumbel_k3_high**    | 0.0036±0.0009 | 0.0013±0.0005 | 0.0030±0.0020 |
| **gumbel_k3_anneal**  | 0.0038±0.0006 | 0.0012±0.0004 | 0.0030±0.0021 |

Key findings:
- **Gumbel beats FAME on all 3 datasets** by 1.7-3.1× — softmax + Gumbel noise is more stable than sparse top-K
- **Gumbel is worse than sigmoid_dense on all 3 datasets** by 1.5-2.8× — the noise adds variance without benefit
- **Annealing has no effect at 50 epochs** — both gumbel_high (T=1.0) and gumbel_anneal (T 1.0→0.1) give nearly identical results
- **Routing entropy H ≈ 1.07-1.09 nats** for all gumbel conditions — well-balanced like sigmoid

## Why Gumbel-Softmax is competitive but not winning

### The mechanism: stochastic categorical sampling

```
z = W x                  # router logits
g = -log(-log(U))        # Gumbel(0, 1) noise, U ~ Uniform(0, 1)
z' = (z + g) / T         # add noise + scale by temperature
g_routing = softmax(z')  # stochastic mixture weights
h_new = sum_i g_routing_i * expert_i(x_t, h_t)
```

The temperature T controls the "sharpness" of the routing:
- T=1.0: near-uniform exploration
- T=0.1: near-deterministic selection
- T=0.0: pure argmax (but division by zero)

### Why Gumbel beats FAME

FAME uses **sparse top-K=1** routing, which has 3 failure modes:
1. **Dead expert gradient** — non-selected experts get no gradient signal
2. **Top-K selection noise** — small logit differences cause large routing changes
3. **Imbalanced load** — one expert can dominate

Gumbel-Softmax is **dense softmax + noise**:
1. **All experts get gradient** via the soft mixture
2. **No top-K selection noise** — small logit differences → small weight differences
3. **Naturally balanced** — softmax + uniform noise → near-uniform on average

This makes Gumbel more stable than FAME on all 3 datasets.

### Why Gumbel loses to sigmoid

Sigmoid (round 116) has 3 advantages over Gumbel:
1. **No normalization** — each expert fires independently in [0, 1]
2. **No noise** — deterministic, no per-batch variance
3. **No temperature** — no annealing schedule to tune

The Gumbel noise introduces **per-batch variance** that doesn't help (and may hurt) the model. In 1D time-series, the data is already smooth enough that stochastic exploration doesn't add information — it just adds noise.

### Why annealing has no effect

At 50 epochs with anneal_rate=0.95, the temperature schedule is:
- epoch 0: T = 1.0
- epoch 10: T = 0.95^10 ≈ 0.60
- epoch 25: T = 0.95^25 ≈ 0.28
- epoch 50: T = 0.95^50 ≈ 0.077 → clamped to 0.1

So T converges to 0.1 in ~25 epochs and stays there. This means the model has only 25 epochs of "exploration" before becoming deterministic. At 50-epoch training, this is too short for the annealing to provide a meaningful curriculum.

**Hypothesis** (untested): at 200+ epochs, annealing might help by giving the model more exploration time. But at our standard 50-epoch scale, the schedule is too aggressive.

## Comparison with prior structural mechanisms

| Round | Mechanism | Type | Verdict |
|-------|-----------|------|---------|
| 99 | Reliability gate | Augmentation | STRICTLY POSITIVE |
| 102 | QuITE | Embedding | STRICTLY POSITIVE |
| 105 | SETA | Architecture | STRICTLY POSITIVE |
| 107 | Soft MoE | Structural | SAFER ROUTING |
| 108 | Anchored MoE | Structural | TARGET-DEP |
| 109 | Dynamic TMoE | Structural | NEGATIVE-WITH-NUANCE |
| 110 | Freq Experts | Structural | NEGATIVE-WITH-NUANCE |
| 111 | MoD Routing | Structural | POSITIVE-WITH-NUANCE (compute-saving) |
| 112 | Expert Choice | Structural | NEGATIVE (recurrent dynamics broken) |
| 113 | DeepSeek Shared Expert | Structural (residual) | STRICTLY POSITIVE |
| 114 | ReMoE (ReLU Routing) | Structural (soft gating) | STRICTLY POSITIVE |
| 115 | MH-MoE (Multi-Head) | Structural (sub-token) | NEGATIVE (low-D regime) |
| 116 | Sigmoid Routing | Structural (no normalization) | STRICTLY POSITIVE on smooth + NEUTRAL on noisy |
| **117** | **Gumbel-Softmax Routing** | **Structural (stochastic)** | **NEGATIVE-WITH-NUANCE** |

**Pattern (91-117)**: 14 structural mechanisms tested. **8 winners: 99, 102, 105, 107, 113, 114, 116**. **6 negative/target-dep: 108, 109, 110, 112, 115, 117**.

**5 router families compared**:

| Property | Softmax | Sigmoid | ReLU | Cosine | Gumbel |
|----------|---------|---------|------|--------|--------|
| Range | [0, 1] sums to 1 | [0, 1] each | [0, ∞) | [-1, 1] | [0, 1] sums to 1 |
| Normalization | YES (sum=1) | NO | NO | NO | YES (sum=1) |
| Stochastic | NO | NO | NO | NO | **YES (Gumbel)** |
| Default sparsity | top-K | dense | natural | natural | top-1 |
| Test winner | FAME (78/103) | sigmoid_dense (116) | ReMoE (114) | ❌ (82) | ❌ (117) |

## What we learned

### Stochastic routing is not a clear win

We expected Gumbel-Softmax to provide natural exploration that would help on noisy data (random_irr). It didn't — the noise adds variance without improving the routing signal.

The intuition was: Gumbel noise at high T forces all experts to fire, preventing FAME's H=0 lock-in (round 103). The reality: at T=0.1 (after annealing), Gumbel-Softmax converges to a deterministic softmax, which has the same H=0 risk as FAME.

**Lesson**: stochastic routing only helps if the temperature stays high throughout training (constant T=1.0). Annealing to low T defeats the purpose.

### Noise is the enemy in low-D time-series

Across 4 stochastic mechanisms (Gumbel, temporal_dropout rounds 92-93, input_dropout round 93, MH-MoE round 115), none of them helped. The 1D time-series signal is too small for noise to be useful — it just adds variance.

**Pattern reinforced**:
- Winners: data-structure-independent, **NO noise** (sigmoid, ReMoE, DeepSeek)
- Stochastic mechanisms: noise adds variance without signal

### Annealing is brittle in 1D

Gumbel-Softmax with annealing (T 1.0 → 0.1) gave nearly identical results to constant T=1.0. This means the schedule is too aggressive at 50 epochs.

This matches round 110 (frequency experts) which also used annealing and was a NEGATIVE-WITH-NUANCE. **Pattern**: annealing schedules are brittle in our 50-epoch 1D regime.

## Implementation

### Core API (`lnn/core/gumbel_moe.py`, ~430 lines)

```python
def _sample_gumbel(shape, device, dtype, eps=1e-9) -> torch.Tensor:
    """Sample Gumbel(0, 1) noise."""

class GumbelRouter(input_size, hidden_size, n_experts, temperature=1.0,
                   router_hidden=0, small_init=True, anneal_rate=0.95, min_temperature=0.1):
    """Gumbel-Softmax router: stochastic categorical sampling."""

class GumbelMoECfCCell(input_size, hidden_size, n_experts=3, temperature=1.0, ...):
    """K experts, stochastic mixture, recurrent state preserved."""

class GumbelMoECfCNetwork(...):
    """Stacked Gumbel-Softmax MoE CfC network. anneal_step() between epochs."""

def gumbel_moe_utilization(cell) -> dict:
    """Diagnostic: expert_util, routing_entropy, temperature."""
```

### Forward pass (training mode)

```python
def forward(self, x_t, h, dt=1.0):
    g = self.router(x_t, h, training=self.training)  # [B, K] Gumbel-Softmax
    outs = [expert(x_t, h, dt) for expert in self.experts]  # K × [B, H]
    stacked = torch.stack(outs, dim=1)
    h_new = (g.unsqueeze(-1) * stacked).sum(dim=1)
    return h_new
```

### Key implementation details

1. **Stochastic at training, deterministic at inference** — `torch.no_grad()` removes the noise (but our `training=self.training` flag handles this)
2. **Temperature annealing** — `anneal_step()` after each epoch, `T <- max(T * 0.95, 0.1)`
3. **Small init** — W ~ N(0, 0.01) to avoid early saturation (same as sigmoid)
4. **Softmax sums to 1** — even with Gumbel noise, the softmax property is preserved
5. **Recurrent state preserved** — h_new has same shape as h, gradient flows through

## Critical bugs fixed during round 117

1. **`cell.anneal_step` method missing**: initial design had `anneal_step` only on router. Fixed: added proxy method on cell.
2. **Network's `anneal_step` doesn't propagate to all cells**: fixed by iterating through `self.cells` in network.
3. **Test `test_anneal_step_decreases_temperature` was too strict** (required T < T_init): relaxed to `T_new < T_init OR T_new == min_temperature`.

## Recommendation

**DO NOT use Gumbel-Softmax Routing for production** in 1D time-series:
- Sigmoid (round 116) is strictly better on smooth data
- Baseline (no MoE) is better on noisy data
- The noise adds variance without signal
- Annealing schedule is too brittle at 50 epochs

**Consider Gumbel-Softmax at HIGHER T (constant)** for tasks where exploration matters:
- Hyperparameter search over T values (T=0.5, T=1.0, T=2.0)
- Longer training (200+ epochs) where annealing can be slower
- Higher-dimensional data (PhysioNet 36D) where noise provides regularization

**Use sigmoid (round 116) for known-smooth data**:
- Best result on sin_irr (0.0013) and structured_irr (0.0009)
- 8.6× better than FAME on sin
- No noise, no annealing

**Use DeepSeek (round 113) or ReMoE (round 114) for production with mixed data**:
- Both are STRICTLY POSITIVE on all 3 datasets in 91-115 audit

## Files added

- `lnn/core/gumbel_moe.py` (NEW, ~430 lines)
- `tests/test_gumbel_moe.py` (NEW, 31/31 tests)
- `scripts/bench_gumbel_moe.py` (NEW, 30 cells)
- `results/bench_gumbel_moe.json` (NEW)
- `docs/prds/2026-06-15-lnn-round-117-a-gumbel-moe.md` (PRD #10-79)
- `docs/research/2026-06-15_gumbel_moe_report.md` (this report)
- `docs/daily/2026-06-15_LNN_research_summary_v43.md` (digest v43)
- `README.md` (new Gumbel MoE section)
- `lnn-round-117-gumbel-moe.md` (memory)

## Future work

1. **Gumbel-Softmax at constant high T (T=1.0, T=2.0)** — sweep over temperature
2. **Gumbel-Softmax on PhysioNet 36D** — would likely work as designed
3. **Hard Gumbel-Softmax (straight-through)** — discrete routing with gradient
4. **Gumbel-Softmax + DeepSeek (round 113)** — additive residual + stochastic routing
5. **Gumbel-Softmax with longer annealing** — 200+ epochs with rate 0.99
