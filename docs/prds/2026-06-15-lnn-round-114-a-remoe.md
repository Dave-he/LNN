# PRD #10-76 — ReMoE (Fully Differentiable MoE with ReLU Routing) for CfC

**Date**: 2026-06-15
**Round**: 114
**Paper**: arXiv:2412.14711 (Wang, Zhu, Chen, December 2024, ICLR 2025) — *ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing*
**Codebase target**: `lnn/core/remoe.py`
**Test file**: `tests/test_remoe.py`
**Bench script**: `scripts/bench_remoe.py`

## Goal

Replace the **TopK+Softmax** sparse routing in our MoE stack with a **fully differentiable ReLU-based router** that has a natural sparsity bias (ReLU zeros out negative logits) and a load-balancing auxiliary loss. Test whether the cleaner gradient flow beats FAME and DeepSeek on the 91-113 audit pattern.

## Background

### What is ReMoE?

**Standard MoE (top-K routing)**:
1. Router: `g = softmax(W x)` — probability distribution over K experts
2. Selection: `top_k(g)` — keep only K' largest, zero out the rest (DISCONTINUOUS, non-differentiable)
3. Aux loss: encourage load balance (Switch Transformer-style)

**ReMoE (ReLU routing)**:
1. Router: `s = W x` — raw K-dim scores
2. ReLU gate: `g = ReLU(s)` — natural sparsity (negative scores → 0)
3. Output: `h = Σ_i g_i * expert_i(x)` — weighted combination, naturally sparse
4. Aux loss: load-balancing loss (e.g., `f * log(f)` where `f` is fraction of total mass per expert)

ReMoE's claim (paper, ICLR 2025):
- **Fully differentiable** routing — no top-K discontinuity, no straight-through estimator
- **Continuous allocation** of compute across tokens and layers
- **Domain specialization** emerges naturally
- **Scales better** with number of experts than top-K MoE

### Why is ReMoE a fit for the 91-113 audit pattern?

The audit pattern: mechanisms that **modify or constrain the recurrent state mixing** are dangerous (rounds 108-110, 112 negative; rounds 91-94 smoothness audit negative; round 99 Reliability Gate + 102 QuITE + 105 SETA + 107 Soft MoE + 113 DeepSeek positive because they preserve recurrent dynamics).

ReMoE **does NOT modify the recurrent dynamics**:
- Each expert is still a regular CfC cell with the same gate-and-update structure
- The router is an auxiliary `g = ReLU(W x)` that doesn't touch the recurrent state
- The combination `h = Σ_i g_i * expert_i(x)` is a **soft, smooth, fully-differentiable** weighted sum
- Unlike top-K routing, there's no hard zeroing → gradient flows to all experts
- Unlike Expert Choice (round 112), the combination is a weighted sum, not a per-expert mean (so no recurrent wash-out)

The structural change vs FAME:
- FAME: `top_k + softmax` — hard selection, partial gradient (only top-K get gradient)
- ReMoE: `ReLU` — soft gating, full gradient to all experts with non-zero gate

### Hypothesis

**H1 (STRICTLY POSITIVE)**: ReMoE beats FAME on the 3 standard datasets (sin/structured/random irregular) by 1.2-2× on test_mse because the full-differentiable routing gives better gradient flow.

**H2 (PARTIAL)**: ReMoE matches or slightly underperforms DeepSeek (round 113) because DeepSeek's **additive residual** structure is the strongest safe pattern, but ReMoE's smoothness may help on noisy data.

**H3 (PARTIAL)**: ReLU routing produces naturally sparse gates (most g_i = 0) and balanced utilization when combined with a load-balancing aux loss.

## Implementation

### Core API (`lnn/core/remoe.py`)

```python
class ReMoERouter(nn.Module):
    """ReLU-based router: g = ReLU(W x) (optionally with 2-layer MLP)."""

class ReMoECfCCell(nn.Module):
    """K experts + ReLU router, soft additive combination (no top-K)."""

class ReMoECfCNetwork(nn.Module):
    """Stacked ReMoE-style CfC network."""

def remoe_utilization(cell) -> dict:
    """Returns mean per-expert gate activation (proxy for utilization)."""

def remoe_load_balancing_loss(cell, target_per_expert=1.0/K) -> Tensor:
    """Aux loss: -Σ_i f_i * log(f_i / target) for balanced gates."""
```

### Forward pass

```python
def forward(self, x_t, h, dt=1.0):
    expert_outs = [expert(x_t, h, dt=dt) for expert in self.experts]  # K × [B, H]
    stacked = torch.stack(expert_outs, dim=1)                          # [B, K, H]
    s = self.router(x_t, h)                                           # [B, K]
    g = F.relu(s)                                                      # [B, K]
    self.last_g = g.detach()
    h_new = (g.unsqueeze(-1) * stacked).sum(dim=1)                    # [B, H]
    return h_new
```

### Key differences from FAME

| Aspect | FAME (round 78) | ReMoE (round 114) |
|--------|------------------|---------------------|
| Router output | `softmax(W x)` | `ReLU(W x)` |
| Sparsity | Hard top-K (straight-through) | Soft natural sparsity (ReLU) |
| Differentiability | Partial (top-K discontinuous) | Fully differentiable |
| Gradient flow | Only top-K experts get grad | All non-zero-gated experts get grad |
| Aux loss | Load balance (Switch style) | Load balance (`f * log f`) |
| Combination | Top-K weighted sum | Full weighted sum (most g_i = 0 anyway) |

## Tests (`tests/test_remoe.py`, ~20 tests)

- TestReMoERouterInit (3): init, shapes, no bias
- TestReMoERouterForward (4): ReLU zeros negatives, gradient flows, sparsity
- TestReMoECfCCellInit (3): K experts, with/without router_hidden, invalid K
- TestReMoECfCCellForward (6): forward shape, gradient flow to all experts, ReLU sparsity, last_g stash
- TestReMoELoadBalancing (2): aux loss shape, zero at uniform
- TestReMoECfCNetwork (4): init, forward dense, last-step, with mask
- TestReMoEDiagnostics (2): utilization no-forward, after-forward
- TestReMoESineSmoke (1): 4 experts converges on toy sin

## Bench (`scripts/bench_remoe.py`)

30 cells (3 datasets × 5 conditions × 2 seeds × 50 epochs):

| Cond | K | Router | Aux loss λ | Description |
|------|---|--------|-------------|-------------|
| `baseline_cfc`      | 1 | n/a   | n/a | Standard CfC (control) |
| `fame_k3_t1`        | 3 | top_k | 0.0 | FAME K=3 top_k=1 (round 78 baseline) |
| `remoe_k3`          | 3 | ReLU  | 0.0 | ReMoE K=3 no aux |
| `remoe_k3_lb001`    | 3 | ReLU  | 0.01 | ReMoE K=3 + load balance λ=0.01 |
| `remoe_k3_lb01`     | 3 | ReLU  | 0.1  | ReMoE K=3 + load balance λ=0.1 |

Datasets: `sin_irr`, `structured_irr`, `random_irr` (same as round 113 for direct comparison).

## Files added

- `lnn/core/remoe.py` (NEW, ~280 lines)
- `tests/test_remoe.py` (NEW, ~25 tests)
- `scripts/bench_remoe.py` (NEW, 30 cells)
- `docs/prds/2026-06-15-lnn-round-114-a-remoe.md` (PRD, this file)
- `docs/research/2026-06-15_remoe_report.md` (full report)
- `docs/daily/2026-06-15_LNN_research_summary_v40.md` (digest v40)
- `README.md` (new ReMoE section)
- `lnn-round-114-remoe.md` (memory)

## Why this is a structural + data-independent + DOESN'T modify recurrent state mixing

Structural change: the **router** is a `ReLU(W x)` instead of `top_k + softmax(W x)`.
Data-independent: ReLU's natural sparsity bias is intrinsic, not data-conditioned.
Doesn't modify recurrent state: each expert is still a CfC cell with standard gate-and-update; the router produces scalar gates that are applied to expert outputs and summed. The hidden state `h_t` is consumed by experts through the standard mechanism — it is never replaced, averaged, or mixed across experts.

This continues the **safe MoE** pattern that has produced 5 STRICTLY POSITIVE winners in rounds 99, 102, 105, 107, 113.

## Future work

1. **ReMoE + DeepSeek**: Combine ReLU routing with shared expert isolation
2. **ReMoE + QuITE (round 102)**: Use QuITE embedding → ReMoE recurrent step
3. **Adaptive load-balancing λ**: increase λ when utilization variance is high
4. **ReLU vs Softplus**: compare ReLU (hard zeros) vs Softplus (smooth)
5. **ReMoE for sequence-level routing**: per-sequence gate instead of per-step
