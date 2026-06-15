# PRD #10-77 — Multi-Head Mixture-of-Experts (MH-MoE) for CfC

**Date**: 2026-06-15
**Round**: 115
**Paper**: arXiv:2404.15045 (Wu, Huang, Wang, Wei, April 2024, NeurIPS 2024) — *Multi-Head Mixture-of-Experts*
**Codebase target**: `lnn/core/mhmoe.py`
**Test file**: `tests/test_mhmoe.py`
**Bench script**: `scripts/bench_mhmoe.py`

## Goal

Implement **Multi-Head Mixture-of-Experts (MH-MoE)** for CfC — split each input into H sub-tokens, route each sub-token to its own expert, process in parallel, concatenate back. The mechanism naturally distributes load across all K experts (fixes the FAME H=0 collapse we saw in round 103) and aligns with the multi-head attention analogy from transformers.

## Background

### What is MH-MoE?

Standard SMoE has two problems:
1. **Low expert activation** during training — many experts are never selected
2. **Inability to analyze multiple semantic concepts** within a single token

MH-MoE's fix: **multi-head mechanism** that splits each input into H sub-tokens, dispatches each to a different expert, then re-integrates.

**Algorithm (transformer setting)**:
1. `x: [B, L, D]` input tokens
2. Split into heads: `x → [B, L, H, D/H] → reshape → [B*L*H, D/H]`
3. Route each sub-token: `g = softmax(W x)` → top-K
4. Each expert is a small MLP with `D/H` input/output
5. Combine: `output: [B*L*H, D/H] → reshape → [B, L, D]`

**Our adaptation for CfC**:
- Sub-token: chunk of the input feature dimension (NOT a chunk of the sequence)
- Each sub-token routed to one of K experts
- Each expert is a **CfC cell** with input/output dim = `D/H`
- Sub-token outputs concatenated back to full D-dim
- Recurrent state is per-cell, preserved across timesteps

### Why is MH-MoE a fit for the 91-114 audit pattern?

The audit pattern: mechanisms that **modify or constrain the recurrent state mixing** are dangerous. MH-MoE does NOT:
- Each sub-token is processed by a separate expert, but the experts all consume the **same hidden state** `h_t`
- The combination is **concatenation** (not averaging), so `h_t` is preserved
- The recurrent state is never modified, mixed, or averaged across experts
- Gradient flows to all K experts naturally (each sub-token picks a different expert on average)

**Critical structural property**:
- K total experts × H heads = K·H parallel sub-token paths
- On average, each expert receives `H · (B/K)` sub-tokens per step → balanced load
- Fixes the **FAME H=0 collapse** we saw in round 103 (QuITE+MoE was needed to fix that)

### Hypothesis

**H1 (STRICTLY POSITIVE)**: MH-MoE with K=4 H=2 beats FAME on all 3 datasets because it fixes the FAME H=0 collapse (every sub-token gets its own expert, so all K are exercised).

**H2 (PARTIAL)**: MH-MoE matches or slightly underperforms DeepSeek (round 113) and ReMoE (round 114) on smooth data — those are the strongest baselines, so MH-MoE just needs to be competitive.

**H3 (PARTIAL)**: H=2 is the sweet spot; H=4 spreads too thin, H=1 reduces to standard MoE.

## Implementation

### Core API (`lnn/core/mhmoe.py`)

```python
class MHRouter(nn.Module):
    """Per-sub-token router: g = softmax(W x), returns top-K experts per sub-token."""

class MHMoECfCCell(nn.Module):
    """K experts, H heads, sub-token splitting, top-K per sub-token."""

class MHMoECfCNetwork(nn.Module):
    """Stacked MH-MoE-style CfC network."""

def mhmoe_utilization(cell) -> dict:
    """Returns expert usage distribution (should be ~uniform when H*K parallel)."""
```

### Forward pass

```python
def forward(self, x_t, h, dt=1.0):
    B, D = x_t.shape
    H, head_dim = self.n_heads, D // self.n_heads
    
    # 1) Split input into H sub-tokens
    sub_tokens = x_t.view(B, H, head_dim)                       # [B, H, head_dim]
    sub_tokens = sub_tokens.reshape(B * H, head_dim)            # [B*H, head_dim]
    
    # 2) Route each sub-token to top-K experts (independent)
    g = self.router(sub_tokens)                                  # [B*H, K]
    top_vals, top_idx = g.topk(self.top_k, dim=-1)               # [B*H, top_k]
    top_vals = top_vals.softmax(dim=-1)                          # normalize
    
    # 3) Process each sub-token by its top-K experts
    expert_outs = torch.stack([e(sub_tokens, h.repeat(H, 1), dt) for e in self.experts], dim=1)  # [B*H, K, head_dim]
    # Gather top-K and weight
    gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, head_dim)  # [B*H, top_k, head_dim]
    selected = expert_outs.gather(1, gather_idx)                 # [B*H, top_k, head_dim]
    routed = (top_vals.unsqueeze(-1) * selected).sum(dim=1)     # [B*H, head_dim]
    
    # 4) Concatenate sub-tokens back
    h_new = routed.view(B, H * head_dim)                         # [B, D]
    return h_new
```

### Key implementation details

1. **Sub-token = feature chunk**: H sub-tokens per timestep, each of dim D/H
2. **Per-sub-token routing**: each sub-token picks its own top-K experts (independent)
3. **Recurrent state shared across sub-tokens**: h is the same for all H sub-tokens of a given timestep
4. **Concatenation (not averaging)**: outputs are concatenated, preserving the D-dim signal
5. **Required: D % H == 0** (asserted at init)

## Tests (`tests/test_mhmoe.py`, ~25 tests)

- TestMHRouterInit (3): init linear, MLP, invalid n_experts
- TestMHRouterForward (4): shape, softmax, top-K sparsity, gradient flow
- TestMHMoECfCCellInit (5): default K=4 H=2, K=2 H=4, invalid D%H, K=1, n_tau
- TestMHMoECfCCellForward (7): forward shape, gradient to all experts, sub-token split, top-K sparsity, hidden state shared, last_g stash
- TestMHMoEUtilization (3): uniform on average, balances across experts
- TestMHMoECfCNetwork (5): init, dense forward, last-step, with mask, gradient flows
- TestMHMoESineSmoke (1): converges on toy sin

## Bench (`scripts/bench_mhmoe.py`)

30 cells (3 datasets × 5 conditions × 2 seeds × 50 epochs):

| Cond | K | H | top_k | Description |
|------|---|---|-------|-------------|
| `baseline_cfc`        | 1 | 1 | n/a | Standard CfC (control) |
| `fame_k3_t1`          | 3 | 1 | 1 | FAME K=3 top_k=1 (round 78 baseline) |
| `mhmoe_k4_h2_t1`      | 4 | 2 | 1 | MH-MoE K=4 H=2 top_k=1 |
| `mhmoe_k4_h2_t2`      | 4 | 2 | 2 | MH-MoE K=4 H=2 top_k=2 |
| `mhmoe_k4_h4_t1`      | 4 | 4 | 1 | MH-MoE K=4 H=4 top_k=1 (more sub-tokens) |

Datasets: `sin_irr`, `structured_irr`, `random_irr` (same as rounds 113-114).

## Why this is structural + data-independent + DOESN'T modify recurrent state mixing

- **Structural**: changes routing topology to multi-head, H parallel sub-tokens
- **Data-independent**: H is a hyperparameter, sub-token splitting is intrinsic
- **Doesn't modify recurrent state**: each expert is a standard CfC cell; the hidden state `h_t` is the same across sub-tokens; the combination is concatenation (not averaging)

This continues the **safe MoE** pattern that has produced 6 STRICTLY POSITIVE winners in rounds 99, 102, 105, 107, 113, 114.

## Future work

1. **MH-MoE + DeepSeek (round 113)**: shared expert + MH-MoE for routed
2. **MH-MoE + ReMoE (round 114)**: ReLU router per sub-token
3. **Per-sub-token top-K=1 with K=H**: forces each sub-token to a different expert
4. **Per-head experts**: K experts per head instead of K shared experts
5. **MH-MoE + QuITE (round 102)**: QuITE embedding → MH-MoE recurrent step
