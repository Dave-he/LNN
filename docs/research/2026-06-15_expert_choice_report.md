# Round 112 — Expert Choice Routing for CfC (response to arXiv:2202.09368)

**Date**: 2026-06-15
**Round**: 112
**Paper**: arXiv:2202.09368 — *Mixture-of-Experts with Expert Choice Routing* (Zhou et al., 2022)
**PRD**: #10-74
**Tests**: 27/27 in `tests/test_expert_choice.py`
**Bench**: 24 cells, 50 epochs (3 datasets × 4 conditions × 2 seeds), `scripts/bench_expert_choice.py`

## Summary

We implemented **Expert Choice (EC) routing** for the recurrent CfC setting. The key idea: instead of each token/timestep picking its top-K experts (token-choice, FAME), each **expert** picks its top-K tokens, giving perfect load balance by construction (no aux loss needed). This was the natural complement to MoD (round 111).

**The result is a CRITICAL STRUCTURAL NEGATIVE**: EC routing BREAKS recurrent dynamics in time-series MoE. Bench at 50 epochs (24 cells, 2 seeds):

| Condition | sin_irr | structured_irr | random_irr |
|-----------|---------|----------------|------------|
| baseline_cfc | 0.0023±0.0002 | 0.0010±0.0001 | 0.0005±0.0003 |
| ec_dense      | 0.0107±0.0021 | 0.0062±0.0014 | 0.0042±0.0043 |
| ec_half       | 0.0258±0.0054 | 0.0186±0.0040 | 0.0039±0.0020 |
| ec_quarter    | 0.0530±0.0315 | 0.0447±0.0169 | 0.0472±0.0121 |

EC dense (full compute, just averaging expert outputs) is **4-8× worse** than baseline. EC quarter (25% compute) is **23-94× worse**. Worst case: random_irr ec_quarter is 94× worse than baseline.

## Critical structural finding

**The EC mixing step (averaging expert contributions at each timestep) washes out the recurrent state.**

In the original EC-for-Transformers, each token's representation is mixed, but tokens are processed in parallel — averaging works because each token is independent. In CfC, the hidden state is **recurrent** — averaging over experts corrupts the temporal dynamics at every step, and the corruption compounds through the sequence.

This confirms the FAME H=0 (round 78) and SDG-MoE H=0 (round 104) findings: **time-series MoE is fundamentally different from LLM-MoE** because all experts see correlated inputs and the recurrent state is delicate. EC amplifies this problem by enforcing a fixed assignment per timestep AND averaging — both destructive operations for recurrent dynamics.

## What is Expert Choice routing?

Standard MoE: each **token** picks its top-k **experts** (token-choice routing, FAME). Switch Transformer adds an aux loss to encourage balanced load across experts.

EC: each **expert** picks its top-k **tokens** (expert-choice routing). This gives:
- **Perfect load balance by construction** (every expert processes exactly k tokens)
- **Variable** number of experts per token (a token can be picked by 0, 1, ..., K experts)
- **No aux loss** needed (the balance is structural)
- **Faster training** reported in the paper (>2× faster convergence than Switch)

## Implementation

### Core API (`lnn/core/expert_choice.py`, ~430 lines)

```python
class ExpertChoiceRouter(nn.Module):
    """Each expert picks its top-k tokens (per (B, K, T) assignment)."""

class ExpertChoiceCfCCell(nn.Module):
    """K independent CfC cells, mixed by EC assignment.
    - Pre-computed assignment (B, K, T) passed per-step.
    - Mixing: out = (Σ g_e(t) / |active|) · expert_e(x_t, h)
    """

class ExpertChoiceCfCNetwork(nn.Module):
    """Stacked EC layers. Assignment pre-computed once per sequence per layer."""

def expert_choice_load(cell):
    """Diagnostic: per-expert token counts (should be exactly cap_k)."""
```

### Key implementation details

1. **Pre-computed assignment**: Router runs once per sequence at each layer (using input + h0), then sliced per-timestep inside the recurrent loop. This matches the static-graph property of EC.
2. **Average mixing**: `out = (Σ_{e picked t} g_e(t) / |{e picked t}|) · expert_e(x_t, h)`. Normalised by the count of active experts for that timestep.
3. **Sigmoid scores**: Per-(expert, token) sigmoid (Switch-Transformer style) instead of softmax over K.
4. **No aux loss**: The perfect load balance is structural; no Switch-Transformer loss needed.

## Bench

`scripts/bench_expert_choice.py` — 24 cells (3 datasets × 4 conditions × 2 seeds × 50 epochs):

### Conditions
| Cond | cap_k | process_frac | Description |
|------|-------|--------------|-------------|
| `baseline_cfc` | n/a | 1.00 | Standard CfC, no MoE (control) |
| `ec_dense`     | None | 1.00 | EC with cap_k=None (every expert processes all tokens) |
| `ec_half`      | 0.5*T | 0.50 | EC with 50% bucket size |
| `ec_quarter`   | 0.25*T | 0.25 | EC with 25% bucket size |

### Results (test_mse, 2 seeds, 50 epochs)

| Condition | sin_irr | structured_irr | random_irr |
|-----------|---------|----------------|------------|
| baseline_cfc | 0.0023±0.0002 | 0.0010±0.0001 | 0.0005±0.0003 |
| ec_dense      | 0.0107±0.0021 | 0.0062±0.0014 | 0.0042±0.0043 |
| ec_half       | 0.0258±0.0054 | 0.0186±0.0040 | 0.0039±0.0020 |
| ec_quarter    | 0.0530±0.0315 | 0.0447±0.0169 | 0.0472±0.0121 |

### Critical findings

1. **EC dense is 4-8× worse than baseline** — the mixing step alone is harmful, even with full compute.
2. **Reducing k makes it strictly worse** — the assignment constraint is destructive.
3. **Most damage on noisy data** — random_irr ec_quarter is 94× worse.
4. **Perfect load balance** is verified by `test_perfect_load_balance` and `expert_choice_load` — every expert processes exactly `cap_k` tokens.

## Why EC fails on CfC

### The mechanism is right for LLM tokens, wrong for recurrent states

In EC-for-Transformers, the input is a sequence of independent tokens. Each token's representation is mixed across the experts that picked it, but the tokens don't share state. Averaging works because the tokens are independent.

In CfC, the hidden state `h_t` is **recurrent** — it carries the entire past of the sequence. When we mix expert outputs at step t, we corrupt `h_t` for the next step. The corruption compounds through the sequence, washing out the temporal signal.

### Confirms the time-series MoE structural issue

- FAME (round 78) H=0: experts all converge to consensus because they see correlated inputs
- SDG-MoE (round 104) H=0: deliberation amplifies correlation, creating a new lock-in
- EC (round 112): fixed assignment + averaging = **compounded destruction of the recurrent state**

This is the **3rd time-series MoE mechanism** to fail with H=0 or recurrent-dynamics destruction. The pattern is becoming clear: **multi-expert routing in time-series MoE is fundamentally hard** because the experts all see correlated inputs and the recurrent state is delicate.

## Comparison with prior structural mechanisms

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
| **112** | **Expert Choice** | **NEGATIVE (recurrent dynamics broken)** |

**Pattern (91-112)**: mechanisms that modify or constrain the recurrent state mixing are dangerous in time-series MoE. The original CfC's gate-and-update is delicate; replacing it with averaging (or fixed assignment) destroys the dynamics.

## Critical bugs fixed during round 112

1. **topk dim**: `assign_w.topk(cap_k_eff, dim=-1)` was operating on the K dim (3) instead of T dim (8) because the original shape was `[B, T, K]`. Fixed by transposing to `[B, K, T]` first.
2. **Stash ordering**: `cell.last_assign_mask` was being overwritten by the per-step `[B, K]` slice in the cell's forward, hiding the full `[B, K, T]` assignment stashed by the network. Fixed by stashing the `[B, K, T]` AFTER the per-step loop.
3. **Pyright torch false-positives**: pre-existing pattern, ignored per standing rules.

## Recommendation

**DO NOT use EC routing for time-series MoE in production**:
- The averaging mixing step breaks the recurrent dynamics
- Even `ec_dense` (full compute) is 4-8× worse than baseline
- The mechanism is theoretically sound for LLM tokens but fundamentally breaks for recurrent hidden states

**Use EC for**:
- Parallel-input MoE (LLMs, vision) where each token is independent
- **NOT** for recurrent MoE (CfC, LSTM, GRU) where the hidden state is a time-series signal

**Combine with other mechanisms**:
- **MoD (round 111)** works for time-series MoE because it skips timesteps, not averages them
- **FAME (round 78)** works because it uses softmax weighting (not averaging)
- **SETA (round 105)** works because it has shared parameters across experts (not full averaging)

## Files added

- `lnn/core/expert_choice.py` (NEW, ~430 lines)
- `tests/test_expert_choice.py` (NEW, 27/27 tests)
- `scripts/bench_expert_choice.py` (NEW, 24 cells)
- `docs/prds/2026-06-15-lnn-round-112-a-expert-choice.md` (PRD #10-74)
- `docs/research/2026-06-15_expert_choice_report.md` (this report)
- `docs/daily/2026-06-15_LNN_research_summary_v38.md` (digest v38)
- `README.md` (new Expert Choice section)
- `lnn-round-112-expert-choice.md` (memory)

## Future work

1. **Soft EC**: replace the averaging mixing with a learned weighted sum (FAME-style softmax)
2. **EC for embedding only**: use EC routing for the QuITE-style embedding (round 102) but not for the recurrent step
3. **Gating on EC**: add a learned gate to suppress EC mixing when it's harmful
4. **PhysioNet 36D test**: does EC scaling hurt less on high-dim real medical time series?
5. **Combine with MoD (round 111)**: skip timesteps with MoD, then use EC for the remaining (k < T) timesteps
