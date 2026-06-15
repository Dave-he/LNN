# PRD #10-93 — Hierarchically Gated RNN for CfC (Round 131)

**Date**: 2026-06-15
**Round**: 131 (response to HGRN, NeurIPS 2023, arXiv:2404.18807)
**Status**: Drafted.

## 1. Why round 131

The HGRN paper (Qi, Yang, Zhao, Wang, Sun, Wei, "Hierarchically Gated Recurrent
Neural Network for Sequence Modeling", NeurIPS 2023) proposes a **gated linear
RNN** where the forget gate has a learnable **lower bound** α that increases
monotonically across layers. The intuition: lower layers forget more (model
local/short-term), upper layers forget less (model long-term).

The mechanism is **structural** — it modifies the recurrent step directly, not
just routing. The audit (rounds 91-130) shows structural mechanisms that
preserve the recurrent state's W·h nonlinearity (DeepSeek 113, LoRA 118, DAG
120, sigmoid 116, sigmoid-MoE) are STRICTLY POSITIVE in 1D, while stochastic
or routing-only mechanisms tend to be negative.

**Question for round 131**: does the HGRN mechanism — bounded forget gate with
hierarchical α — improve our 1D time-series stack?

The mechanism has two parts we can test independently:
- **Free gate** (α=0): linear recurrence `h_t = (1-gate) * h_{t-1} + gate * (W x_t)`,
  exactly as a linear RNN with input gate. The baseline against which HGRN
  improves.
- **Bounded gate** (α=fixed): same but with `gate = max(α, sigmoid(W_g x_t))`.
  The lower bound is a single scalar per cell.
- **Hierarchical** (α=per-layer monotonic): in a stacked network, α_l increases
  with layer index l, so deeper layers forget less.

## 2. Hypotheses

- **H1 (bounded gate helps vs free gate on noisy data)**: with α=0.1,
  test_mse on `random_irr` is < the free-gate baseline (because bounded
  forgetting prevents the state from being wiped by noisy inputs).
- **H2 (hierarchical beats uniform)**: with α_l = l/(L-1) * α_max, the
  stacked network has lower test_mse than a flat α=α_max/2 across all
  layers.
- **H3 (preserves recurrent nonlinearity)**: the bounded-gate variant is at
  least as stable as the free-gate variant (no NaN/divergence in 100 epochs).

## 3. Plan

### 3.1 Implementation (`lnn/core/hgrn_cfc.py`)

Two classes:

```python
class HGRNCfCCell(nn.Module):
    """Gated linear recurrence with bounded forget gate.
    
    h_t = (1 - gate_t) * h_{t-1} + gate_t * tanh(W_x @ x_t + b)
    gate_t = max(alpha, sigmoid(W_g @ x_t + b_g))  # alpha is per-cell
    """
    def __init__(self, input_size, hidden_size, alpha_init=0.1,
                 learn_alpha=True, nonlinearity='tanh'):
        ...

class HGRNCfCStackedNetwork(nn.Module):
    """Stacked HGRN cells with optional hierarchical alpha.
    
    If hierarchical=True, alpha_l = l/(L-1) * alpha_max (monotonic in l).
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2,
                 alpha_init=0.1, hierarchical=True, alpha_max=0.7):
        ...
```

### 3.2 Tests (`tests/test_hgrn_cfc.py`)

20 unit tests covering:
- Init: α is learnable, init value correct
- Init: alpha=False disables learnable parameter
- Forward: shape preservation
- Gate bound: gate values >= alpha
- Hierarchical: alpha_l increases monotonically across layers
- Gradient flow: to W_x, W_g, alpha (when learnable=True)
- Stability: no NaN/Inf in 100-step forward with random init
- Alpha fixed (not learnable) doesn't accumulate grad

### 3.3 Bench (`scripts/bench_hgrn_cfc.py`)

4 conditions × 3 datasets × 2 seeds = 24 cells:
- `cfc` (baseline)
- `hgrn_free` (no lower bound, α=0)
- `hgrn_bounded` (single learnable α)
- `hgrn_hierarchical` (per-layer monotonic α)

Datasets: sin_irr, structured_irr, random_irr (D=2, missing_rate=0.3)

## 4. Expected outcomes

Given the audit pattern:
- **Best case** (probability ~25%): HGRN bounded is the **13th STRICTLY
  POSITIVE** winner. H1 + H2 + H3 all confirmed.
- **Likely case** (probability ~50%): HGRN free matches baseline, bounded
  helps on noisy data (random_irr), hierarchical helps on structured.
  This is **TARGET-DEPENDENT-WITH-NUANCE**.
- **Worst case** (probability ~25%): All HGRN variants lose. Linear
  recurrence lacks the nonlinearity of CfC and the gating alone doesn't
  recover it. 16th negative.

## 5. Why this is worth testing

The audit has 12 STRICTLY POSITIVE winners. The most successful STRUCTURAL
mechanisms are DeepSeek (113, additive residual), LoRA (118, low-rank), DAG
(120, structural aggregation). All modify the recurrent step but PRESERVE
nonlinearity.

HGRN's bounded gate is a small, clean structural change. Even if it loses
in 1D, the failure mode is informative: it tells us that gating alone
(replacing W·h with linear recurrence) is insufficient — you need the full
CfC nonlinearity.

## 6. Files to create

- `lnn/core/hgrn_cfc.py` (~200 lines)
- `tests/test_hgrn_cfc.py` (~300 lines, 20 tests)
- `scripts/bench_hgrn_cfc.py` (~250 lines, 24 cells)
- `docs/research/2026-06-15_hgrn_cfc_report.md`
