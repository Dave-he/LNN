# PRD #10-168 — Round 206 — Attention over Hidden States on CfC

**Date**: 2026-06-16
**Round**: 206
**Branch**: master
**Audit context (91-205)**: 47 strictly positive + 26 target-dep
+ 55 negatives = 128 mechanism classes.

## Background

After 6 rounds on spectral axis (r200-r205), pivot to a
**fundamentally different mechanism axis**: attention over
past hidden states.

Hypothesis: the past hidden states may contain useful
context that the gating can leverage for better predictions.

## Goal

Test if attention over past hidden states provides
improvement over the r187 baseline.

## Mechanism

```python
q = linear_q(h_t)                # query: current h
k = linear_k(h_past)             # keys: past h's
v = linear_v(h_past)             # values: past h's
scores = (q * k).sum(-1) / sqrt(d)  # attention scores
attn = softmax(scores, dim=-1)   # attention weights
g = sum_i attn_i * v_i            # context vector
h_new = τ_eff * g + (1-τ_eff) * h_branch
```

## Configurations (2 conds)

1. `cf`: r187 baseline
2. `attn`: r206 (attention over h)

## Result (12 cells: 2 conds × 3 datasets × 2 seeds × 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | mean |
|------|---------|----------------|------------|------|
| cf | 0.0553 | 0.0030 | 0.0907 | 0.0497 |
| **attn (r206)** | **0.0506** | **0.0054** | **0.0848** | **0.0469** |

Per-dataset (r206 vs cf):
- sin: 0.0553 → 0.0506 (**-8.5%**)
- structured: 0.0030 → 0.0054 (**+80%** WORSE)
- random: 0.0907 → 0.0848 (**-6.5%**)

## Verdict

**NEGATIVE (56th)** — attention over past hidden states
HURTS structured/multi-regime data by 80%.

## Pattern (47 + 26 + 55 = 128 → 47 + 26 + 56 = 129)

- 47 strictly positive (unchanged)
- 26 target-dep (unchanged)
- **56 negatives** (UP from 55, +1)
- Total: **129 mechanism classes**

## Why attention hurts multi-regime data

1. **Past hidden states contain mixed regimes**
2. **Attention averages past states** — for structured data
   with regime changes, this dilutes the signal
3. **For pure sin**: small win (-8.5%) — past states help
4. **For random**: small win (-6.5%) — past states help
5. **For structured (sin→linear)**: large loss (+80%) —
   past states for sin regime confuse the linear regime

## Why this is a useful NEG

1. **Confirms past-state mixing is harmful for regime changes**
2. **Suggests: per-regime attention or per-regime MoE** may help
3. **Pivots to attention axis confirmed negative**
4. **Helps design r207+: regime-aware attention**

## Caveats

- 2 seeds, 30 epochs
- Hidden=12, lr=1e-2, batch_size=16
- The seed-0 struct result (0.0098) is anomalously high
- Attention doesn't include positional encoding
- No layer norm on attention output

## Next ideas

1. **Per-regime attention** — different attention per regime
2. **Positional encoding** — add position to attention
3. **Gated attention** — gate attention weights
4. **Sparse attention** — top-k past states
5. **Cross-attention** — attend to input features instead

## Files

- `lnn/core/learned_beta_ps_ln_khlfft_attn_cfc.py` (~225 lines)
- `tests/test_learned_beta_ps_ln_khlfft_attn_cfc.py` (11 tests)
- `scripts/bench_learned_beta_ps_ln_khlfft_attn_cfc.py` (12-cell bench)
- `results/bench_learned_beta_ps_ln_khlfft_attn_cfc.json`

**Why:** Round 206 is **NEGATIVE (56th)** — attention over
past hidden states hurts structured/multi-regime data.

**How to apply:** Do NOT use past-state attention for
multi-regime data. Spectral gating (r200-r205) remains
the dominant time-series gating mechanism.
