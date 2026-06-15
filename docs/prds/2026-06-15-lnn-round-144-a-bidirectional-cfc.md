# PRD #10-106 — Round 144: Bidirectional CfC

**Date**: 2026-06-15
**Round**: 144
**Session**: /loop 1h #69
**Author**: heyongxian + Claude (MiniMax-M3)
**Status**: PROPOSAL

## 1. Paper / Background

**Bidirectional Recurrent Neural Networks**
Mike Schuster, Kuldip K. Paliwal
1997 IEEE Transactions on Signal Processing

The idea: run two separate recurrent passes (forward and backward)
over the input sequence, then combine their hidden states. The
forward pass sees `x[0..t]`, the backward pass sees `x[T..t]`. The
combined hidden state at timestep t has access to the FULL
sequence context (both past and future).

This is a CLASSIC technique for sequence labeling (POS tagging,
NER, speech recognition) where future context is available at
inference time. For our 1D regression task (predict y_t from x_t),
bidirectional CfC should help because:
- For smooth data (sin_irr): the model can use the full period
  to refine y_t prediction
- For structured data (structured_irr): the model can see both
  regimes when processing the boundary
- For noisy data (random_irr): bidirectional MIGHT OVERFIT because
  it has access to future noise patterns

## 2. Why this round

The 91-143 audit pattern is:
- **13 winners**: 12 MoE (expert-based) + GIS (additive shortcut)
  + QuITE (input embedding)
- **4 target-dependent**: LN 135, conv 137, glu_residual 139,
  decoupled/IndRNN 143 (all input-side processing)
- **14 negatives**: per-step modifications, alternatives, regularizers

Bidirectional CfC tests a NEW pattern not yet in the audit:
**forward+backward structural addition**. This is different from
input-side processing (LN, conv, GLU+skip) and different from
expert routing (MoE). It's a structural addition to the recurrent
computation.

## 3. Hypothesis

**H1 (Bidirectional helps on smooth data)**: with bidirectional
pass, test_mse on `sin_irr` is < baseline.
**PASS criterion**: test_mse ≤ baseline × 0.95.

**H2 (Bidirectional helps on structured data)**: with bidirectional
pass, test_mse on `structured_irr` is < baseline.
**PASS criterion**: test_mse ≤ baseline × 0.95.

**H3 (Bidirectional doesn't hurt on noisy data)**: with
bidirectional pass, test_mse on `random_irr` is not worse than
baseline by >10%.
**PASS criterion**: test_mse ≤ baseline × 1.10.

## 4. Design

### 4.1 Bidirectional CfC

`BidirectionalCfCCell` — wraps two CfCCells (forward and backward),
combines outputs.

```python
class BidirectionalCfCCell(nn.Module):
    def __init__(self, input_size, hidden_size, ...):
        self.forward_cell = CfCCell(input_size, hidden_size, ...)
        self.backward_cell = CfCCell(input_size, hidden_size, ...)
```

Forward pass:
```python
# Forward pass
h_fwd = [zeros]
for t in range(T):
    h_fwd[t+1] = forward_cell(x_t, h_fwd[t])

# Backward pass
h_bwd = [zeros]
for t in range(T-1, -1, -1):
    h_bwd[t] = backward_cell(x_t, h_bwd[t+1])

# Combined hidden state at each t
h_combined[t] = concat(h_fwd[t+1], h_bwd[t])
```

Output: `head(h_combined[t])` for each t.

### 4.2 Stacked bidirectional network

`BidirectionalCfCStackedNetwork` — 2-layer stack of bidirectional
CfC. Each layer has its own forward and backward CfC cells.

For layer L > 0, the input is the previous layer's bidirectional
hidden state (concat of forward + backward).

### 4.3 Variant: weighted bidirectional

`BidirectionalWeightedCfCCell` — learns a per-timestep weight
α_t that combines forward and backward:

```
h_combined[t] = α_t * h_fwd[t] + (1 - α_t) * h_bwd[t]
```

where `α_t = σ(W_α [h_fwd[t], h_bwd[t]])`.

This is a more expressive variant that lets the model dynamically
decide how much to use forward vs backward context.

## 5. Risks

1. **Bidirectional overfits on noisy data** — the model has access
   to future noise patterns, which it can memorize.
2. **2x parameter count** — forward + backward CfC cells
   (compared to unidirectional).
3. **NaN handling in backward pass** — when computing backward,
   `h_bwd[t+1]` may have NaN-propagated values. Need careful
   masking.

## 6. Validation

- **Unit tests** (≥18): init, forward, gradient flow, stability,
  NaN handling, stacked smoke, weighted variant smoke.
- **Bench**: 18 cells (3 conds × 3 datasets × 2 seeds × 30 epochs).
- **Datasets**: `sin_irr`, `structured_irr`, `random_irr`.

## 7. Expected Verdict

**Speculative TARGET-DEPENDENT** or **HONEST POSITIVE**:

- If bidirectional helps smooth and structured but hurts noisy:
  → TARGET-DEPENDENT (similar to glu_residual 139, decoupled 143)
- If bidirectional wins on all 3 datasets: → STRICTLY POSITIVE
  (15th winner!)
- If bidirectional loses on all 3: → NEGATIVE

Bidirectional is a STRUCTURAL ADDITION (not a per-step
modification), so it has a higher POSITIVE probability than
NEGATIVE based on the 91-143 audit pattern.

## 8. Files

- `lnn/core/bidirectional_cfc.py` (new)
- `tests/test_bidirectional_cfc.py` (new)
- `scripts/bench_bidirectional_cfc.py` (new)
- `results/bench_bidirectional_cfc.json` (new)
- `docs/research/2026-06-15_bidirectional_cfc_report.md` (new)
- `lnn-round-144-bidirectional-cfc.md` (memory)
