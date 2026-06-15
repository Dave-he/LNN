# PRD #10-97 — Layer Normalization for CfC (Round 135)

**Date**: 2026-06-15
**Round**: 135 (response to Layer Normalization, Ba/Kiros/Hinton 2016, arXiv:1607.06450)
**Status**: Drafted.

## 1. Why round 135

The **Layer Normalization** paper (Ba, Kiros, Hinton, 2016) introduced
a per-sample normalization that is well-suited for RNNs:

```
mean = mean(h)  # over feature dim
var  = var(h)
h_norm = (h - mean) / sqrt(var + eps)
h_out = gamma * h_norm + beta
```

Unlike Batch Normalization, Layer Norm does not depend on batch
statistics — it works the same at training and inference time, and
handles variable-length sequences and small batches naturally.

For RNNs, the canonical placement is to apply Layer Norm to the
**gates' input** or to the **hidden state** after the recurrent step.
Ba et al. 2016 §3.2 recommend applying LN to the **gate input** (the
linear projection output, not the recurrent step's output). The
key insight: gates with inputs at different scales get pushed to
saturating regions; LN keeps the gate input at a consistent scale.

### 1.1 Mechanism for CfC

Apply Layer Norm to the **combined input** [x, h] BEFORE the f-gate,
g-branch, and h-branch linear projections. This is the Ba et al. 2016
"in the recurrent layer" pattern::

    combined = [x, h]              # raw
    combined = LayerNorm(combined) # normalize per-sample
    f = sigmoid(W_f combined)
    g = tanh(W_g combined)
    h_out = tanh(W_h combined)
    decay = sigmoid(-f * time_scale)
    h_new = decay * g + (1-decay) * h_out

The Layer Norm parameters gamma, beta are LEARNED (initialized to
gamma=1, beta=0, which is identity at start).

### 1.2 Why this should win per the 91-134 audit

The audit shows:
- 13 STRICTLY POSITIVE winners all preserve the recurrent step + add
  useful structure (MoE experts, input-side processing, additive
  shortcuts).
- 8 negatives propose alternatives to the recurrent step (HGRN,
  Antisymm, etc.) or add unsupervised terms (FastWeights).

Layer Norm:
- **Preserves W·h** and CfC's f-gate (the recurrent step is
  unchanged).
- **Adds a useful structure** — keeps the gate input at a
  consistent scale, preventing saturation and improving gradient
  flow.
- **Is structural** — modifies the input to the recurrent step.
- **Is well-established** — Ba et al. 2016 showed 2-7× speedup on
  attention and RNN tasks.

The risk: LN might be redundant with CfC's own normalization
(time_scale parameter) or might over-constrain the representation.

## 2. Hypotheses

- **H1 (LN helps on noisy data)**: with LN, test_mse on
  `random_irr` is < unconstrained CfC baseline (because LN prevents
  the gate from saturating on noisy inputs).
- **H2 (LN helps on regime switching)**: with LN, test_mse on
  `structured_irr` is < baseline (regime switches are exactly when
  gate saturation hurts most).
- **H3 (no regression on smooth data)**: with LN, test_mse on
  `sin_irr` is not worse than baseline by >5%.

## 3. Plan

### 3.1 Implementation (`lnn/core/layer_norm_cfc.py`)

Two classes:
- `LayerNormCfCCell(nn.Module)`: standard 3-branch CfC cell with
  Layer Norm applied to combined = [x, h] BEFORE the linear
  projections.
- `LayerNormCfCStackedNetwork(nn.Module)`: 2-layer stack with
  per-cell LN.

Key design choices:
- LN applied to the **input of f-gate/g-branch/h-branch**, NOT to
  the output of the recurrent step (Ba et al. 2016 §3.2).
- LN parameters are per-cell (each cell has its own gamma, beta).
- LN is applied separately to the input of f, g, h (three separate
  LayerNorms).
- eps = 1e-5 (default for layer norm).
- gamma initialized to 1.0, beta to 0.0 (identity at start).

### 3.2 Tests (`tests/test_layer_norm_cfc.py`)

20+ unit tests covering:
- Init: gamma=1, beta=0, eps=1e-5.
- LN is identity at init: when gamma=1, beta=0, LN(combined) = combined.
- Forward: shape preservation.
- Forward: h stays bounded.
- Gradient: flows to gamma and beta.
- Stacked: gradient flows to all layers.
- Smoke: learns toy sin.
- Sanity: LN output has mean ≈ 0, var ≈ 1 (over feature dim).

### 3.3 Bench (`scripts/bench_layer_norm_cfc.py`)

18 cells (3 conds × 3 datasets × 2 seeds × 30 epochs):
- `cfc` (baseline)
- `ln_cfc` (LN applied to combined = [x, h])
- `ln_separate` (LN applied separately to x and h)

Datasets: sin_irr, structured_irr, random_irr (D=2, missing_rate=0.3)

## 4. Expected outcomes

- **Best case (probability ~30%)**: H1 + H2 + H3 all confirmed.
  Layer Norm is the **14th STRICTLY POSITIVE** winner. Per-sample
  normalization helps 1D time-series.
- **Likely case (probability ~50%)**: H3 confirmed, H1/H2 partial.
  **TARGET-DEPENDENT-WITH-NUANCE**.
- **Worst case (probability ~20%)**: All 3 hypotheses rejected. LN
  is redundant with CfC's own normalization. 9th negative.

## 5. Why this is worth testing

The 91-134 audit strongly suggests "additive + useful" mechanisms
win. Layer Norm is the most well-established "additive" mechanism
in the RNN literature (Ba et al. 2016) and we haven't tested it
yet on CfC. If it wins, it would be a high-confidence production
candidate.

## 6. Files to create

- `lnn/core/layer_norm_cfc.py` (~200 lines)
- `tests/test_layer_norm_cfc.py` (~300 lines, 20+ tests)
- `scripts/bench_layer_norm_cfc.py` (~250 lines, 18 cells)
- `docs/research/2026-06-15_layer_norm_cfc_report.md`
