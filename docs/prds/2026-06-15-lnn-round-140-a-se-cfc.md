# PRD #10-102 — Squeeze-and-Excitation (SE) Channel Attention for CfC (Round 140)

**Date**: 2026-06-15
**Round**: 140 (response to SE-Net literature, Hu et al. 2017)
**Status**: Drafted.

## 1. Why round 140

**Squeeze-and-Excitation Networks (SE-Net)** (Hu et al. 2017, "Squeeze-
and-Excitation Networks", CVPR 2018) introduced channel-wise attention
that recalibrates feature responses. The mechanism:

```
SE(x):
  z = GlobalAvgPool(x)         # [B, C]
  z = ReLU(W1 z)               # [B, C/r]
  z = sigmoid(W2 z)            # [B, C] in [0, 1]
  return z * x                 # recalibrated
```

For CfC, we don't have spatial features, but we have input channels.
A per-channel attention could:
- Adaptively weight which input features matter
- Be computed from the hidden state (cross-attention-like)
- Add useful input-side structure that the f-gate doesn't provide

This is structurally:
- Additive (preserves recurrent step)
- Input-side (modulates x before CfC)
- Universal performance booster in CNNs (ILSVRC 2017 winner)
- Different from CfC's f-gate (per-hidden scalar) and GLU (per-input
  from input only) — SE can use HIDDEN state info to compute attention

## 2. Mechanism

```
h_pool = mean(h)                 # [B, hidden_size] (could be last h)
score = sigmoid(W_score [x_t, h]) # [B, D_in] in [0, 1]
x_se = score * x_t                # [B, D_in] recalibrated
h_t = cf_c_step(x_se, h_{t-1})    # standard 3-branch CfC
```

Three variants:
- `se_input`: SE computed from input only (W_score * x_t)
- `se_hid`: SE computed from hidden only (W_score * h)
- `se_concat`: SE computed from concat [x, h]

## 3. Hypotheses

- **H1 (SE helps on smooth data)**: with SE channel attention,
  test_mse on `sin_irr` is < baseline.
- **H2 (SE helps on structured data)**: with SE channel attention,
  test_mse on `structured_irr` is < baseline.
- **H3 (no regression on noisy data)**: with SE channel attention,
  test_mse on `random_irr` is not worse than baseline by >10%.

## 4. Why this should win per the 91-139 audit

The audit shows:
- 13 STRICTLY POSITIVE winners all preserve the recurrent step + add
  useful structure (input-side processing, MoE experts, additive
  shortcuts).
- 3 TARGET-DEPENDENT (LN 135, conv 137, glu_residual 139) — all
  input-side processing that helps smooth, neutral/mild-regression
  on noisy.
- 11 negatives propose alternatives, add unsupervised terms, add
  regularizers, add redundant info, or create bottlenecks.

SE channel attention:
- **Preserves the recurrent step** entirely.
- **Adds useful input-side structure** — per-channel attention
  computed from input and/or hidden state.
- **Is structural** — modifies the input, not the recurrent step.
- **Different from CfC's f-gate** — f-gate is per-hidden scalar over
  [x, h], SE is per-input channel (different dimensions!).
- **Different from GLU (round 139)** — GLU uses input only, SE can
  use both input and hidden (cross-attention style).

The risk: SE adds parameters that could overfit on noisy data. But
CNN history shows SE is a robust universal performance booster.

## 5. Plan

### 5.1 Implementation (`lnn/core/se_cfc.py`)

Two classes:
- `SECfCCell(nn.Module)`: standard 3-branch CfC cell with SE
  channel attention.
- `SECfCStackedNetwork(nn.Module)`: 2-layer stack with SE on each
  layer's input.

Key design choices:
- SE score computed from concat [x, h] (cross-attention style).
- Sigmoid bounded in [0, 1].
- CfC recurrent step is unchanged.
- Per-layer SE.

### 5.2 Tests (`tests/test_se_cfc.py`)

20+ unit tests covering:
- Init: SE parameters.
- Forward: shape preservation.
- Score: SE scores are bounded in [0, 1].
- Gradient: flows to SE weights.
- Stacked: gradient flows to all layers.
- Smoke: learns toy sin.
- Sanity: SE=0 zero out the input, SE=1 pass through.

### 5.3 Bench (`scripts/bench_se_cfc.py`)

18 cells (3 conds × 3 datasets × 2 seeds × 30 epochs):
- `cfc` (baseline)
- `se_concat` (SE from concat [x, h])
- `se_input` (SE from input only)

Datasets: sin_irr, structured_irr, random_irr (D=2, missing_rate=0.3)

## 6. Expected outcomes

- **Best case (~40%)**: H1 + H2 + H3 all confirmed. SE is the
  **14th STRICTLY POSITIVE** winner.
- **Likely case (~40%)**: H1 + H3 confirmed, H2 partial.
  **TARGET-DEPENDENT-WITH-NUANCE** (helps smooth, neutral noisy).
- **Worst case (~20%)**: All 3 hypotheses rejected. 12th negative.

## 7. Why this is worth testing

The 91-139 audit strongly suggests "input-side processing + add to
recurrent step" mechanisms win. QuITE 102, GIS 134, glu_residual 139
were winners/target-dep. SE channel attention is a 5-line addition
that could be a 14th winner. The cross-attention style (using hidden
state) makes it different from GLU and could be the key differentiator.

## 8. Files to create

- `lnn/core/se_cfc.py` (~200 lines)
- `tests/test_se_cfc.py` (~300 lines, 20+ tests)
- `scripts/bench_se_cfc.py` (~250 lines, 18 cells)
- `docs/research/2026-06-15_se_cfc_report.md`
