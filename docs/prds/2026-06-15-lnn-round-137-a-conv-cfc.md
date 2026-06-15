# PRD #10-99 — 1D Convolutional Input Preprocessing for CfC (Round 137)

**Date**: 2026-06-15
**Round**: 137 (response to Convolutional RNN literature, e.g. ConvLSTM 2015)
**Status**: Drafted.

## 1. Why round 137

Convolutional Recurrent Networks (e.g., ConvLSTM, Shi et al. 2015)
apply 1D (or 2D) convolutions to the input sequence BEFORE the
recurrent step. This is a well-established pattern that combines
the local pattern extraction of CNNs with the temporal dynamics of
RNNs.

For CfC, this means:
- The input is first passed through a 1D causal convolution
- The conv output is fed to the standard CfC recurrent step
- The recurrent step is unchanged

### 1.1 Mechanism

```
x_conv = Conv1d_causal(x)        # [B, T, D_in] -> [B, T, D_conv]
# Standard CfC
h_t = cf_c_step(x_conv[t], h_{t-1})
```

The 1D causal convolution has kernel size k (e.g., k=3) and uses
causal padding so the output at time t only depends on inputs at
times ≤ t.

### 1.2 Why this should win per the 91-136 audit

The audit shows:
- 13 STRICTLY POSITIVE winners all preserve the recurrent step + add
  useful structure (MoE experts, input-side processing, additive
  shortcuts).
- 1 TARGET-DEPENDENT (LN round 135).
- 9 negatives propose alternatives to the recurrent step or add
  unsupervised/regularizer terms.

1D Conv + CfC:
- **Preserves the recurrent step** entirely.
- **Adds useful input-side structure** — captures local temporal
  patterns before feeding to CfC.
- **Is structural** — modifies the input, not the recurrent step.
- **Similar to QuITE+MoE (round 103)** but simpler (1D conv vs
  attention). QuITE+MoE was the 28th winner; 1D Conv should be
  even simpler and possibly more robust.

The risk: 1D conv might be redundant with CfC's W·h (which already
captures local patterns) or with QuITE (which already does input
processing).

## 2. Hypotheses

- **H1 (conv helps on smooth data)**: with 1D conv preprocessing,
  test_mse on `sin_irr` is < baseline (conv captures local sine
  patterns).
- **H2 (conv helps on structured data)**: with 1D conv
  preprocessing, test_mse on `structured_irr` is < baseline (conv
  captures regime transitions).
- **H3 (no regression on noisy data)**: with 1D conv preprocessing,
  test_mse on `random_irr` is not worse than baseline by >10%.

## 3. Plan

### 3.1 Implementation (`lnn/core/conv_cfc.py`)

Two classes:
- `ConvCfCCell(nn.Module)`: standard 3-branch CfC cell preceded by
  a 1D causal convolution applied to the input.
- `ConvCfCStackedNetwork(nn.Module)`: 2-layer stack with shared
  conv preprocessing.

Key design choices:
- 1D causal conv with kernel size 3 and same number of output
  channels as input.
- Conv weights initialized with small std (0.1).
- No bias in the conv (to keep it simple, can be added later).
- CfC recurrent step is unchanged.

### 3.2 Tests (`tests/test_conv_cfc.py`)

20+ unit tests covering:
- Init: conv weights and CfC parameters.
- Forward: shape preservation.
- Causal: output at time t only depends on input at times ≤ t.
- Gradient: flows to conv weights.
- Stacked: gradient flows to all layers.
- Smoke: learns toy sin.
- Sanity: conv output at t depends on [t-2, t-1, t] (kernel=3).

### 3.3 Bench (`scripts/bench_conv_cfc.py`)

18 cells (3 conds × 3 datasets × 2 seeds × 30 epochs):
- `cfc` (baseline)
- `conv_k3` (1D conv with kernel=3)
- `conv_k5` (1D conv with kernel=5)

Datasets: sin_irr, structured_irr, random_irr (D=2, missing_rate=0.3)

## 4. Expected outcomes

- **Best case (probability ~35%)**: H1 + H2 + H3 all confirmed.
  1D Conv + CfC is the **15th STRICTLY POSITIVE** winner.
- **Likely case (probability ~45%)**: H1 + H3 confirmed, H2
  partial. **TARGET-DEPENDENT-WITH-NUANCE** (helps smooth/structured,
  neutral noisy).
- **Worst case (probability ~20%)**: All 3 hypotheses rejected.
  1D conv is redundant with CfC's W·h. 10th negative.

## 5. Why this is worth testing

The 91-136 audit strongly suggests "input-side processing + add to
recurrent step" mechanisms win. QuITE+MoE (round 103) was a winner.
1D Conv is the simplest input-side processing I can think of. If it
wins, it would be a high-confidence production candidate (no extra
recurrent parameters).

## 6. Files to create

- `lnn/core/conv_cfc.py` (~200 lines)
- `tests/test_conv_cfc.py` (~300 lines, 20+ tests)
- `scripts/bench_conv_cfc.py` (~250 lines, 18 cells)
- `docs/research/2026-06-15_conv_cfc_report.md`
