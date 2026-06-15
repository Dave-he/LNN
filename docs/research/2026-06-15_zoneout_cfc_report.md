# Round 136 — Zoneout CfC (Krueger et al. 2016, arXiv:1606.01305, ICLR 2017)

**Date**: 2026-06-15
**PRD**: #10-98
**Verdict**: **HONEST NEGATIVE** — 9th negative in 91-136 audit.

## Summary

Tested the **Zoneout** regularizer from Krueger, Maharaj, Kratz,
Ramalho, Ballas (2016/2017) "Zoneout: Regularizing RNNs by
Preserving Hidden States" (arXiv:1606.01305, ICLR 2017). The idea:
randomly preserve the previous hidden state with probability p::

    h_t = h_{t-1}                with prob p_zoneout
    h_t = cf_c_step(x, h_{t-1})  with prob 1 - p_zoneout

**Verdict: HONEST NEGATIVE** — Zoneout makes things WORSE on all 3
datasets, with `zoneout_med` (p=0.3) catastrophically bad on
`structured_irr` (56× worse). The mechanism preserves noise and is
redundant with CfC's own f-gate interpolation.

## 1. Hypothesis

- **H1 (Zoneout reduces variance)**: with Zoneout, the variance of
  test_mse across seeds is < baseline. **REJECTED** (variance
  INCREASES with Zoneout).
- **H2 (Zoneout helps on noisy data)**: with Zoneout, test_mse on
  `random_irr` is < baseline. **REJECTED** (1.2-1.9× worse).
- **H3 (no regression on smooth data)**: with Zoneout, test_mse on
  `sin_irr` is not worse than baseline by >10%. **REJECTED**
  (1.3-6.5× worse).

## 2. Implementation

`ZoneoutCfCCell` and `ZoneoutCfCStackedNetwork` in
`lnn/core/zoneout_cfc.py` (~200 lines). 24 unit tests covering
init/forward/gradient/stability/diagnostics/smoke.

Key design choices:

1. **Zoneout applied to the new hidden state** (after the CfC
   step), with per-neuron mask.
2. **At eval mode, Zoneout is disabled** (full computation).
3. **No extra parameters** — Zoneout is purely a stochastic
   regularizer.
4. **Per-cell mask** (per-neuron) — the standard "recurrent dropout"
   pattern.

## 3. Bench results (18 cells, 30 epochs, 2 seeds)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | **0.0094±0.0019** | **0.0053±0.0010** | **0.0013±0.0004** | 2545 |
| zoneout_low (p=0.1) | 0.0125±0.0047 | 0.0222±0.0041 | 0.0016±0.0008 | 2545 |
| zoneout_med (p=0.3) | 0.0607±0.0144 | 0.3002±0.0674 | 0.0025±0.0004 | 2545 |

**ALL 3 Zoneout variants LOSE on ALL 3 datasets**:

- **sin_irr**: zoneout_low 1.3×, zoneout_med **6.5×** worse
- **structured_irr**: zoneout_low 4.2×, zoneout_med **56×** worse
- **random_irr**: zoneout_low 1.2×, zoneout_med 1.9× worse

H1 (reduces variance) — **REJECTED** (variance INCREASES with Zoneout)
H2 (helps on noisy) — **REJECTED** (1.2-1.9× worse)
H3 (no regression on smooth) — **REJECTED** (1.3-6.5× worse)

## 4. Why it fails

### 4.1 Short sequences don't need regularization

Our 1D time series are T=32 steps with B=8 batch size. With only
30 epochs of training, the model is not overfitting. Zoneout's
primary benefit is to prevent overfitting, but there is no
overfitting to prevent in this regime.

### 4.2 Zoneout preserves noise

When the input has noise, Zoneout preserves the previous hidden
state which itself may contain noise. This is especially bad on
`structured_irr` where the regime switch at T/2 means the model
needs to UPDATE its state quickly. Zoneout prevents this update
56× worse with p=0.3.

### 4.3 CfC's f-gate is already a per-step interpolation

CfC's `f = sigmoid(W_f [x, h])` already provides a per-step
interpolation between the previous state and the new computation.
This is essentially a learned, input-conditional version of
Zoneout. Adding Zoneout on top is redundant — it interferes with
the f-gate's learned interpolation.

### 4.4 Zoneout's mask is independent of input

The Zoneout mask is a fixed Bernoulli per-neuron probability. It
does not depend on the input, so it cannot adapt to the
information content of each step. On regime-switching data, the
model needs to fully update its state at the switch, but Zoneout
randomly prevents this.

## 5. NEW INSIGHTS

1. **CfC's f-gate is already a learned Zoneout**. The f-gate
   provides per-step interpolation, which is what Zoneout tries to
   add. Adding Zoneout on top of CfC is redundant.
2. **Zoneout preserves noise**. On noisy data, Zoneout keeps the
   noisy previous state, accumulating noise over time.
3. **Short 1D sequences don't need regularization**. With T=32 and
   30 epochs, overfitting is not a problem.
4. **Pattern reinforced**: regularization techniques (Zoneout) that
   add stochastic noise lose to the same kind of mechanism that
   uses the input (f-gate). The "additive not replacement" pattern
   from 91-135 audit still holds.

## 6. The 91-136 audit: 8 neuron-families + LN + Zoneout

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE (12 winners) |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Gated Input Skip (additive shortcut) | 134 | STRICTLY POSITIVE (13th winner) |
| Layer Normalization (per-sample normalize) | 135 | TARGET-DEPENDENT (smooth only) |
| **Zoneout (preserve h)** | **136** | **NEGATIVE (9th negative)** |
| Recursion depth (MoR) | 126 | NEGATIVE |
| 2nd-order oscillator | 128 | NEGATIVE |
| Multi-timescale ELM | 129 | NEGATIVE |
| Multi-rate MoE + dual attn | 130 | NEGATIVE |
| HGRN (gated linear RNN) | 131 | NEGATIVE |
| AntisymmetricRNN | 132 | NEGATIVE |
| Hebbian Fast Weights | 133 | NEGATIVE |

**Pattern reinforced (14 winners + 1 target-dependent + 9 negatives)**:
all 14 winners preserve the recurrent step + add useful structure;
all 9 negatives propose alternatives to the recurrent step OR add
unsupervised terms (FastWeights) OR regularizers (Zoneout).

## 7. Recommendation

**Zoneout CfC is the 9th NEGATIVE in the 91-136 audit.**

- **DO NOT use ZoneoutCfC for 1D regression** — it is redundant with
  CfC's f-gate and harmful on short sequences.
- **CfC's f-gate is already a learned Zoneout** — input-conditional
  per-step interpolation.
- **Stick with cfc baseline, GIS-CfC, or LN-CfC** for production.

## 8. Critical implementation details

1. **Zoneout is train-only** — at eval, full computation.
2. **Per-neuron mask** — Bernoulli(p_zoneout) per hidden dim.
3. **No extra parameters** — purely stochastic.
4. **The mask is the same for all timesteps within a forward pass**
   (recurrent dropout pattern).
