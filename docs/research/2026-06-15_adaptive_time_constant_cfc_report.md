# Round 141 — Adaptive Time-Constant CfC (Graves 2016)

**Date**: 2026-06-15
**PRD**: #10-103
**Verdict**: **HONEST NEGATIVE** — 13th negative in 91-141 audit.

## Summary

Tested **Adaptive Time-Constant** (Graves 2016, Adaptive Computation
Time) for CfC. The mechanism makes the per-neuron time constant a
function of the input:

```
tau = softplus(W_tau [x_t, h]) + 1.0     # [B, hidden_size]
f = sigmoid(W_f [x_t, h])                # [B, hidden_size]
g = tanh(W_g [x_t, h])
h_out = tanh(W_h [x_t, h])
decay = sigmoid(-f * tau)                # adaptive decay
h_new = decay * g + (1-decay) * h_out
```

**Verdict: HONEST NEGATIVE** — Both ATC variants LOSE on most
datasets. The f-gate already provides per-step time adaptation;
adding an explicit input-conditional tau is REDUNDANT.

## 1. Hypothesis

- **H1 (ATC helps on smooth data)**: with adaptive time constant,
  test_mse on `sin_irr` is < baseline. **REJECTED** (atc_concat
  ties, atc_input 1.6× worse).
- **H2 (ATC helps on structured data)**: with adaptive time
  constant, test_mse on `structured_irr` is < baseline. **REJECTED**
  (atc_concat 1.8× worse, atc_input 1.9× worse).
- **H3 (no regression on noisy data)**: with adaptive time constant,
  test_mse on `random_irr` is not worse than baseline by >10%.
  **REJECTED** (atc_concat 3.8× worse, atc_input 1.7× worse).

## 2. Implementation

`AdaptiveTimeConstantCfCCell` and `AdaptiveTimeConstantCfCStackedNetwork`
in `lnn/core/adaptive_time_constant_cfc.py` (~250 lines). 20 unit
tests covering init/forward/gradient/stability/tau-bounded/stacked/smoke.

Key design choices:

1. **Tau computed via softplus + 1.0** — keeps tau positive and
   bounded in [1, ∞).
2. **Init at time_scale_init=1.0** — bias = softplus_inv(0) = -inf,
   clamped to -log(1000) so initial tau ≈ 1.0.
3. **CfC recurrent step is unchanged** — only the time constant is
   made input-conditional.
4. **Per-layer ATC** — each stacked layer has its own tau network.
5. **Two modes**: `concat` (from [x, h]) and `input` (from x only).

## 3. Bench results (18 cells, 30 epochs, 2 seeds)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | **0.0094±0.0019** | **0.0053±0.0010** | **0.0013±0.0004** | 2545 |
| atc_concat | 0.0092±0.0016 | 0.0096±0.0001 | 0.0050±0.0004 | 3345 |
| atc_input | 0.0155±0.0074 | 0.0100±0.0056 | 0.0022±0.0013 | 2833 |

**Headline numbers**:

- `atc_concat` on `sin_irr`: TIES (0.0092 vs 0.0094)
- `atc_concat` on `structured_irr`: 1.8× worse (0.0053 → 0.0096)
- `atc_concat` on `random_irr`: 3.8× WORSE (0.0013 → 0.0050)

- `atc_input` on `sin_irr`: 1.6× worse (0.0094 → 0.0155)
- `atc_input` on `structured_irr`: 1.9× worse (0.0053 → 0.0100)
- `atc_input` on `random_irr`: 1.7× worse (0.0013 → 0.0022)

H1+H2+H3 all REJECTED.

## 4. Why it fails

### 4.1 The f-gate already provides per-step time adaptation

CfC's f-gate is per-hidden scalar over [x, h] that controls per-step
interpolation. The time constant `tau` multiplies f to control the
decay rate. The f-gate already adapts the effective time constant
per-step because:

- f is input-conditional (per [x, h])
- f changes per timestep (as input changes)
- The product `f * tau` gives the effective decay per neuron

Adding an explicit input-conditional tau is REDUNDANT with what the
f-gate already does.

### 4.2 The fixed time_scale=1.0 is the right inductive bias

A fixed time_scale forces the model to learn the appropriate time
scale during training. This is a "good" inductive bias — the model
has to learn the right scale, not adapt to a per-step varying scale.
With a fixed scale, the f-gate learns the per-step adaptation.

Making tau input-conditional removes this constraint and adds noise
(the tau computation depends on input which has noise).

### 4.3 The bias init issue

For `time_scale_init=1.0`, `softplus_inv(0) = log(exp(0) - 1) = -inf`.
The implementation has to clamp to a minimum, which means the
initial bias is some arbitrary large negative number. This makes
the early training unstable.

For `time_scale_init > 1.0`, the bias is well-defined.

### 4.4 Extra parameters overfit on noisy data

288-800 extra params (11-31% more) get used to fit noise on noisy
data, leading to 1.7-3.8× regression.

## 5. NEW INSIGHTS

1. **The f-gate already provides per-step time adaptation**. Adding
   an input-conditional tau is REDUNDANT — the f-gate changes per-step
   already.
2. **Fixed time_scale is the right inductive bias**. The model
   learns the appropriate scale during training, not adapts per-step.
3. **Softplus_inv has a singularity at y=0** — implementation must
   clamp to avoid -inf bias. This is a common gotcha.
4. **Pattern reinforced**: input-conditional modifications to the
   recurrent step (that aren't simply additive shortcuts) tend to
   lose — the f-gate is the right level of conditioning.

## 6. The 91-141 audit: 18 mechanism classes

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE (12 winners) |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Gated Input Skip (additive shortcut) | 134 | STRICTLY POSITIVE (13th winner) |
| Layer Normalization | 135 | TARGET-DEPENDENT (smooth only) |
| 1D Convolutional Input Preprocessing | 137 | TARGET-DEPENDENT (smooth wins, noisy catastrophic) |
| GLU + Identity Skip | 139 | TARGET-DEPENDENT (smooth wins, mild noisy regression) |
| **Adaptive Time-Constant (Graves 2016)** | **141** | **NEGATIVE (13th negative)** |
| SE Channel Attention | 140 | NEGATIVE (12th) |
| GLU alone (glu_basic) | 139 | NEGATIVE (11th) |
| Sinusoidal Time Embedding | 138 | NEGATIVE (10th) |
| Zoneout | 136 | NEGATIVE (9th) |
| Recursion depth (MoR) | 126 | NEGATIVE |
| 2nd-order oscillator | 128 | NEGATIVE |
| Multi-timescale ELM | 129 | NEGATIVE |
| Multi-rate MoE + dual attn | 130 | NEGATIVE |
| HGRN | 131 | NEGATIVE |
| AntisymmetricRNN | 132 | NEGATIVE |
| Hebbian Fast Weights | 133 | NEGATIVE |

**Pattern reinforced (13 winners + 3 target-dependent + 13 negatives)**:
- All 13 winners preserve the recurrent step + add useful structure
  (input-side: GIS, QuITE; expert-side: MoE; additive skip).
- 3 target-dependent (LN, 1D Conv, GLU+skip) all add input-side
  processing that helps smooth and is neutral/mild-regression on
  noisy.
- 13 negatives span alternatives (MoR, oscillator, etc.),
  unsupervised terms (FastWeights), regularizers (Zoneout),
  redundant info (time-emb), bottlenecks (glu_basic, SE), and now
  ATC (per-step time modification).

## 7. Recommendation

**Adaptive Time-Constant CfC is the 13th NEGATIVE in the 91-141 audit.**

- **DO NOT use ATC for 1D-CfC** — the f-gate already provides
  per-step time adaptation. Adding input-conditional tau is
  REDUNDANT.
- **Fixed time_scale is the right inductive bias** — the model
  learns the appropriate scale during training.
- **Stick with cfc baseline, GIS-CfC, glu_residual-CfC, or
  LN-CfC** for production.

## 8. Critical implementation details

1. **Tau = softplus(W [x, h]) + 1.0** — keeps tau positive and
   bounded in [1, ∞).
2. **Bias init for tau_net** — `bias = softplus_inv(time_scale_init
   - 1.0)`, clamped to a minimum to avoid -inf at time_scale_init=1.0.
3. **CfC recurrent step is unchanged** — only the time constant is
   made input-conditional.
4. **Per-layer ATC** — each stacked layer has its own tau network.
5. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements — torch imports work fine at runtime
   via `.venv312/bin/python`.
