# PRD #10-103 — Adaptive Time-Constant CfC (Round 141)

**Date**: 2026-06-15
**Round**: 141 (response to adaptive computation time, Graves 2016)
**Status**: Drafted.

## 1. Why round 141

CfC's `time_constant` parameter (a.k.a. `time_scale` in our impl) is
per-neuron but **FIXED across timesteps**. Every timestep, every
neuron uses the same time constant. This is a known limitation:

- On smooth data, large time constants (slow updates) work well
- On regime switches, small time constants (fast updates) work well
- A FIXED time constant can't adapt to BOTH

**Adaptive Time-Constant** (Graves 2016, "Adaptive Computation Time
for Recurrent Neural Networks" — spirit) makes the time constant a
function of the input:

```
tau = softplus(W_tau [x_t, h]) + 1.0     # [B, hidden_size]
decay = sigmoid(-f * tau)                  # adaptive per (batch, t, neuron)
h_new = decay * g + (1-decay) * h_out
```

This is structurally:
- Additive (preserves the recurrent step entirely)
- Input-conditional (a new axis of variation, not redundant with f-gate)
- Per-neuron (matches CfC's per-neuron time_scale granularity)

## 2. Mechanism

```
tau = softplus(W_tau [x_t, h]) + 1.0     # [B, hidden_size], positive
f = sigmoid(W_f [x_t, h])                # [B, hidden_size] in [0, 1]
g = tanh(W_g [x_t, h])                   # [B, hidden_size]
h_out = tanh(W_h [x_t, h])               # [B, hidden_size]
decay = sigmoid(-f * tau)                # adaptive decay
h_new = decay * g + (1-decay) * h_out
```

Two variants:
- `atc_concat`: tau from concat [x, h] (cross-attention style)
- `atc_input`: tau from input only (cheaper, 278 fewer params)

## 3. Hypotheses

- **H1 (ATC helps on smooth data)**: with adaptive time constant,
  test_mse on `sin_irr` is < baseline.
- **H2 (ATC helps on structured data)**: with adaptive time
  constant, test_mse on `structured_irr` is < baseline (regime
  switch is more detectable with adaptive tau).
- **H3 (no regression on noisy data)**: with adaptive time constant,
  test_mse on `random_irr` is not worse than baseline by >10%.

## 4. Why this should win per the 91-140 audit

The audit shows:
- 13 STRICTLY POSITIVE winners all preserve the recurrent step + add
  useful structure (input-side processing, MoE experts, additive
  shortcuts).
- 3 TARGET-DEPENDENT (LN 135, conv 137, glu_residual 139).
- 12 negatives propose alternatives, add unsupervised terms, add
  regularizers, add redundant info, or create bottlenecks.

Adaptive time-constant:
- **Preserves the recurrent step** entirely.
- **Adds useful structure** — input-conditional time constant is a
  NEW axis of variation that the f-gate doesn't provide.
- **Is structural** — modifies the time constant, not the
  recurrent step.
- **Different from f-gate** — f-gate is per-hidden scalar over
  [x, h], tau is per-hidden scalar over [x, h] BUT multiplies f
  (acts on the decay rate, not the gate value directly).

The risk: tau computation adds parameters that could overfit on
noisy data, but the softplus is bounded in [1, ∞) so it can't
zero out the time scale (unlike sigmoid gates).

## 5. Plan

### 5.1 Implementation (`lnn/core/adaptive_time_constant_cfc.py`)

Two classes:
- `AdaptiveTimeConstantCfCCell(nn.Module)`: standard 3-branch CfC
  cell with input-conditional time constant.
- `AdaptiveTimeConstantCfCStackedNetwork(nn.Module)`: 2-layer stack
  with ATC on each layer.

Key design choices:
- Softplus + 1.0 to keep tau positive (and >= 1.0).
- Tau from concat [x, h] (default) or input only.
- CfC recurrent step is otherwise unchanged.
- Per-layer ATC.

### 5.2 Tests (`tests/test_adaptive_time_constant_cfc.py`)

20+ unit tests covering:
- Init: ATC parameters.
- Forward: shape preservation.
- Tau: tau is positive (>= 1.0).
- Gradient: flows to tau weights.
- Stacked: gradient flows to all layers.
- Smoke: learns toy sin.
- Sanity: tau is input-conditional (different x gives different tau).

### 5.3 Bench (`scripts/bench_adaptive_time_constant_cfc.py`)

18 cells (3 conds × 3 datasets × 2 seeds × 30 epochs):
- `cfc` (baseline)
- `atc_concat` (tau from concat [x, h])
- `atc_input` (tau from input only)

Datasets: sin_irr, structured_irr, random_irr (D=2, missing_rate=0.3)

## 6. Expected outcomes

- **Best case (~45%)**: H1 + H2 + H3 all confirmed. ATC is the
  **14th STRICTLY POSITIVE** winner.
- **Likely case (~35%)**: H1 + H3 confirmed, H2 partial.
  **TARGET-DEPENDENT-WITH-NUANCE** (helps smooth, neutral noisy).
- **Worst case (~20%)**: All 3 hypotheses rejected. The f-gate
  already provides enough adaptation. 13th negative.

## 7. Why this is worth testing

The 91-140 audit strongly suggests mechanisms that ADD useful
structure to the recurrent step (input-side, expert-side, additive
skip) win. Adaptive time-constant is a 5-line addition that adds a
new axis of variation (per-step tau) that the f-gate doesn't have.
If it wins, it would be a high-confidence production candidate
(very simple, well-motivated, structurally clean).

## 8. Files to create

- `lnn/core/adaptive_time_constant_cfc.py` (~200 lines)
- `tests/test_adaptive_time_constant_cfc.py` (~300 lines, 20+ tests)
- `scripts/bench_adaptive_time_constant_cfc.py` (~250 lines, 18 cells)
- `docs/research/2026-06-15_adaptive_time_constant_cfc_report.md`
