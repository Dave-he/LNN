# PRD #10-101 — Gated Linear Unit (GLU) Input Modulation for CfC (Round 139)

**Date**: 2026-06-15
**Round**: 139 (response to GLU literature, Dauphin 2017 + LSTM input gates)
**Status**: Drafted.

## 1. Why round 139

**Gated Linear Units (GLU)** (Dauphin et al. 2017, "Language Modeling
with Gated Convolutional Networks") are a simple but powerful gating
mechanism:

```
GLU(x) = sigmoid(W_1 x) * W_2 x
```

The sigmoid gate modulates which input features are "let through" to
the next layer. This is essentially an LSTM-style input gate applied
to a feedforward layer.

For CfC, the f-gate is shared across [x, h] and produces a single
scalar per hidden dim. This means the same gate controls BOTH input
and hidden contribution. A separate input gate (GLU) would give
the cell finer control: it can decide per-feature whether to let x
through, BEFORE the f-gate mixes it with h.

This is structurally:
- Additive (preserves recurrent step)
- Input-side (modulates x before CfC)
- LSTM-style input gate (a known winner in RNN literature)

## 2. Mechanism

```
x_gate = sigmoid(W_gate x_t)          # [B, D_in] in [0, 1]
x_gated = x_gate * x_t                # [B, D_in] modulated input
h_t = cf_c_step(x_gated, h_{t-1})     # standard 3-branch CfC
```

Three variants:
- `glu_basic`: simple input gate (single linear)
- `glu_residual`: input gate + identity skip (gated + identity, LSTM-style)
- `glu_per_feature`: per-feature sigmoid (each input dim has its own gate)

## 3. Hypotheses

- **H1 (GLU helps on smooth data)**: with GLU input modulation,
  test_mse on `sin_irr` is < baseline.
- **H2 (GLU helps on structured data)**: with GLU input modulation,
  test_mse on `structured_irr` is < baseline.
- **H3 (no regression on noisy data)**: with GLU input modulation,
  test_mse on `random_irr` is not worse than baseline by >10%.

## 4. Why this should win per the 91-138 audit

The audit shows:
- 13 STRICTLY POSITIVE winners all preserve the recurrent step + add
  useful structure (MoE experts, input-side processing, additive
  shortcuts).
- 2 TARGET-DEPENDENT (LN 135, conv 137).
- 10 negatives propose alternatives to the recurrent step, add
  unsupervised terms, add regularizers, or add redundant info.

GLU input modulation:
- **Preserves the recurrent step** entirely.
- **Adds useful input-side structure** — per-feature input gate that
  the recurrent step can use.
- **Is structural** — modifies the input, not the recurrent step.
- **Similar to LSTM input gate** (a known winner in RNN literature).
- **Different from CfC's f-gate** — f-gate is per-hidden-dim scalar
  over [x, h], GLU is per-input-feature scalar over x only.

The risk: GLU adds parameters that could overfit on noisy data, but
LSTM-style input gating is well-established as a robust mechanism.

## 5. Plan

### 5.1 Implementation (`lnn/core/gated_linear_unit_cfc.py`)

Two classes:
- `GatedLinearUnitCfCCell(nn.Module)`: standard 3-branch CfC cell
  with GLU input modulation.
- `GatedLinearUnitCfCStackedNetwork(nn.Module)`: 2-layer stack with
  GLU on each layer's input.

Key design choices:
- 1D conv-free, pure linear input gate.
- Sigmoid bounded in [0, 1].
- CfC recurrent step is unchanged.
- gate_dim = input_size (per-feature gate).

### 5.2 Tests (`tests/test_gated_linear_unit_cfc.py`)

20+ unit tests covering:
- Init: GLU gate parameters.
- Forward: shape preservation.
- Gate: output is bounded in [0, 1] for input gate.
- Gradient: flows to GLU weights.
- Stacked: gradient flows to all layers.
- Smoke: learns toy sin.
- Sanity: gate=0 zero out the input, gate=1 pass through.

### 5.3 Bench (`scripts/bench_gated_linear_unit_cfc.py`)

18 cells (3 conds × 3 datasets × 2 seeds × 30 epochs):
- `cfc` (baseline)
- `glu_basic` (per-feature sigmoid gate)
- `glu_residual` (gated + identity skip)

Datasets: sin_irr, structured_irr, random_irr (D=2, missing_rate=0.3)

## 6. Expected outcomes

- **Best case (~45%)**: H1 + H2 + H3 all confirmed. GLU is the
  **14th STRICTLY POSITIVE** winner.
- **Likely case (~35%)**: H1 + H3 confirmed, H2 partial.
  **TARGET-DEPENDENT-WITH-NUANCE** (helps smooth/structured,
  neutral noisy).
- **Worst case (~20%)**: All 3 hypotheses rejected. The f-gate
  already provides per-feature modulation. 11th negative.

## 7. Why this is worth testing

The 91-138 audit strongly suggests "input-side processing + add to
recurrent step" mechanisms win. QuITE+MoE (round 103), GIS (round 134)
were winners. GLU is a 5-line addition that could be a 14th winner.
If it wins, it would be a high-confidence production candidate
(very simple, well-established in LSTM literature).

## 8. Files to create

- `lnn/core/gated_linear_unit_cfc.py` (~200 lines)
- `tests/test_gated_linear_unit_cfc.py` (~300 lines, 20+ tests)
- `scripts/bench_gated_linear_unit_cfc.py` (~250 lines, 18 cells)
- `docs/research/2026-06-15_gated_linear_unit_cfc_report.md`
