# PRD #10-96 — Gated Input Skip for CfC (Round 134)

**Date**: 2026-06-15
**Round**: 134 (response to Highway Networks, Srivastava, Greff, Schmidhuber, 2015, arXiv:1505.00387)
**Status**: Drafted.

## 1. Why round 134

The **Highway Networks** paper (Srivastava, Greff, Schmidhuber, 2015)
introduced a learned gated skip connection that lets a neural network
decide per-step how much to use a "transform" path vs a "carry" path.
The mechanism is:

```
y = H(x, W_H) * T(x, W_T) + x * C(x, W_C)
```

where `H` is the transform, `T` is the transform gate, and `C` is the
carry gate. When `T = 1` and `C = 0`, the layer is purely transform.
When `T = 0` and `C = 1`, the layer is purely carry (skip).

This is the foundation of ResNet-style skip connections and has been
shown to help with very deep networks. The question for round 134 is:
**does a gated input skip help CfC on 1D time series?**

### 1.1 Mechanism

For CfC, we add a learnable skip from input to the hidden state update:

```
h_new = cf_c(x, h)        # standard CfC step
skip  = W_skip @ x         # input skip projection
gate  = sigmoid(W_gate @ [x, h])   # input-conditional gate
h_t   = h_new + gate * skip         # gated skip
```

The skip provides a DIRECT path from input to hidden state, bypassing
the recurrent dynamics. The gate controls when to use this leak.

### 1.2 Why this should win per the 91-133 audit

The audit shows:
- 12 STRICTLY POSITIVE winners all preserve the recurrent step + add
  useful structure (MoE experts, input-side processing).
- 8 negatives propose alternatives to the recurrent step (HGRN,
  Antisymm, etc.) or add unsupervised terms (FastWeights).

Gated Input Skip:
- **Preserves W·h** and CfC's f-gate (the recurrent step is unchanged).
- **Adds a useful structure** — the input skip provides a low-pass
  filter on the input that can bypass the recurrent dynamics when
  useful.
- **Is structural** — modifies the recurrent step's output.

The risk: the skip might add noise to the hidden state, similar to
FastWeights' F@h term.

## 2. Hypotheses

- **H1 (skip helps on noisy data)**: with the skip, test_mse on
  `random_irr` is < unconstrained CfC baseline (because the skip
  provides a direct path from input to hidden state that bypasses
  noisy recurrent dynamics).
- **H2 (skip helps on regime switching)**: with the skip, test_mse
  on `structured_irr` (regime switch at T/2) is < baseline.
- **H3 (no regression on smooth data)**: with the skip, test_mse on
  `sin_irr` is not worse than baseline by >10%.

## 3. Plan

### 3.1 Implementation (`lnn/core/gated_input_skip_cfc.py`)

Two classes:
- `GatedInputSkipCfCCell(nn.Module)`: single recurrent step with
  the gated input skip.
- `GatedInputSkipCfCStackedNetwork(nn.Module)`: 2-layer stack.

Key design choices:
- Skip is a linear projection W_skip: input_size -> hidden_size.
- Gate is a linear + sigmoid: (input_size + hidden_size) -> hidden_size.
- Skip is added to the recurrent step's output (not replaced).
- The gate depends on both x and h, so it can be input-conditional.

### 3.2 Tests (`tests/test_gated_input_skip_cfc.py`)

20+ unit tests covering:
- Init: skip and gate are initialized.
- Forward: shape preservation.
- Forward: h stays bounded.
- Skip in isolation: when gate=1, output = cf_c + W_skip x.
- Skip zero: when gate=0, output = cf_c.
- Gradient: flows to W_skip and W_gate.
- Stacked: gradient flows to all layers.
- Smoke: learns toy sin.

### 3.3 Bench (`scripts/bench_gated_input_skip_cfc.py`)

18-24 cells (3 conds × 3 datasets × 2 seeds × 30 epochs):
- `cfc` (baseline)
- `gis_weak` (small init for W_skip)
- `gis_strong` (large init for W_skip)
- `gis_init_one` (init gate bias to +1.0, making gate initially 0.73)

Datasets: sin_irr, structured_irr, random_irr (D=2, missing_rate=0.3)

## 4. Expected outcomes

- **Best case (probability ~30%)**: H1 + H2 + H3 all confirmed. Gated
  Input Skip is the **13th STRICTLY POSITIVE** winner. The skip
  provides useful shortcut connections.
- **Likely case (probability ~50%)**: H3 confirmed, H1/H2 partial.
  **TARGET-DEPENDENT-WITH-NUANCE**.
- **Worst case (probability ~20%)**: All 3 hypotheses rejected. The
  skip adds noise similar to FastWeights. 19th negative.

## 5. Why this is worth testing

The 91-133 audit strongly suggests "additive + useful" mechanisms win.
Gated Input Skip is the cleanest "additive" mechanism I haven't tested.
The risk is the skip might add high-frequency noise (like FastWeights)
or be redundant with CfC's f-gate (which already provides per-step
interpolation).

## 6. Files to create

- `lnn/core/gated_input_skip_cfc.py` (~200 lines)
- `tests/test_gated_input_skip_cfc.py` (~300 lines, 20+ tests)
- `scripts/bench_gated_input_skip_cfc.py` (~250 lines, 18-24 cells)
- `docs/research/2026-06-15_gated_input_skip_cfc_report.md`
