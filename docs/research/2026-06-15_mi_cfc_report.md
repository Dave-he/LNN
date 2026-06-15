# Round 142 — Multiplicative Integration CfC (Wu et al. 2016 NIPS)

**Date**: 2026-06-15
**PRD**: #10-104
**Verdict**: **HONEST NEGATIVE** — 14th negative in 91-142 audit.

## Summary

Tested **Multiplicative Integration** (Wu et al. 2016, "On
Multiplicative Integration with Recurrent Neural Networks", NIPS
2016) for CfC. The mechanism replaces the additive integration
`W_x x + W_h h` with the element-wise product `W_x x ⊙ W_h h`:

```
# Standard additive CfC:
combined = W_x x + W_h h
f = σ(W_f combined + b_f)
g = tanh(W_g combined + b_g)
h_out = tanh(W_h combined + b_h)
h_t = σ(-f·τ) * g + (1 - σ(-f·τ)) * h_out

# Multiplicative CfC (this round):
x_proj = W_x x                # [B, hidden]
h_proj = W_h h                # [B, hidden]
inter = x_proj * h_proj       # [B, hidden]   <-- element-wise
f = σ(W_f inter + b_f)
g = tanh(W_g inter + b_g)
h_out = tanh(W_h inter + b_h)
h_t = σ(-f·τ) * g + (1 - σ(-f·τ)) * h_out
```

**Verdict: HONEST NEGATIVE** — Both MI variants LOSE on all 3
datasets. The mi_pure variant is catastrophic (3.6-19.4× worse);
the mi_x_residual variant is also worse (1.4-6.2×).

## 1. Hypothesis

- **H1 (MI helps on smooth data)**: with multiplicative
  integration, test_mse on `sin_irr` is < baseline.
  **REJECTED** (mi_pure 3.6× worse, mi_x_residual 1.4× worse).
- **H2 (MI helps on structured data)**: with multiplicative
  integration, test_mse on `structured_irr` is < baseline.
  **REJECTED** (mi_pure 5.2× worse, mi_x_residual 1.8× worse).
- **H3 (no regression on noisy data)**: with multiplicative
  integration, test_mse on `random_irr` is not worse than
  baseline by >10%. **REJECTED** (mi_pure 19.4× worse,
  mi_x_residual 6.2× worse).

## 2. Implementation

`MultiplicativeIntegrationCfCCell` and
`MultiplicativeIntegrationXResidualCfCCell` plus
`MultiplicativeIntegrationCfCStackedNetwork` in
`lnn/core/multiplicative_integration_cfc.py` (~250 lines).
26 unit tests covering init/forward/gradient/stability/stacked/
smoke.

Two variants:

1. **mi_pure**: `inter = x_proj ⊙ h_proj` (pure multiplicative)
2. **mi_x_residual**: `inter = x_proj ⊙ h_proj + x_proj` (additive
   x residual to handle the h=0 chicken-and-egg problem)

Both use the standard 3-branch CfC on `inter` (f_gate, g_branch,
h_branch) with the CfC time-constant `time_scale`.

Key design choices:

1. **Bias init for h=0 symmetry breaking**: `f_gate.bias = 1.0`,
   `g_branch.bias = 0.5`, `h_branch.bias = 0.5`. Without this,
   the multiplicative product is 0 when h=0, and h_new stays at 0.
2. **Same param count as baseline** (2545) — no extra params, the
   trade-off is structural (replaces concat-linear with
   element-wise product).
3. **Per-layer MI** — each stacked layer has its own x_proj, h_proj,
   and 3 gates.

## 3. Bench results (18 cells, 30 epochs, 2 seeds)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | **0.0094±0.0019** | **0.0053±0.0010** | **0.0013±0.0004** | 2545 |
| mi_pure | 0.0343±0.0021 | 0.0277±0.0050 | 0.0252±0.0111 | 2545 |
| mi_x_residual | 0.0128±0.0017 | 0.0095±0.0011 | 0.0080±0.0014 | 2545 |

**Headline numbers**:

- `mi_pure` on `sin_irr`: 3.6× worse (0.0094 → 0.0343)
- `mi_pure` on `structured_irr`: 5.2× worse (0.0053 → 0.0277)
- `mi_pure` on `random_irr`: **19.4× WORSE** (0.0013 → 0.0252)

- `mi_x_residual` on `sin_irr`: 1.4× worse (0.0094 → 0.0128)
- `mi_x_residual` on `structured_irr`: 1.8× worse (0.0053 → 0.0095)
- `mi_x_residual` on `random_irr`: 6.2× worse (0.0013 → 0.0080)

H1+H2+H3 all REJECTED for both variants.

## 4. Why it fails

### 4.1 Multiplicative amplifies noise

The product `x_proj * h_proj` is more sensitive to noise than the
additive `W_x x + W_h h`. On `random_irr` (noise-dominated), the
product amplifies noise and the model cannot recover. This is
exactly the failure mode that Wu et al. 2016 warned about for
noisy data.

### 4.2 The h=0 chicken-and-egg problem

When h=0 at t=0, the multiplicative product is 0. Even with bias
init (f_gate.bias=1.0), the gradient to h_proj is 0 (the
multiplicative product has h_proj as a multiplicative factor).
This creates a "cold start" issue that the model has to overcome
through training.

The x_residual variant partially mitigates this by adding x_proj
to the product, but the multiplicative product still contributes
noise.

### 4.3 Concat gives more information

The standard CfC uses `concat([x, h])` to give the gates access
to BOTH x and h, separately. The multiplicative integration
forces a specific interaction `W_x x ⊙ W_h h` which:
- Loses information about x and h individually (the product is
  one signal, not two)
- Forces the model to learn a multiplicative decomposition

For CfC, this is harmful because the gates already provide
per-hidden-dim conditioning via `f * τ`.

### 4.4 CfC's f-gate already provides multiplicative-like conditioning

CfC's f-gate is `σ(W_f [x, h])` and the time constant is
multiplicative in the decay: `decay = σ(-f * τ)`. The structure
is already multiplicative at the gate level. Adding multiplicative
integration at the input level is REDUNDANT and harmful.

## 5. NEW INSIGHTS

1. **Concat > element-wise product for 1D time-series**. The
   multiplicative integration idea works in language modeling
   (Wu 2016) but FAILS in 1D time-series with noise. The
   difference: language has high signal-to-noise, time-series
   often doesn't.
2. **Multiplicative amplifies noise**. The product `x * h` is
   more sensitive to noise than `x + h`. On random_irr, this
   is catastrophic (19.4× worse for mi_pure).
3. **The f-gate already provides per-step multiplicative-like
   conditioning**. `decay = σ(-f * τ)` is the multiplicative
   structure. Adding multiplicative at the input level is
   REDUNDANT.
4. **Pattern reinforced**: structural changes that REPLACE the
   input integration (vs ADD to it) lose — concat → element-wise
   product loses information.

## 6. The 91-142 audit: 19 mechanism classes

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE (12 winners) |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Gated Input Skip (additive shortcut) | 134 | STRICTLY POSITIVE (13th winner) |
| Layer Normalization | 135 | TARGET-DEPENDENT (smooth only) |
| 1D Convolutional Input Preprocessing | 137 | TARGET-DEPENDENT (smooth wins, noisy catastrophic) |
| GLU + Identity Skip | 139 | TARGET-DEPENDENT (smooth wins, mild noisy regression) |
| **Multiplicative Integration (Wu 2016)** | **142** | **NEGATIVE (14th negative)** |
| Adaptive Time-Constant (Graves 2016) | 141 | NEGATIVE (13th) |
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

**Pattern reinforced (13 + 3 + 14 = 30 tests)**:
- 13 winners all preserve recurrent step + add useful structure
  (input-side: GIS, QuITE; expert-side: MoE; additive skip).
- 3 target-dependent (LN, 1D Conv, GLU+skip) all add input-side
  processing that helps smooth and is neutral/mild-regression on
  noisy.
- 14 negatives span alternatives (MoR, oscillator, etc.),
  unsupervised terms (FastWeights), regularizers (Zoneout),
  redundant info (time-emb), bottlenecks (glu_basic, SE),
  per-step time modifications (ATC), and now **multiplicative
  integration** (replaces concat with element-wise product).

## 7. Recommendation

**Multiplicative Integration CfC is the 14th NEGATIVE in the
91-142 audit.**

- **DO NOT use MI-CfC for 1D time-series** — multiplicative
  interaction amplifies noise and the f-gate already provides
  per-step multiplicative-like conditioning.
- **Stick with cfc baseline, GIS-CfC, glu_residual-CfC, or
  LN-CfC** for production.
- **Wu 2016's MI-RNN works for language modeling** (high
  signal-to-noise) but FAILS for 1D time-series with noise.

## 8. Critical implementation details

1. **`inter = x_proj ⊙ h_proj`** for pure MI, or
   **`inter = x_proj ⊙ h_proj + x_proj`** for x-residual variant.
2. **Bias init for h=0 symmetry breaking**: f_gate.bias=1.0,
   g_branch.bias=0.5, h_branch.bias=0.5.
3. **CfC gates (f_gate, g_branch, h_branch) operate on `inter`**
   — the same as the standard CfC.
4. **Per-layer MI** — each stacked layer has its own x_proj,
   h_proj, and 3 gates.
5. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements — torch imports work fine at runtime
   via `.venv312/bin/python`.
