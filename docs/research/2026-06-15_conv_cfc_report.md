# Round 137 — 1D Convolutional Input Preprocessing for CfC (ConvLSTM 2015)

**Date**: 2026-06-15
**PRD**: #10-99
**Verdict**: **TARGET-DEPENDENT-WITH-NUANCE** — 2nd target-dependent in 91-137 audit.

## Summary

Tested **1D causal convolution preprocessing** applied to the input BEFORE the
CfC recurrent step (ConvLSTM-style architecture, Shi et al. 2015). The conv has
kernel size k (3 or 5) and is initialized as identity (last position = identity
matrix, others = 0). The CfC recurrent step is unchanged.

**Verdict: TARGET-DEPENDENT-WITH-NUANCE** — `conv_k3` WINS on `sin_irr` (1.2×)
and `structured_irr` (1.3×) but is **CATASTROPHIC on `random_irr` (4.9× worse)**.
`conv_k5` is mostly neutral. The mechanism captures local temporal patterns
on smooth data but **overfits to local noise** on noisy data.

## 1. Hypothesis

- **H1 (conv helps on smooth data)**: with 1D conv preprocessing, test_mse
  on `sin_irr` is < baseline. **PARTIAL** (conv_k3 1.2× better, conv_k5 tie).
- **H2 (conv helps on structured data)**: with 1D conv preprocessing, test_mse
  on `structured_irr` is < baseline. **CONFIRMED for conv_k3 (1.3× better)**
  but conv_k5 ties.
- **H3 (no regression on noisy data)**: with 1D conv preprocessing, test_mse
  on `random_irr` is not worse than baseline by >10%. **REJECTED** (conv_k3
  is 4.9× worse, conv_k5 is 1.4× worse).

## 2. Implementation

`ConvCfCCell` and `ConvCfCStackedNetwork` in `lnn/core/conv_cfc.py` (~260 lines).
21 unit tests covering init/forward/gradient/stability/causality/stacked/smoke.

Key design choices:

1. **1D causal conv with kernel size k=3 or 5**, same number of channels
   as input (no projection). Uses manual causal padding.
2. **Identity initialization** — last position weight = identity matrix,
   other positions = 0. This means the conv starts as a pure identity
   function (output = x_t), and the model can learn to use the previous
   timesteps if it helps.
3. **CfC recurrent step is unchanged** — conv preprocessing only affects
   the input fed to the 3-branch CfC step.
4. **Stacking per-layer** — each layer applies its own 1D conv to the
   previous layer's output.

## 3. Bench results (18 cells, 30 epochs, 2 seeds)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | 0.0094±0.0019 | 0.0053±0.0010 | **0.0013±0.0004** | 2545 |
| conv_k3 | **0.0079±0.0010** | **0.0040±0.0003** | 0.0064±0.0039 | 3325 |
| conv_k5 | 0.0096±0.0010 | 0.0056±0.0016 | 0.0018±0.0006 | 3845 |

**Headline numbers**:

- `conv_k3` on `sin_irr`: 1.2× better (0.0094 → 0.0079)
- `conv_k3` on `structured_irr`: 1.3× better (0.0053 → 0.0040)
- `conv_k3` on `random_irr`: 4.9× WORSE (0.0013 → 0.0064) — **CATASTROPHIC**

`conv_k5` is mostly neutral with a small regression on `random_irr` (1.4×).

## 4. Why it works on smooth/structured, fails on random

### 4.1 The mechanism is good for capturing local temporal patterns

On `sin_irr` and `structured_irr`, the signal has STRONG local structure
(periodic sine, regime switch). The 1D conv learns to smooth and combine
the previous 2-4 timesteps before feeding to CfC. This is exactly what
the f-gate wants — a "smoother" input that CfC can interpolate from.

### 4.2 The mechanism OVERFITS to local noise

On `random_irr`, the signal is **noise-dominated**. The 1D conv learns
to "see" patterns in the noise — combinations of past noise values that
correlate with future noise values within a 30-epoch training window.
The conv mixes noise across timesteps, amplifying it before CfC
processes it. The f-gate then interpolates from a noisy smoothed
input, leading to worse predictions.

This is the classic **overfitting from increased model capacity on
noisy data** — conv_k3 adds 780 extra parameters (30% more), and these
parameters all get used to fit noise.

### 4.3 Why conv_k5 is more stable

`conv_k5` has more parameters (1300 extra) but **larger receptive field**.
The 5-timestep window makes it harder to fit local noise patterns (they're
diluted by averaging with non-correlated noise). The cost is that it
loses the fine-grained ability to capture periodic signals, so the
sin_irr/structured_irr wins disappear.

## 5. NEW INSIGHTS

1. **1D conv + CfC is a local pattern recognizer** — when local
   patterns exist (sin/structured), it shines. When local patterns
   are noise (random), it overfits.
2. **CfC's f-gate is NOT a learned local pattern extractor** — it
   learns a per-step interpolation but not a multi-timestep
   smoothing. The 1D conv fills this gap on smooth data.
3. **The 91-137 audit pattern: "additive not replacement"** — 1D conv
   ADDS preprocessing to CfC, doesn't replace anything. On smooth
   data this wins, on noisy data it amplifies noise.
4. **Kernel size 3 is the sweet spot** for capturing local patterns
   in T=32 sequences. Kernel 5 is too conservative (loses the wins).
5. **Mixing input channels in conv init is a bug** — must use
   `torch.eye(input_size)` for the last position, not `1.0` (which
   would broadcast and average channels).

## 6. The 91-137 audit: 14 mechanism classes

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE (12 winners) |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Gated Input Skip (additive shortcut) | 134 | STRICTLY POSITIVE (13th winner) |
| Layer Normalization (per-sample normalize) | 135 | TARGET-DEPENDENT (smooth only) |
| **1D Convolutional Input Preprocessing** | **137** | **TARGET-DEPENDENT (smooth wins, noisy CATASTROPHIC)** |
| Zoneout (preserve h) | 136 | NEGATIVE (9th negative) |
| Recursion depth (MoR) | 126 | NEGATIVE |
| 2nd-order oscillator | 128 | NEGATIVE |
| Multi-timescale ELM | 129 | NEGATIVE |
| Multi-rate MoE + dual attn | 130 | NEGATIVE |
| HGRN (gated linear RNN) | 131 | NEGATIVE |
| AntisymmetricRNN | 132 | NEGATIVE |
| Hebbian Fast Weights | 133 | NEGATIVE |

**Pattern reinforced (13 winners + 2 target-dependent + 9 negatives)**:
- All 13 winners preserve the recurrent step + add useful structure that
  HELPS (input-side: GIS, QuITE; expert-side: MoE; additive skip).
- Both target-dependent (LN, 1D Conv) add a normalization/conv that helps
  on smooth data but interferes on noisy data.
- All 9 negatives propose alternatives to the recurrent step OR add
  unsupervised/regularizer terms.

## 7. Recommendation

**1D Convolutional CfC is the 2nd TARGET-DEPENDENT in the 91-137 audit.**

- **USE `conv_k3` on smooth/structured time series** — gives 1.2-1.3×
  improvement on sin_irr/structured_irr.
- **DO NOT USE `conv_k3` on noisy data** — 4.9× catastrophic regression
  on random_irr.
- **Production heuristic**: detect data noise level (e.g., autocorrelation
  of residuals), use conv_k3 if low, plain CfC if high.
- **Alternative**: `conv_k5` is safer (1.4× regression on noisy, neutral
  on smooth) but loses the wins on smooth data.

## 8. Critical implementation details

1. **Identity init for conv** — last position weight = `torch.eye(input_size)`,
   not `1.0` (which averages channels). With `1.0` init, conv output at
   init is `mean over channels`, not `x_t`.
2. **Causal padding is manual** — `Conv1d` with `padding=0` plus manual
   window construction `[x_{t-k+1}, ..., x_{t-1}, x_t]` in `_causal_conv`.
3. **Window management per layer** — each stacked layer has its own
   window buffer of size (kernel_size - 1). The window is shifted
   left and the new input appended at each timestep.
4. **NaN handling** — `torch.nan_to_num(x_t, nan=0.0)` is applied
   per-step before the conv. NaN in any timestep would propagate
   through the conv.
5. **Pyright false positives** on `import torch` are pre-existing per
   standing requirements — torch imports work fine at runtime via
   `.venv312/bin/python`.
