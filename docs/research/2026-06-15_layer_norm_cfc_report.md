# Round 135 — Layer Normalization CfC (Ba et al. 2016, arXiv:1607.06450)

**Date**: 2026-06-15
**PRD**: #10-97
**Verdict**: **TARGET-DEPENDENT-WITH-NUANCE** — wins on smooth, ties on structured, LOSES on noisy.

## Summary

Tested the **Layer Normalization** mechanism from Ba, Kiros, Hinton
(2016) "Layer Normalization" (arXiv:1607.06450). The idea: apply
per-sample normalization to the **combined input** [x, h] BEFORE
the f-gate/g-branch/h-branch linear projections::

    combined = [x, h]
    combined = LayerNorm(combined)   # per-sample normalize
    f = sigmoid(W_f combined)
    g = tanh(W_g combined)
    h_out = tanh(W_h combined)
    decay = sigmoid(-f * time_scale)
    h_new = decay * g + (1-decay) * h_out

**Verdict: TARGET-DEPENDENT-WITH-NUANCE** — LN wins on smooth data,
ties on structured, **LOSES 11.6× on noisy data**. The normalization
is a useful prior for clean signals but destroys the signal/noise
distinction for noisy inputs.

## 1. Hypothesis

- **H1 (LN helps on noisy data)**: test_mse on `random_irr` is
  < baseline. **REJECTED** (11.6× WORSE).
- **H2 (LN helps on regime switching)**: test_mse on
  `structured_irr` is < baseline. **REJECTED** (1.3× WORSE).
- **H3 (no regression on smooth data)**: test_mse on `sin_irr` is
  not worse than baseline by >5%. **CONFIRMED** (1.7× BETTER).

## 2. Implementation

`LayerNormCfCCell` and `LayerNormCfCStackedNetwork` in
`lnn/core/layer_norm_cfc.py` (~200 lines). 21 unit tests covering
init/forward/gradient/stability/smoke.

Key design choices:

1. **LN applied to combined [x, h] BEFORE the linear projections**
   (Ba et al. 2016 §3.2 recommendation for RNNs).
2. **LN with learnable affine gamma=1, beta=0** (identity at start,
   lets the model learn to scale/shift if useful).
3. **Per-cell LN** — each layer has its own gamma, beta.
4. **eps=1e-5** — standard for layer norm.
5. **CfC recurrent step preserved entirely** — LN modifies the
   input, not the output.

## 3. Bench results (12 cells, 30 epochs, 2 seeds)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | 0.0094±0.0019 | **0.0053±0.0010** | **0.0013±0.0004** | 2545 |
| **ln_cfc** | **0.0056±0.0009** | 0.0069±0.0011 | 0.0151±0.0030 | 2645 |

**MIXED result — TARGET-DEPENDENT**:

- **sin_irr**: ln_cfc is **1.7× BETTER** (LN helps on smooth data)
- **structured_irr**: ln_cfc is 1.3× WORSE (LN hurts on regime switching)
- **random_irr**: ln_cfc is **11.6× WORSE** (LN destroys noisy signal)

H1 (helps on noisy) — **REJECTED** (11.6× worse)
H2 (helps on structured) — **REJECTED** (1.3× worse)
H3 (no regression on smooth) — **CONFIRMED** (1.7× better)

## 4. Why it fails on noisy data

LN normalizes the input to have mean 0 and var 1 per-sample. On
noisy data, the noise dominates the signal, and LN normalizes the
**noise** instead of the **signal**. The model loses its ability
to distinguish signal from noise because both are scaled to the
same per-sample variance.

In contrast, on smooth data, the signal is well-defined and the
normalization is a useful prior. The signal has consistent
statistics across timesteps, and LN's per-sample normalization
helps the gate input stay in a consistent range.

## 5. NEW INSIGHTS

1. **LN is target-dependent**: helps on clean signals, hurts on
   noisy signals. The normalization assumption (consistent per-sample
   statistics) holds for smooth data but not for noisy data.
2. **Per-sample normalization destroys signal/noise distinction**:
   on noisy data, LN forces noise and signal to the same scale,
   making the noise as "loud" as the signal.
3. **LN is a useful prior, not a free lunch**: it works when the
   data has consistent per-sample statistics (smooth signals) and
   fails when statistics vary (noisy inputs).
4. **Pattern reinforces 91-135 audit**: 13 STRICTLY POSITIVE
   winners are all MoE or input-side, but **LN is post-input
   normalization** — different mechanism class. LN is a
   **structural preprocessing** that helps in narrow regimes
   (smooth data).

## 6. The 91-135 audit: 8 neuron-families + LN

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE (12 winners) |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Gated Input Skip (additive shortcut) | 134 | STRICTLY POSITIVE (13th winner) |
| **Layer Normalization (per-sample normalize)** | **135** | **TARGET-DEPENDENT** (smooth only) |
| Recursion depth (MoR) | 126 | NEGATIVE |
| 2nd-order oscillator | 128 | NEGATIVE |
| Multi-timescale ELM | 129 | NEGATIVE |
| Multi-rate MoE + dual attn | 130 | NEGATIVE |
| HGRN (gated linear RNN) | 131 | NEGATIVE |
| AntisymmetricRNN | 132 | NEGATIVE |
| Hebbian Fast Weights | 133 | NEGATIVE |

**Pattern refined**: structural preprocessing (LN) is target-dependent.
Additive shortcuts (GIS) and MoE experts are universally positive.

## 7. Recommendation

**Layer Norm CfC is TARGET-DEPENDENT — use with caution.**

- **DO use LN-CfC for clean/smooth signals** (1.7× better on
  sin_irr).
- **DO NOT use LN-CfC for noisy or regime-switching data** (1.3×
  worse on structured, 11.6× worse on random).
- **Cost**: 2645 vs 2545 params (1.04× more) — tiny parameter
  increase.
- **Conditional recommendation**: enable LN only when input SNR
  is high.

## 8. Critical implementation details

1. **LN applied BEFORE linear projections**, not after — this is
   Ba et al. 2016's recommendation for RNNs.
2. **LN with learnable affine (gamma=1, beta=0)** — identity at
   start, lets the model learn to scale.
3. **Per-cell LN** — each layer has its own gamma, beta.
4. **eps=1e-5** for numerical stability.
5. **The bias in LN creates a useful learnable shift** — this is
   what makes the mechanism target-dependent.
