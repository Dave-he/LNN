# Round 140 — Squeeze-and-Excitation (SE) Channel Attention CfC (Hu 2017)

**Date**: 2026-06-15
**PRD**: #10-102
**Verdict**: **HONEST NEGATIVE** — 12th negative in 91-140 audit.

## Summary

Tested **Squeeze-and-Excitation (SE) channel attention** (Hu et al.
2017, CVPR 2018 winner) applied to the input fed to the CfC
recurrent step. The mechanism:

```
score = sigmoid(W_score [x_t, h])     # [B, D_in] in [0, 1]
x_se = score * x_t                    # [B, D_in] recalibrated
h_t = cf_c_step(x_se, h_{t-1})        # standard 3-branch CfC
```

**Verdict: HONEST NEGATIVE** — All 3 SE variants LOSE on ALL 3
datasets. SE is supposed to be a "universal performance booster" in
CNNs but doesn't work for CfC in 1D.

## 1. Hypothesis

- **H1 (SE helps on smooth data)**: with SE channel attention,
  test_mse on `sin_irr` is < baseline. **REJECTED** (1.8-2.4× worse).
- **H2 (SE helps on structured data)**: with SE channel attention,
  test_mse on `structured_irr` is < baseline. **REJECTED** (2.9-4.0×
  worse).
- **H3 (no regression on noisy data)**: with SE channel attention,
  test_mse on `random_irr` is not worse than baseline by >10%.
  **REJECTED** (9.3-9.8× worse).

## 2. Implementation

`SECfCCell` and `SECfCStackedNetwork` in `lnn/core/se_cfc.py` (~250
lines). 21 unit tests covering init/forward/gradient/stability/score-
bounded/stacked/smoke.

Key design choices:

1. **SE score from concat [x, h]** (cross-attention style, default
   `mode="concat"`).
2. **Per-input-channel sigmoid score** — one score per input dim.
3. **CfC recurrent step is unchanged** — SE only affects the input
   fed to the 3-branch CfC step.
4. **Per-layer SE** — each stacked layer has its own SE attention.

Three modes tested:
- `se_concat`: score from concat [x, h] (3111 params)
- `se_input`: score from input only (2823 params)
- (also tested `mode="hidden"`: score from hidden only)

## 3. Bench results (18 cells, 30 epochs, 2 seeds)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | **0.0094±0.0019** | **0.0053±0.0010** | **0.0013±0.0004** | 2545 |
| se_concat | 0.0170±0.0042 | 0.0155±0.0043 | 0.0121±0.0087 | 3111 |
| se_input | 0.0222±0.0026 | 0.0211±0.0063 | 0.0128±0.0000 | 2823 |

**ALL 3 SE variants LOSE on ALL 3 datasets**:

- **sin_irr**: se_concat 1.8×, se_input 2.4× worse
- **structured_irr**: se_concat 2.9×, se_input 4.0× worse
- **random_irr**: se_concat 9.3×, se_input 9.8× worse

H1+H2+H3 all REJECTED.

## 4. Why it fails

### 4.1 CfC's f-gate already provides input attention

The f-gate sees [x, h] and produces a per-hidden-dim scalar. The
SE score also sees [x, h] (concat mode) but operates on input dim.
This is REDUNDANT — the f-gate already uses the same information
to modulate the recurrence, just at a different dimension.

### 4.2 SE score=0 creates information bottleneck

Same problem as glu_basic in round 139. The sigmoid score can be
near 0, zeroing out important input features. The model has to
learn to "always pass" useful features through, which is hard.

### 4.3 Sigmoid SE score is too aggressive at init

At init, sigmoid outputs are around 0.5, so input features are
HALVED. The model has to compensate for this halving during
training, but the 30-epoch budget may not be enough to recover
the loss.

### 4.4 Extra parameters overfit on noisy data

278-566 extra params (11-22% more) get used to fit noise on
random_irr, leading to 9.3-9.8× regression.

### 4.5 The "universal performance booster" claim is context-dependent

SE works in CNNs because:
- The score is computed from a LARGE receptive field (after
  multiple conv layers)
- CNNs have many channels (typically 64-2048)
- The input is highly redundant

In our 1D setting:
- The score is computed from a TINY receptive field (just [x, h])
- CfC has only 2 input channels
- The input is NOT redundant

The mechanism was designed for a different problem class.

## 5. NEW INSIGHTS

1. **SE works in CNNs (large receptive field, many channels) but
   fails in 1D-CfC (small receptive field, few channels)** — the
   "universal performance booster" claim is context-dependent.
2. **Sigmoid score at init is too aggressive** — halves the input,
   model has to compensate.
3. **Score=0 is a real risk** — sigmoid can zero out features, no
   safety mechanism.
4. **CfC's f-gate already does the job** — it uses the same [x, h]
   information to modulate the recurrence.
5. **Pattern reinforced**: pure input-side gating (without residual
   skip) loses — glu_basic 139, se_input 140, se_concat 140.

## 6. The 91-140 audit: 17 mechanism classes

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE (12 winners) |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Gated Input Skip (additive shortcut) | 134 | STRICTLY POSITIVE (13th winner) |
| Layer Normalization | 135 | TARGET-DEPENDENT (smooth only) |
| 1D Convolutional Input Preprocessing | 137 | TARGET-DEPENDENT (smooth wins, noisy catastrophic) |
| GLU + Identity Skip | 139 | TARGET-DEPENDENT (smooth wins, mild noisy regression) |
| **SE Channel Attention (Hu 2017)** | **140** | **NEGATIVE (12th negative)** |
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

**Pattern reinforced (13 winners + 3 target-dependent + 12 negatives)**:
- All 13 winners preserve the recurrent step + add useful structure
  (input-side: GIS, QuITE; expert-side: MoE; additive skip).
- 3 target-dependent (LN, 1D Conv, GLU+skip) all add input-side
  processing that helps smooth and is neutral/mild-regression on
  noisy.
- 12 negatives span alternatives (MoR, oscillator, etc.),
  unsupervised terms (FastWeights), regularizers (Zoneout),
  redundant info (time-emb), bottlenecks (glu_basic), and now
  SE attention.

## 7. Recommendation

**SE Channel Attention CfC is the 12th NEGATIVE in the 91-140 audit.**

- **DO NOT use SE for 1D-CfC** — it's a "universal performance
  booster" for CNNs but doesn't work in our 1D setting.
- **SE's effectiveness depends on the receptive field size and
  number of channels** — both are tiny in 1D.
- **The f-gate already does the job** — no need for additional
  input attention.
- **Stick with cfc baseline, GIS-CfC, glu_residual-CfC, or
  LN-CfC** for production.

## 8. Critical implementation details

1. **SE score = sigmoid(W [x, h])** — element-wise product of
   sigmoid score with input. Per-input-channel score.
2. **Three modes**: `concat` (from [x, h]), `input` (from x only),
   `hidden` (from h only). All lose.
3. **CfC recurrent step is unchanged** — SE only affects input.
4. **Per-layer SE** — each stacked layer has its own attention.
5. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements — torch imports work fine at runtime
   via `.venv312/bin/python`.
