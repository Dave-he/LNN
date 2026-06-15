# Round 138 — Sinusoidal Time Embedding CfC (Vaswani 2017)

**Date**: 2026-06-15
**PRD**: #10-100
**Verdict**: **HONEST NEGATIVE** — 10th negative in 91-138 audit.

## Summary

Tested **sinusoidal time embedding** (Transformer-style positional
encoding) applied to the input BEFORE the CfC recurrent step. The
embedding is computed as:

```
t_emb = [sin(2π t/T * w_i), cos(2π t/T * w_i)] for i in 1..D_te/2
x_aug = concat([x_t, t_emb])
h_t = cf_c_step(x_aug, h_{t-1})
```

**Verdict: HONEST NEGATIVE** — All 3 time-emb variants LOSE on ALL
3 datasets. The cell already has implicit time information through
its recurrent state, and adding explicit time embedding is
redundant.

## 1. Hypothesis

- **H1 (time emb helps on smooth data)**: with time embedding, test_mse
  on `sin_irr` is < baseline. **REJECTED** (1.2-1.4× worse).
- **H2 (time emb helps on structured data)**: with time embedding,
  test_mse on `structured_irr` is < baseline. **REJECTED** (1.4-1.6×
  worse).
- **H3 (no regression on noisy data)**: with time embedding, test_mse
  on `random_irr` is not worse than baseline by >10%. **REJECTED**
  (3.3-4.5× worse).

## 2. Implementation

`SinusoidalTimeEmbCfCCell` and `SinusoidalTimeEmbCfCStackedNetwork`
in `lnn/core/sinusoidal_time_emb_cfc.py` (~230 lines). 25 unit tests
covering init/forward/gradient/stability/embedding-utility/stacked/smoke.

Key design choices:

1. **Sinusoidal embedding (Vaswani 2017)** with 4 or 8 dimensions,
   `max_period=10000.0` (Transformer default).
2. **Concat to input** — `x_aug = [x_t, t_emb]`, the cell's
   projection then sees both.
3. **CfC recurrent step is unchanged** — time embedding only
   affects the input fed to the 3-branch CfC step.
4. **Time is normalized to [0, 1]** — `t_norm = t / (T - 1)`.
5. **Per-layer time embedding** — each stacked layer has its own
   time embedding (concatenated to its own input).

## 3. Bench results (18 cells, 30 epochs, 2 seeds)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | **0.0094±0.0019** | **0.0053±0.0010** | **0.0013±0.0004** | 2545 |
| te_d4 | 0.0135±0.0013 | 0.0072±0.0002 | 0.0043±0.0013 | 2929 |
| te_d8 | 0.0116±0.0003 | 0.0086±0.0014 | 0.0059±0.0007 | 3313 |

**ALL 3 time-emb variants LOSE on ALL 3 datasets**:

- **sin_irr**: te_d4 1.4×, te_d8 1.2× worse
- **structured_irr**: te_d4 1.4×, te_d8 1.6× worse
- **random_irr**: te_d4 3.3×, te_d8 4.5× worse

H1+H2+H3 all REJECTED.

## 4. Why it fails

### 4.1 The cell already has implicit time information

The f-gate's `time_constant` is per-neuron and the cell's state
evolves over time. By step t, the hidden state `h_t` already
encodes the cumulative time history (it's a learned ODE state). The
f-gate uses `sigmoid(-f * time_scale * dt)` to modulate per-step
updates, which is essentially a learned time-aware interpolation.

Adding an explicit time embedding is REDUNDANT — the cell already
knows what timestep it's at, just implicitly through its state.

### 4.2 Sinusoidal embedding is non-adaptive

The sinusoidal embedding is FIXED (no learnable parameters in the
embedding itself). It just provides a "clock signal" that says
"you're at time t". The cell can already derive this from the
recurrence — each step's input is processed and propagated through
the hidden state, building up a "where am I in the sequence"
representation.

### 4.3 Extra parameters amplify overfitting

te_d4 adds 384 params (15% more), te_d8 adds 768 params (30% more).
These extra parameters all get used to fit noise on noisy data,
leading to 3.3-4.5× regression on `random_irr`.

### 4.4 CfC's recurrence IS the time encoding

In a transformer, positional encoding is needed because attention
is permutation-equivariant (no inherent order). In CfC, the
recurrence gives a STRONG order signal: `h_t` is a function of
`[x_0, x_1, ..., x_t, h_0]`. There's no need for explicit time
embedding because the recurrence already encodes time.

## 5. NEW INSIGHTS

1. **CfC's recurrence is its own positional encoding**. The
   recurrent state `h_t` is a function of all previous inputs,
   which means it implicitly encodes "how much time has elapsed".
2. **Sinusoidal time embedding is redundant for RNNs**. It only
   helps in architectures without inherent order (Transformers).
3. **Pattern reinforced**: input-side processing only helps when it
   adds something the recurrent step doesn't provide. Time
   embedding adds nothing new.
4. **Extra parameters on noisy data = overfitting**. te_d4 (15% more
   params) is already enough to fit noise and lose 3.3× on
   random_irr.
5. **Different from QuITE/GIS winners**: QuITE handles MISSING
   DATA (the f-gate can't do that), GIS adds an additive SHORTCUT
   (the f-gate doesn't have that). Time embedding adds nothing the
   f-gate doesn't already implicitly do.

## 6. The 91-138 audit: 15 mechanism classes

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE (12 winners) |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Gated Input Skip (additive shortcut) | 134 | STRICTLY POSITIVE (13th winner) |
| Layer Normalization (per-sample normalize) | 135 | TARGET-DEPENDENT (smooth only) |
| 1D Convolutional Input Preprocessing | 137 | TARGET-DEPENDENT (smooth wins, noisy catastrophic) |
| **Sinusoidal Time Embedding** | **138** | **NEGATIVE (10th negative)** |
| Zoneout (preserve h) | 136 | NEGATIVE (9th negative) |
| Recursion depth (MoR) | 126 | NEGATIVE |
| 2nd-order oscillator | 128 | NEGATIVE |
| Multi-timescale ELM | 129 | NEGATIVE |
| Multi-rate MoE + dual attn | 130 | NEGATIVE |
| HGRN (gated linear RNN) | 131 | NEGATIVE |
| AntisymmetricRNN | 132 | NEGATIVE |
| Hebbian Fast Weights | 133 | NEGATIVE |

**Pattern reinforced (13 winners + 2 target-dependent + 10 negatives)**:
- All 13 winners preserve the recurrent step + add useful structure
  that HELPS (input-side: GIS, QuITE; expert-side: MoE; additive skip).
- 2 target-dependent (LN, 1D Conv) add a transformation that helps
  on smooth data and hurts on noisy data.
- 10 negatives propose alternatives to the recurrent step OR add
  unsupervised/regularizer terms OR add REDUNDANT information (this
  round).

## 7. Recommendation

**Sinusoidal Time Embedding CfC is the 10th NEGATIVE in the 91-138 audit.**

- **DO NOT use time embedding for CfC** — the recurrent state is
  its own positional encoding.
- **Time embedding helps Transformers** (no inherent order) but
  not RNNs (recurrence gives order).
- **Stick with cfc baseline, GIS-CfC, or LN-CfC** for production.

## 8. Critical implementation details

1. **Sinusoidal embedding is parameter-free** — `sin` and `cos`
   with frequencies `exp(-log(max_period) * i / half)`.
2. **Time is normalized to [0, 1]** — `t_norm = t / (T - 1)`.
3. **At t=0, embedding is `[0, ..., 0, 1, ..., 1]`** (sin(0)=0,
   cos(0)=1) — useful sanity check.
4. **Concat to input** (not add) — keeps the embedding independent
   of input scale.
5. **Per-layer time embedding** — each layer has its own time
   signal.
6. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements — torch imports work fine at runtime
   via `.venv312/bin/python`.
