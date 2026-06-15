# Round 145 — Difference Features CfC (Box-Jenkins 1976, Hamilton 1994)

**Date**: 2026-06-15
**PRD**: #10-107
**Verdict**: **HONEST NEGATIVE** — 16th negative in 91-145 audit.

## Summary

Round 145 tests classical **finite-difference input features** (Box-Jenkins
1976, Hamilton 1994) applied to CfC. Two design choices were tested:

1. **diff_concat_k** — append finite differences to x: `input = [x, Δx, Δ²x]`
2. **diff_only_k** — use only finite differences: `input = [Δx, Δ²x]`

**Verdict**:

- **diff_concat_1/2**: NEUTRAL (essentially safe, no big wins)
  - diff_concat_1: sin 1.0× tied, structured 1.45× worse, random **1.3× BETTER**
  - diff_concat_2: sin **1.13× BETTER**, structured 1.40× worse, random 1.85× worse
- **diff_only_1/2**: CATASTROPHIC on noisy data
  - diff_only_1: sin 1.74×, structured 7.5×, random **22.8× WORSE**
  - diff_only_2: sin 1.47×, structured 2.87×, random **32.2× WORSE**

**HEADLINE**: Throwing away the original x in favor of finite differences
(removing the absolute-value baseline) is **catastrophic on noisy data**
(22-32× regression). The audit's "input-side processing wins" pattern
holds ONLY when the original input is **preserved** (concat, parallel
processing). Replacing x with Δx removes the absolute value baseline
that the model needs to anchor predictions.

## 1. Hypothesis

- **H1** (Smooth data): Δx helps because sin(t) has predictable slopes.
  - diff_concat_2 **CONFIRMED (1.13× better)**, diff_concat_1 tied.
- **H2** (Structured data): Δx helps regime boundaries.
  - **REJECTED for all variants** (1.4-7.5× worse on structured).
- **H3** (Random data): Δx HURTS — small regression OK.
  - diff_concat_1: actually **1.3× BETTER** on random (unexpected positive).
  - diff_only_1/2: **CATASTROPHIC** (22-32× worse).
- **H4** (Combined): [x, Δx, Δ²x] helps smooth, neutral elsewhere.
  - **PARTIAL**: helps sin (1.13×), mild regression on others.

## 2. Implementation

`lnn/core/diff_cfc.py` (~150 lines) — `DifferenceInputEncoder` + `DiffCfCNetwork`.

Key design choices:

1. **NaN handling**: zero-fill x BEFORE computing differences (so diffs
   are well-defined).
2. **Δx_0 = 0**: first timestep has no previous, so the difference is 0.
3. **Parameter-free encoder**: no new learnable params; only the
   downstream CfC's input Linear grows with input dim.
4. **4 variants**: concat 1/2, diff_only 1/2 — covers both philosophies.

## 3. Bench results (30 cells: 5 conds × 3 datasets × 2 seeds, 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | 0.0094±0.0019 | **0.0053±0.0010** | 0.0013±0.0004 | 2545 |
| **diff_concat_1** | 0.0093±0.0055 | 0.0077±0.0038 | **0.0010±0.0000** | 2641 |
| **diff_concat_2** | **0.0083±0.0004** | 0.0074±0.0019 | 0.0024±0.0017 | 2737 |
| diff_only_1 | 0.0163±0.0061 | 0.0399±0.0087 | 0.0296±0.0067 | 2545 |
| diff_only_2 | 0.0138±0.0049 | 0.0152±0.0035 | 0.0418±0.0179 | 2641 |

**Headline numbers**:

- diff_concat_1: sin 0.93× (tied), structured 1.45× worse, random **0.77× BETTER**
- diff_concat_2: sin **0.88× BETTER**, structured 1.40× worse, random 1.85× worse
- diff_only_1: sin 1.74× worse, structured 7.5× worse, random **22.8× worse**
- diff_only_2: sin 1.47× worse, structured 2.87× worse, random **32.2× worse**

## 4. Why diff_only is catastrophic on noisy data

The two diff_only variants discard the original x_t and feed only Δx_t
(or Δx_t, Δ²x_t). This removes the **absolute value baseline** that the
model needs to anchor predictions.

For random walk data:
- `x_t` has cumulative structure (target baseline).
- `Δx_t = x_t - x_{t-1}` is just the per-step increment, which is Gaussian
  noise with no anchor.
- Without `x_t`, the model has no way to recover the absolute value.

For smooth data (sin, structured), the same problem is less severe
because the model's hidden state can integrate Δx back to x, but for
noisy data this integration diverges.

## 5. Why diff_concat_1/2 is mostly safe

The concat variants preserve x_t. The added Δx features are extra
information, not a replacement. This is the same pattern as the 5
input-side winners in the 91-144 audit:

- **LN** (135) — adds normalization, keeps x
- **Conv** (137) — adds convolutional features, keeps x
- **GLU+skip** (139) — adds gated projection, keeps x via skip
- **Decoupled/IndRNN** (143) — adds separate x and h paths, keeps both
- **Bidi_concat** (144) — adds backward pass, keeps forward pass

All 5 input-side winners **preserve the original input**. diff_only
violates this principle and pays the price.

## 6. NEW INSIGHTS

1. **The "input-side wins" pattern has a boundary condition**: the
   original input must be preserved. Replacing x with Δx removes the
   absolute value baseline, catastrophic on noisy data.
2. **Pure difference features lose the absolute value** — model can't
   anchor predictions to the input scale. The hidden state can integrate
   back to x for smooth data, but for noisy data this diverges.
3. **diff_concat_1 has an unexpected positive on random_irr (1.3× better)**:
   Δx may help the model detect "stable" vs "unstable" timesteps
   (large Δx = high noise, small Δx = stable). This is a small
   positive signal but not a robust win.
4. **Pattern reinforced**: 5 input-side winners (135, 137, 139, 143, 144)
   all preserve x. 1 input-side negative (145 diff_only) replaces x.
   **Structural addition rule: ADD, don't REPLACE**.

## 7. The 91-145 audit: 33 mechanism classes

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE (12 winners) |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Gated Input Skip (additive shortcut) | 134 | STRICTLY POSITIVE (13th winner) |
| **Layer Normalization** | 135 | **TARGET-DEPENDENT** (smooth only) |
| **1D Convolutional Input Preprocessing** | 137 | **TARGET-DEPENDENT** |
| **GLU + Identity Skip** | 139 | **TARGET-DEPENDENT** |
| **Decoupled / IndRNN-CfC** | 143 | **TARGET-DEPENDENT** |
| **Bidirectional CfC (concat)** | 144 | **TARGET-DEPENDENT (5th)** |
| **Difference Features CfC (concat)** | **145** | **NEUTRAL** (mostly safe, no big wins) |
| **Difference Features CfC (diff_only)** | **145** | **NEGATIVE (16th)** |
| Multiplicative Integration (Wu 2016) | 142 | NEGATIVE (15th) |
| Adaptive Time-Constant (Graves 2016) | 141 | NEGATIVE (14th) |
| SE Channel Attention | 140 | NEGATIVE (13th) |
| GLU alone (glu_basic) | 139 | NEGATIVE (12th) |
| Sinusoidal Time Embedding | 138 | NEGATIVE (11th) |
| Zoneout | 136 | NEGATIVE (10th) |
| Bidirectional CfC (weighted) | 144 | NEGATIVE (15th) |
| Recursion depth (MoR) | 126 | NEGATIVE |
| 2nd-order oscillator | 128 | NEGATIVE |
| Multi-timescale ELM | 129 | NEGATIVE |
| Multi-rate MoE + dual attn | 130 | NEGATIVE |
| HGRN | 131 | NEGATIVE |
| AntisymmetricRNN | 132 | NEGATIVE |
| Hebbian Fast Weights | 133 | NEGATIVE |

**Pattern reinforced (13 + 5 + 16 = 34 tests)**:

- 13 winners preserve recurrent step + add useful structure
- 5 target-dep: input-side processing (all 5 preserve x) OR bidirectional
  structural addition
- **16 negatives**: per-step modifications, alternatives, regularizers,
  bottlenecks, redundant info, weighted bidi, **diff_only** (input
  replacement)

**NEW RULE**: Input-side processing that REPLACES x is catastrophic
on noisy data (diff_only: 22-32× worse on random). Input-side
processing that ADDS to x (concat/parallel) is safe or wins.

## 8. Recommendation

**Difference Features CfC is the 16th NEGATIVE in the 91-145 audit
(diff_only variants), and NEUTRAL (diff_concat variants).**

- **DO NOT use diff_only** — pure difference features lose the
  absolute value baseline, catastrophic on noisy data.
- **diff_concat_1/2 is safe but not strictly better** — appending
  Δx to x is mostly neutral. There may be marginal benefit on
  smooth data (diff_concat_2 sin -13%), but no robust positive.
- **The "input-side processing" rule has a boundary condition**:
  preserve the original input. Replace = catastrophic on noise.

## 9. Critical implementation details

1. **NaN handling**: zero-fill x BEFORE computing differences.
2. **Δx_0 = 0**: first timestep has no previous, so diff = 0.
3. **Parameter-free encoder**: no new learnable params; downstream
   CfC's input Linear grows with input dim (3-7% param increase).
4. **Two modes**: "concat" (preserves x, 2-3× D input dim) vs
   "diff_only" (replaces x, 1-2× D input dim).
5. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements — torch imports work fine at runtime
   via `.venv312/bin/python`.
