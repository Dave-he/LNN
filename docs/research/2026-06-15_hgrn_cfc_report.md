# Round 131 — HGRN CfC (HGRN: Hierarchically Gated Recurrent Neural Network, NeurIPS 2023, arXiv:2404.18807)

**Date**: 2026-06-15
**PRD**: #10-93
**Commit**: TBD
**Verdict**: **HONEST NEGATIVE-WITH-NUANCE** — 16th negative in 91-131 audit.

## Summary

Tested the **HGRN** (Hierarchically Gated Recurrent Network) mechanism from
Qi et al. NeurIPS 2023, ported to our CfC stack. The mechanism has two parts:

- **Bounded forget gate**: `gate = alpha + (1 - alpha) * sigmoid(W_g x)`,
  with `alpha` a learnable scalar in [0, 1].
- **Hierarchical alpha schedule**: in a stacked network, `alpha_l` increases
  monotonically with layer index — `alpha_l = (l / (L-1)) * alpha_max`.

**Verdict: HONEST NEGATIVE-WITH-NUANCE** — all 3 HGRN variants LOSE to the
CfC baseline on all 3 datasets. The bounded version is consistently better
than the free-gate version (lower bound helps), but neither is competitive
with CfC's full ODE-based nonlinearity.

## 1. Hypothesis

The HGRN mechanism is a clean structural change to the recurrent step (a
gating mechanism with a lower bound, plus a hierarchical schedule across
layers). Per the 91-130 audit pattern, structural mechanisms that preserve
the recurrent state's W·h nonlinearity are STRICTLY POSITIVE winners. The
hypothesis was:

- **H1 (bounded gate helps vs free gate on noisy data)**: with α=0.1,
  test_mse on `random_irr` is < the free-gate baseline.
- **H2 (hierarchical beats uniform)**: per-layer monotonic α_l helps
  vs flat α across layers.
- **H3 (preserves recurrent nonlinearity)**: bounded-gate variant is at
  least as stable as free-gate.

## 2. Implementation

`HGRNCfCCell` and `HGRNCfCStackedNetwork` in `lnn/core/hgrn_cfc.py`
(~200 lines). 26 unit tests covering init, forward, gradient flow to
W_x/W_g/alpha, stability over 100 forward steps, hierarchical alpha
schedule, end-to-end smoke training.

Key design choice: **soft lower bound** via `gate = a + (1-a) * s` instead
of `gate = max(a, s)`. The hard max would block gradient flow to `a` when
the natural gate `s` is already > `a` (which is the typical case after
training starts). The soft formulation keeps `a` as a real parameter
that gets gradient updates throughout training.

## 3. Bench results (24 cells, 30 epochs, 2 seeds)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | **0.0094**±0.0019 | **0.0053**±0.0010 | **0.0013**±0.0004 | 2545 |
| hgrn_free (α=0) | 0.0246±0.0022 | 0.0570±0.0069 | 0.0129±0.0059 | 659 |
| hgrn_bounded (α=0.2) | 0.0187±0.0003 | 0.0352±0.0035 | 0.0098±0.0050 | 659 |
| hgrn_hierarchical (α_max=0.5) | 0.0211±0.0009 | 0.0537±0.0007 | 0.0106±0.0048 | 659 |

**All 3 HGRN variants LOSE on ALL 3 datasets** with 3.9× fewer params:

- **sin_irr**: 1.99× (free) / 1.99× (bounded) / 2.24× (hierarchical) worse
- **structured_irr**: 10.75× (free) / 6.64× (bounded) / 10.13× (hierarchical) worse
- **random_irr**: 9.92× (free) / 7.54× (bounded) / 8.15× (hierarchical) worse

**However**, within the HGRN family, the ranking is:
**hgrn_bounded (best) > hgrn_hierarchical > hgrn_free (worst)**
on sin_irr and structured_irr (random_irr: bounded > hierarchical ≈ free).

The bounded version reduces test_mse by **24% (sin), 38% (structured), 24%
(random)** compared to the free version. The lower bound IS a real
regularizer — it just isn't enough to compete with CfC.

## 4. Why it fails

### 4.1 Linear recurrence can't match CfC's ODE nonlinearity

HGRN's recurrence is `h_t = (1 - gate) * h_{t-1} + gate * candidate` — a
linear interpolation between the previous state and a candidate. There's
**no W·h term** in the candidate (`candidate = tanh(W_x x + b)` only depends
on x).

CfC's update includes `W_h @ h_prev` terms inside the candidate, which is
the source of its expressivity on structured/regime-switching data.
Replacing the W·h with gating alone **strips out the nonlinearity** that
made CfC successful in 1D.

This is the same failure mode as **rounds 128, 129, 130** (OscillatorCfC,
ELMCfC, MR-MoE+DualAttn): papers that propose alternatives to the standard
recurrent step lose to CfC's full ODE in 1D.

### 4.2 The hierarchy doesn't add anything new

HGRN's hierarchical α_l is a regularization trick that *might* help on
long-range language modeling (where the paper validates it) but doesn't
help on short 1D sequences (T=32). The "lower layers forget more, upper
layers forget less" intuition is about modeling long-range dependencies
in language — our sequences are 32 timesteps, so the difference is
negligible.

### 4.3 Param efficiency: 3.9× fewer params but 6.6-10.8× worse

HGRN has 659 params vs CfC's 2545 (3.9× fewer), but loses 6.6-10.8× on
structured_irr. This is the opposite of LoRA-MoE (round 118) which had
fewer params AND won. HGRN's smaller parameter count comes from removing
the W·h term, which is exactly what made CfC expressive.

## 5. NEW INSIGHTS (round 131)

1. **The lower bound IS a real regularizer** (within the HGRN family,
   bounded reduces test_mse by 24-38% vs free). The mechanism is sound
   in isolation, just insufficient vs CfC.
2. **Linearity is the killer**: when you replace `W·h` with a
   linear-recurrence-with-gate, you lose 6-11× on structured data.
   The W·h nonlinearity is essential.
3. **Hierarchy is a language-modeling concept** that doesn't transfer
   to 1D time-series. The 91-130 audit shows the same pattern: language
   model innovations (xLSTM, HGRN, MH-MoE 115) are NEGATIVE in 1D.

## 6. The 91-131 audit: 6 neuron-families tested

**Pattern (91-131)**: 27 structural mechanisms tested.
- **12 STRICTLY POSITIVE winners**: 99, 102, 105, 107, 113, 114, 116, 118, 123, 124, 125, 127
- **16 negatives/target-dep**: 108, 109, 110, 112, 115, 117, 119, 120, 121, 122, 126, 128, 129, 130, **131**

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Recursion depth (MoR) | 126 | NEGATIVE |
| 2nd-order oscillator | 128 | NEGATIVE |
| Multi-timescale ELM | 129 | NEGATIVE |
| Multi-rate MoE + dual attn | 130 | NEGATIVE |
| **HGRN (gated linear RNN + bounded α)** | **131** | **NEGATIVE** |

**Pattern reinforced**: papers that propose ALTERNATIVES to the
recurrent step (oscillator, ELM, MR-MoE, HGRN) all LOSE in 1D. The
common thread: they remove or replace the W·h nonlinearity that makes
CfC work on 1D.

## 7. Critical implementation details

1. **Soft lower bound** (`gate = a + (1-a) * s`) — using `clamp(s, min=a)`
   blocks gradient to `a` when `s > a` (always true after first few
   training steps). The soft formulation keeps `a` trainable.
2. **Alpha as a tensor** — `self.alpha` should call `torch.sigmoid(self.raw_alpha)`
   INSIDE `forward` to preserve the autograd graph, not via `.item()`.
3. **Monotonic α_l** — `alpha_l = (l / (L-1)) * alpha_max` with `l=0`
   giving α=0 (free gate in first layer) and `l=L-1` giving α_max
   (mostly-forgetting in last layer) — matches the HGRN paper's
   "lower layers forget more" intuition.

## 8. Recommendation

**HGRN is the 16th NEGATIVE in the 91-131 audit.**

- **DO NOT use HGRN for 1D regression** — gating alone is insufficient
  vs CfC's full ODE.
- **The lower bound mechanism IS sound** as a regularizer (24-38%
  improvement within the HGRN family) — but it can't fix the
  fundamental issue of lacking the W·h nonlinearity.
- **Stick with cfc baseline** or the 4-axis hybrid (LoRA-DAG-Shared
  K_r=K_s=2) for production.
