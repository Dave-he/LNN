# Round 130 — MR-MoE + Dual Attention CfC (arXiv:2606.12240 Zong, Boker, Eldardiry 2026)

**Date**: 2026-06-15
**PRD**: #10-92
**Commit**: TBD
**Verdict**: **HONEST NEGATIVE-WITH-NUANCE** — 15th negative in 91-130 audit.

## Summary

Tested the **Multi-Rate Mixture of Experts (MR-MoE)** framework
from arXiv:2606.12240 (Zong, Boker, Eldardiry, 10 June 2026, NeurIPS
2026 submission) combined with the **dual attention** module
proposed in the same paper:

- **Feature-level attention**: per-step input gate `α_t ∈ [0,1]^D`
  from a small MLP over `[x_t; h_prev]`. Applied as `x_t' = α_t ⊙ x_t`.
- **Temporal attention**: softmax over a window of past hidden states
  (default window=4) to focus on informative history.
- K=3 CfC experts, each with **distinct τ_init** (0.1, 1.0, 10.0) —
  fast / medium / slow.
- Standard FAME-style top-K softmax router over `[x_t'; h_prev]`.

**Verdict: HONEST NEGATIVE-WITH-NUANCE** — all 3 MR-MoE variants
LOSE to the CfC baseline on all 3 datasets, with 3.8× more parameters.
The dual attention is the worst, suggesting the added complexity
**hurts more than helps** in 1D regression.

## 1. Hypothesis

The paper claims the multi-rate + dual attention design gives the
cell a 3-axis inductive bias (per-expert τ + feature gate + temporal
gate) that should win on **structured_irr** (regime switch) — one
expert can lock onto the slow drift and another on the fast
transient.

## 2. Implementation

`MRMoEDualAttnCfCCell` and `MRMoEDualAttnCfCNetwork` in
`lnn/core/mr_moe_dualattn_cfc.py` (~250 lines). 27 unit tests
covering init, forward, router sparsity, feature attn sigmoid
bound, temporal window, gradient flow, NaN handling.

## 3. Bench results (24 cells, 30 epochs, 2 seeds)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | **0.0094**±0.0019 | **0.0053**±0.0010 | **0.0013**±0.0004 | 2545 |
| mr_moe_k3_uniform (all τ=1.0, no attn) | 0.0129±0.0001 | 0.0137±0.0006 | 0.0022±0.0008 | 9717 |
| mr_moe_k3_multirate (τ={0.1,1,10}, no attn) | 0.0130±0.0044 | 0.0133±0.0041 | 0.0051±0.0033 | 9717 |
| mr_moe_k3_dualattn (multi-rate + dual attn) | 0.0225±0.0027 | 0.0239±0.0106 | 0.0242±0.0160 | 9717 |

**All 3 MR-MoE variants LOSE on ALL 3 datasets** with 3.8× more params:

- **sin_irr**: 1.4× (uniform) / 1.4× (multirate) / 2.4× (dualattn) worse
- **structured_irr**: 2.6× / 2.5× / 4.5× worse
- **random_irr**: 1.7× / 3.9× / **18.6×** worse (dualattn worst)

## 4. Why it fails

### 4.1 The multi-rate experts (τ=0.1, 1.0, 10.0) don't help in 1D

With only 2 input features (D=2), the regime-switch structure of
structured_irr is encoded in a single time-varying signal. The
multi-rate experts each get the same `x_t` and `h_prev`, so the
**only** way to specialize is through the router. But with top-K=2
sparse routing, the router tends to send similar inputs to similar
experts — defeating the purpose of having different τs.

The `time_scale` parameter (per-neuron `[H]`) is the **only** way
for the multi-rate inductive bias to actually manifest. In our
1D setting, the slow expert (τ=10) essentially behaves like a
nearly-constant bias that adds no information.

### 4.2 The dual attention actively hurts

The dual attention is the **worst** variant. The feature-level
attention gate `α ∈ [0,1]^D` acts as a per-step input mask. In a
1D setting with D=2, the gate either zeros out the signal entirely
or passes it through — there's no middle ground that helps. The
temporal attention, with window=4 and softmax over only 4 past
states, produces a smooth but uninformative blend.

The 0.1× context_bias on x_t_gated is a small enough scale that
it shouldn't break the network, but it adds gradient noise through
the softmax + projection path.

### 4.3 The paper's setting is different

MR-MoE was evaluated on **multivariate sepsis-like time series**
(D=36+) with **rich temporal structure** (lab values, vitals).
The dual attention can meaningfully filter out noisy features in
D=36, but in D=2 there's nothing to filter. The multi-rate
experts can specialize in high-dimensional feature spaces, but
in 1D there's no room for specialization.

## 5. The 91-130 audit: 5 neuron-families + 1 attention-family tested

**Pattern (91-130)**: 26 structural mechanisms tested.
- **12 STRICTLY POSITIVE winners**: 99, 102, 105, 107, 113, 114, 116, 118, 123, 124, 125, 127
- **14 negatives/target-dep**: 108, 109, 110, 112, 115, 117, 119, 120, 121, 122, 126, 128, 129, **130**

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Recursion depth (MoR) | 126 | NEGATIVE |
| 2nd-order oscillator | 128 | NEGATIVE |
| Multi-timescale ELM | 129 | NEGATIVE |
| **Multi-rate MoE + dual attn** | **130** | **NEGATIVE** |

**NEW INSIGHT (round 130)**: The MR-MoE paper's multi-rate + dual
attention design is **specialized for high-dimensional
multivariate time series** (D ≥ 36). In our 1D regression
setting (D=2), the multi-rate experts have nothing to specialize
on, and the dual attention gates have nothing to filter. The
paper's claim of "consistent AUROC/AUPRC improvements" requires
the rich D of clinical time series — **not transferable to
1D regression**.

This is the same pattern as **round 129 (ELM)**: a paper that
works on biological / spike-based data doesn't translate to
1D continuous regression. **Architectures designed for high-D
inputs suffer in low-D settings**.

## 6. Critical implementation details

1. **`fill_` in `no_grad` context** — needed to override
   CfCCell's default `time_scale = ones(hidden_size)` initialization
   with the per-expert τ_init. With `requires_grad=True` after fill_,
   gradient flows correctly.
2. **Softmax of 1 element has zero grad** — temporal window of size 1
   produces uniform [1.0] attention with no learnable signal. The
   test uses window=2 to avoid this.
3. **Detached temporal window** — `h_new.detach()` is stored in the
   window to keep the autograd graph acyclic. The temporal context
   biases only the input, not the recurrent state — preserves the
   W_h·h nonlinearity property.
4. **Top-K sparse routing** — same FAME-style router as round 78.
   K=3 experts, top_k=2.

## 7. Future work

1. **Test MR-MoE on PhysioNet 36D** — paper's setting, may match
   better. But this is a different model class entirely.
2. **K=1 (no MoE, just dual attention)** — would isolate the
   attention contribution from the multi-rate experts. Likely also
   negative in 1D.
3. **Larger hidden_size** — current H=16 may be too small to give
   the multi-rate experts room to differentiate. H=64 might help.
4. **K=5 with top_k=2** — more experts to choose from, but no
   reason to expect different result.

## 8. Recommendation

**MR-MoE + Dual Attention is the 14th NEGATIVE in the 91-130 audit**.

- **DO NOT use MR-MoE for 1D regression** — multi-rate + dual
  attention adds 3.8× params for 1.4-19× worse test_mse.
- **For multivariate PhysioNet-style data** — MR-MoE may be
  worth re-testing, but it's a different model class.
- **Stick with cfc baseline** or the 4-axis hybrid
  (LoRA-DAG-Shared K_r=K_s=2) for production.
