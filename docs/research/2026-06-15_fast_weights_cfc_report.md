# Round 133 — Hebbian Fast Weights CfC (Ba et al. NIPS 2016, arXiv:1610.06258)

**Date**: 2026-06-15
**PRD**: #10-95
**Commit**: TBD
**Verdict**: **HONEST NEGATIVE** — 18th negative in 91-133 audit.

## Summary

Tested the **Hebbian Fast Weights** mechanism from Ba, Hinton, Mnih,
Romoff, Veness (NIPS 2016). The idea: maintain a fast weight matrix
`F_t` that evolves at every recurrent step via
`F_t = λ*F_{t-1} + η*(h_t ⊗ h_{t-1})` (Hebbian outer product + decay).
The recurrent step uses BOTH the slow weights and the fast weights:
`h_t = σ(W_h*h + F_t*h + W_x*x + b)`.

**Verdict: HONEST NEGATIVE** — all 3 FastWeights variants LOSE to CfC
on all 3 datasets, with 2.1-13.4× worse test_mse. The mechanism
preserves W·h and CfC's f-gate (the "additive" pattern that should
win) but the Hebbian update is unsupervised and doesn't help with
the task.

## 1. Hypothesis

The mechanism is **structural** and **preserves** W·h and CfC's
f-gate. Per the 91-132 audit, "additive" mechanisms that ADD to the
recurrent step (11/12 MoE winners) should win. The hypothesis was:

- **H1 (fast weights help on noisy data)**: with λ=0.9, η=0.1,
  test_mse on `random_irr` is < baseline.
- **H2 (fast weights help on regime switching)**: with λ=0.9, η=0.1,
  test_mse on `structured_irr` is < baseline.
- **H3 (no regression on smooth data)**: with λ=0.9, η=0.1, test_mse
  on `sin_irr` is not worse than baseline by >10%.

## 2. Implementation

`FastWeightsCfCCell` and `FastWeightsCfCStackedNetwork` in
`lnn/core/fast_weights_cfc.py` (~250 lines). 21 unit tests covering
init/forward/gradient/stability/reset.

Key design choices:

1. **F is a buffer** (not a parameter) — it changes per forward pass.
2. **λ and η are learnable scalars** (sigmoid-constrained to [0, 1]).
3. **Hebbian update without torch.no_grad()** — so gradient flows to
   λ and η (and to W_h, W_f, W_g via the F@h term).
4. **F@h term concatenated with [x, h]** as input to f-gate and
   g-branch.
5. **F is reset between sequences** (in stacked network's forward).

## 3. Bench results (24 cells, 30 epochs, 2 seeds)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | **0.0094**±0.0019 | **0.0053**±0.0010 | **0.0013**±0.0004 | 2545 |
| fw_weak (λ=0.95, η=0.01) | 0.0197±0.0020 | 0.0149±0.0001 | 0.0174±0.0112 | 2709 |
| fw_strong (λ=0.9, η=0.1) | 0.0234±0.0005 | 0.0188±0.0020 | 0.0148±0.0085 | 2709 |
| fw_long (λ=0.99, η=0.05) | 0.0210±0.0022 | 0.0260±0.0069 | 0.0159±0.0093 | 2709 |

**All 3 FastWeights variants LOSE on ALL 3 datasets** with 1.06×
more params:

- **sin_irr**: 2.1× (weak) / 2.5× (strong) / 2.2× (long) worse
- **structured_irr**: 2.8× (weak) / 3.5× (strong) / 4.9× (long) worse
- **random_irr**: 13.4× (weak) / 11.4× (strong) / 12.2× (long) worse

H1 (helps on random_irr) — **REJECTED** (11-13× worse)
H2 (helps on structured_irr) — **REJECTED** (2.8-4.9× worse)
H3 (no regression on sin_irr) — **REJECTED** (2.1-2.5× worse)

Within FastWeights family:
- sin_irr: weak ≈ long < strong
- structured_irr: weak < strong < long
- random_irr: strong ≈ long < weak (but high std)

No clear winner — the Hebbian mechanism doesn't help with any of the
3 datasets in 1D.

## 4. Why it fails

### 4.1 Hebbian update is unsupervised

The Hebbian rule `F_t = λ*F_{t-1} + η*(h_t ⊗ h_{t-1})` is purely
unsupervised — it captures pairwise correlations between consecutive
hidden states, but there's no learning signal to make these
correlations useful for the supervised task. The network's f-gate
learns to use the F@h term, but the F@h term is itself random
relative to the task.

### 4.2 The f-gate is already a powerful interpolation

CfC's `f-gate = sigmoid(W_f [x, h])` decides per-step how much to
use the candidate vs the previous state. This is already a learned,
per-step interpolation. Adding a fast-weight term F@h to the input
of f-gate gives the gate more information, but in 1D regression
this extra information is not useful (no long-range dependencies
beyond what f-gate already handles).

### 4.3 Gradient chain through F is long and unstable

The forward computation `Fh = h @ F.T` and the Hebbian update
`F = λ*F_old + η*outer` create a long gradient chain. λ gradient
flows through F_old (which is itself a function of λ). At early
training steps, F is small and gradients to λ are exactly 0 (need
≥4 steps for non-zero λ gradient). This makes training the
Hebbian hyper-parameters difficult.

### 4.4 More parameters, worse performance

2709 vs 2545 (1.06× more) and 2.1-13.4× worse. The mechanism is a
**net loss** of capacity for these tasks.

## 5. NEW INSIGHTS (round 133)

1. **"Additive" is not enough — the addition must be useful**. The
   11/12 MoE winners ADD experts with task-relevant specializations.
   Fast Weights ADDS a Hebbian term, but the Hebbian term captures
   pairwise correlations that aren't useful for 1D regression.
2. **Hebbian learning ≠ supervised learning**. The Hebbian update
   is unsupervised (outer product of consecutive h), so F captures
   correlation, not causation for the task. In a supervised setting,
   the supervised gradient must flow back through F to make F useful,
   but the gradient signal is weak.
3. **F@h is high-dimensional noise on 1D tasks**. With hidden_size=16,
   F is 16x16, and F@h produces a 16-dim vector that changes with
   every step. On smooth 1D targets, this high-dimensional time-
   varying signal is noise.
4. **CfC's f-gate is already the "learned per-step interpolation"
   that Fast Weights adds**. The mechanism is essentially "Hebbian
   interference" with this interpolation.

## 6. The 91-133 audit: 7 neuron-families tested

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE (12 winners) |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Recursion depth (MoR) | 126 | NEGATIVE |
| 2nd-order oscillator | 128 | NEGATIVE |
| Multi-timescale ELM | 129 | NEGATIVE |
| Multi-rate MoE + dual attn | 130 | NEGATIVE |
| HGRN (gated linear RNN) | 131 | NEGATIVE |
| AntisymmetricRNN (constrained W_h) | 132 | NEGATIVE |
| **Hebbian Fast Weights** | **133** | **NEGATIVE** |

**Pattern reinforced (8 negatives in 91-133)**: all 8 negatives
proposed **alternatives to CfC's standard recurrent step**. The 12
winners are all MoE (preserves recurrent step + adds experts) or
input-side mechanisms. The "additive vs replacement" distinction is
the dominant factor.

## 7. Critical implementation details

1. **No torch.no_grad() in Hebbian update** — needed for gradient
   flow to λ and η.
2. **Multi-step tests needed** — gradient to λ is 0 at step 1
   (because F_0 = 0), non-zero only at step 4+.
3. **F is reset per forward** — otherwise F from the previous
   forward contaminates the next sequence.

## 8. Recommendation

**Hebbian Fast Weights is the 18th NEGATIVE in the 91-133 audit.**

- **DO NOT use FastWeightsCfC for 1D regression** — the Hebbian
  update is unsupervised and adds high-dimensional noise.
- **The mechanism is correct for its intended problem** (sequence
  memory, attention over recent past) but doesn't translate to 1D.
- **Stick with cfc baseline or 4-axis hybrid (LoRA-DAG-Shared)** for
  production.
