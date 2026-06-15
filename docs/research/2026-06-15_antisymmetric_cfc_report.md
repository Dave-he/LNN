# Round 132 — AntisymmetricRNN CfC (AntisymmetricRNN: A Dynamical System View on Recurrent Neural Networks, Chang et al. ICLR 2019)

**Date**: 2026-06-15
**PRD**: #10-94
**Commit**: TBD
**Verdict**: **HONEST NEGATIVE** — 17th negative in 91-132 audit.

## Summary

Tested the **AntisymmetricRNN** mechanism from Chang, Chen, Haber, Chu
(ICLR 2019, arXiv:1811.02243). The idea is to constrain the
hidden-to-hidden weight matrix `M` to be ANTISYMMETRIC (`M = -M^T`),
giving the recurrent dynamics **marginal stability**: all eigenvalues
of M are pure imaginary, so the system neither diverges (real part > 0)
nor collapses (real part < 0), but oscillates in a bounded region.

**Verdict: HONEST NEGATIVE** — both Antisymmetric variants LOSE to CfC
on all 3 datasets, with up to 53× worse test_mse. The constraint
provides theoretical stability but the practical 1D-regression
performance collapses.

## 1. Hypothesis

The mechanism is **structural** (constrains W_h directly) and
**preserves the W·h nonlinearity** (W_h @ h_prev is still in the
candidate). Per the 91-131 audit, mechanisms that ADD a useful
inductive bias to the recurrent step while preserving W·h are
STRICTLY POSITIVE (12 winners). The hypothesis was:

- **H1 (stability constraint helps on noisy data)**: with antisymmetric
  M, test_mse on `random_irr` is < unconstrained CfC.
- **H2 (oscillatory behavior helps regime switching)**: with antisymmetric
  M, test_mse on `structured_irr` (regime switch at T/2) is < baseline.
- **H3 (no regression on smooth data)**: with antisymmetric M, test_mse
  on `sin_irr` is not worse than baseline by >10%.

## 2. Implementation

`AntisymmetricMatrix`, `AntisymmetricCfCCell`, and
`AntisymmetricCfCStackedNetwork` in `lnn/core/antisymmetric_cfc.py`
(~200 lines). 23 unit tests covering: antisymmetry at init (M + M^T = 0,
diagonal = 0), pure-imaginary eigenvalues, n*(n-1)/2 effective params,
gradient flow to upper-triangle storage U, lower-triangle grad = 0,
stability over 100 forward steps, end-to-end smoke training.

Key design choices:

1. **Antisymmetry by construction** — store `U` (full matrix parameter)
   and reconstruct `M = U - U^T` (with diagonal forced to 0). This
   guarantees antisymmetry without explicit constraint projection.
2. **In-place triu_** — used `triu_` (in-place) to initialize the
   upper triangle of U (the non-in-place version returned a copy that
   wasn't bound to the parameter, a subtle bug we caught in unit tests).
3. **Euler step update** — `h_t = h_{t-1} + dt * (tanh(M @ h + W_x x + b) - h_{t-1})`
   with dt=0.1.

## 3. Bench results (18 cells, 30 epochs, 2 seeds)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | **0.0094**±0.0019 | **0.0053**±0.0010 | **0.0013**±0.0004 | 2545 |
| antisym_small (init=0.1) | 0.3638±0.0344 | 0.4036±0.0403 | 0.0648±0.0042 | 881 |
| antisym_large (init=0.5) | 0.1650±0.0702 | 0.3276±0.0366 | 0.0691±0.0056 | 881 |

**Both Antisymm variants LOSE on ALL 3 datasets** with 2.9× fewer params:

- **sin_irr**: 38.7× (small) / 17.6× (large) worse
- **structured_irr**: 76.2× (small) / 61.8× (large) worse
- **random_irr**: 49.8× (small) / 53.2× (large) worse

H1 (helps on random_irr) — **REJECTED** (49-53× worse)
H2 (helps on structured_irr) — **REJECTED** (62-76× worse)
H3 (no regression on sin_irr) — **REJECTED** (18-39× worse)

Within the Antisymm family, **large > small** on sin_irr and
structured_irr (init_scale=0.5 better than 0.1), but still much worse
than CfC.

## 4. Why it fails

### 4.1 Fixed dt=0.1 vs CfC's learned f-gate

AntisymmetricCfC uses a fixed `dt=0.1` Euler interpolation:
`h_t = h + dt * (candidate - h)`. This is much less flexible than
CfC's `f-gate` (sigmoid output that decides per-step how much to use
the candidate vs the previous state). CfC can learn to use f-gate=0.99
to fully overwrite the state, or f-gate=0.01 to keep the state
unchanged. AntisymmetricCfC can only do dt=0.1.

### 4.2 No direct input-to-gate path

CfC's f-gate and g-branch both depend on `[x, h]` (concatenated).
This means the network can decide based on input alone whether to
update the state. AntisymmetricCfC's `candidate = tanh(M h + W_x x + b)`
only uses M@h in the nonlinearity, and the Euler step `h + dt * (cand - h)`
interpolates. There's no input-conditional "use new candidate" decision.

### 4.3 Half the parameters, much less expressivity

881 params vs 2545 (2.9× fewer). The antisymmetric constraint
removes half the parameters AND constrains them to be related
(M[i,j] = -M[j,i]). This is a strong constraint that reduces
expressivity. The paper's claim that "marginal stability" helps
implicitly assumes the system has enough capacity to model the task
under this constraint.

### 4.4 Oscillatory dynamics is wrong for sin/structured targets

The pure-imaginary eigenvalues of M mean the system oscillates.
This is a poor fit for sin_irr (target is a slow oscillation, not
a fast state-space rotation) and structured_irr (target is a step
change, not an oscillation). The rotational dynamics in state-space
doesn't align with the regression target.

## 5. NEW INSIGHTS (round 132)

1. **Marginal stability ≠ good 1D regression fit**. Antisymmetric
   weights are stable by construction, but the rotation tendency
   is mismatched with smooth 1D targets. The stability argument is
   correct mathematically but wrong empirically for these targets.
2. **Fixed dt is a killer**. CfC's learned f-gate is one of its
   key advantages — it can learn per-step interpolation factors.
   AntisymmetricCfC's fixed dt=0.1 is much less flexible.
3. **Half-parameter constraint + Euler step = underfitting**. 881
   params is too few for the expressivity needed to model 3
   datasets, even with the W·h nonlinearity preserved.
4. **The paper's claim is for a different problem class**. The
   paper validates on chaotic systems and longer sequences. 1D
   regression on T=32 sequences is a different regime where
   CfC's learned gating wins.

## 6. The 91-132 audit: 7 neuron-families tested

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Recursion depth (MoR) | 126 | NEGATIVE |
| 2nd-order oscillator | 128 | NEGATIVE |
| Multi-timescale ELM | 129 | NEGATIVE |
| Multi-rate MoE + dual attn | 130 | NEGATIVE |
| HGRN (gated linear RNN) | 131 | NEGATIVE |
| **AntisymmetricRNN (constrained W_h)** | **132** | **NEGATIVE** |

**Pattern reinforced**: papers that propose **alternatives to the
recurrent step's core mechanism** all LOSE in 1D. Common thread:
they remove or replace the **learned per-step f-gate** that makes
CfC work on 1D data. The 6 negatives (rounds 128, 129, 130, 131,
132, plus recursion-depth 126) all share this property.

## 7. Critical implementation details

1. **In-place triu_** — `self.U.triu_(1).uniform_(...)` not
   `self.U.triu(1).uniform_(...)`. The non-in-place version returns
   a copy, so the parameter stays all zeros.
2. **Antisymmetry at all times** — the `M = U - U^T` reconstruction
   preserves antisymmetry through any gradient step (no projection
   needed).
3. **Non-zero h for gradient tests** — gradient to M is 0 when h=0
   (because M @ 0 = 0). Test must use non-zero h to exercise the
   gradient.

## 8. Recommendation

**AntisymmetricRNN is the 17th NEGATIVE in the 91-132 audit.**

- **DO NOT use AntisymmetricCfC for 1D regression** — the stability
  constraint is mathematically interesting but empirically bad.
- **The mechanism is correct for its intended problem** (chaotic
  systems, long sequences, stability analysis) but doesn't translate
  to 1D regression.
- **Stick with cfc baseline or 4-axis hybrid (LoRA-DAG-Shared)** for
  production.
