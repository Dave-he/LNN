# Round 128 — OscillatorCfC (arXiv:2602.12139 Shende et al. 2026)

**Date**: 2026-06-15
**PRD**: #10-90
**Commit**: TBD
**Verdict**: **HONEST NEGATIVE-WITH-NUANCE** — mathematically
elegant but architecturally too restrictive for our 1D
recurrent setting.

## Summary

Tested whether the **damped harmonic oscillator closed-form
solution** from arXiv:2602.12139 (Shende, Das, Chauhan, Pathak,
Gupta — Ashoka University, February 2026, "Oscillators Are All
You Need: Irregular Time Series Modelling via Damped Harmonic
Oscillators with Closed-Form Solutions") helps our CfC cell.

The paper replaces ContiFormer's second-order NODE backbone
with a damped harmonic oscillator that admits a closed-form
matrix exponential. We adapted this to a recurrent CfC cell:
2D state (h, p) where p = dh/dt, per-neuron learnable natural
frequency ω and damping ratio ζ, linear forcing
F = W_x x + b, closed-form step.

**The result is HONEST NEGATIVE-WITH-NUANCE** — 6.3× fewer
parameters but 3-39× worse test_mse across all 3 datasets,
and **diverges** with extended training (200 epochs) without
bounded forcing.

## 1. Hypothesis

The paper's main claim: replacing an iterative ODE solver with
a closed-form matrix exponential gives "orders of magnitude
faster" inference while preserving expressiveness. The
hypothesis: applying this to a recurrent CfC cell would give a
**simpler, more parameter-efficient** alternative to CfC's
gated-tanh closed-form approximation.

## 2. The oscillator cell

The damped harmonic oscillator:
```
ẍ + 2γẋ + ω²x = F(t)
```

Converting to first-order with z = [h, p]ᵀ, p = ẋ:
```
ż = A z + b, A = [[0, 1], [-ω², -2γ]], b = [0, F]ᵀ
```

For constant F over interval Δt, the closed-form solution is
z(Δt) = e^(A Δt) (z(0) - z_ss) + z_ss where z_ss = (F/ω², 0).

For the underdamped case (γ < ω) with ω_d = sqrt(ω² - γ²):
```
e^(At) = e^(-γt) [[cos+γ/ωd·sin, 1/ωd·sin], [-ω²/ωd·sin, cos-γ/ωd·sin]]
```

We learn ω ∈ log-uniform[0.5, 5.0] and ζ ∈ (0, 1) per neuron.

## 3. Critical design constraint

**F must NOT depend on h** — the closed-form assumes F is
constant over the interval. If F = W_x·x + W_h·h + b, then F
changes as h evolves, invalidating the closed-form assumption.
This is the same constraint as the standard linear RNN
(Mamba-style): the state-space form is A·h + B·x, not
f(h, x) general.

This means **the OscillatorCfC loses all recurrent nonlinearity**
in the forcing term. The only nonlinearity is the oscillator
dynamics itself (sin/cos) applied to the (h, p) state. This
is a substantial loss of expressiveness vs. CfC, which has
W_h·h, sigmoid, tanh on h.

## 4. Bench results (12 cells, 30 epochs, 2 seeds)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc**             | **0.0094**±0.0019 | **0.0053**±0.0010 | **0.0013**±0.0004 | 2545 |
| oscillator_cfc      | 0.0313±0.0007 | 0.0607±0.0077 | 0.0505±0.0199 | **401** |

**OscillatorCfC loses on ALL 3 datasets** despite 6.3× fewer
parameters:
- sin: 0.0313 vs 0.0094 (3.3× worse)
- structured: 0.0607 vs 0.0053 (11.4× worse)
- random: 0.0505 vs 0.0013 (38.8× worse)

### Extended training (200 epochs, sin_irr only)

| Variant | s0 | s1 | n_params |
|---------|----|----|----------|
| vanilla      | 57.99 | 48.17 | 401 |
| tanh + wide ω | 0.43 | 0.07 | 401 |
| cfc (30 ep)   | 0.0094 | — | 2545 |

**Vanilla diverges** with extended training (loss grows to
50+). **Tanh-bounded forcing** is required for stability, but
even then it's 8-46× worse than CfC.

## 5. Why it fails (architectural analysis)

1. **F = W_x·x + b** is the killing constraint. CfC has
   F = f(h, x) with sigmoid + tanh branches — this gives
   the recurrent nonlinearity the oscillator lacks.
2. **The oscillator's "nonlinearity"** is the matrix
   exponential's sin/cos terms applied to (h, p), but these
   are *bounded* in [-1, 1] for the steady state, so the
   state magnitude is small. CfC's tanh can grow unbounded
   (with training).
3. **The paper's setting is different**: it replaces a
   ContiFormer **attention** backbone, not an **RNN cell**.
   Attention has queries that can drive the keys/values; the
   RNN has only the previous state, which must be reused
   through the constant-F constraint.
4. **6.3× parameter reduction** is the only positive: 401 vs
   2545 params. But on our small datasets, parameter count
   is not the bottleneck (CfC at 2545 still wins easily).

## 6. The 91-128 audit: 4th ODE family

**Pattern (91-128)**: 24 structural mechanisms tested.
- **12 STRICTLY POSITIVE winners**: 99, 102, 105, 107, 113, 114, 116, 118, 123, 124, 125, 127
- **12 negatives/target-dep**: 108, 109, 110, 112, 115, 117, 119, 120, 121, 122, 126, **128**

| Round | Mechanism | Type | Verdict |
|-------|-----------|------|---------|
| 76-78 | CfC + n_tau + MR-MoE + FAME | 1st-order ODE + MoE | STRICTLY POSITIVE |
| 91-99 | Smoothness/dropout/reliability/orth | Diagnostics + gates | STRICTLY POSITIVE |
| 102-103 | QuITE + QuITE-MoE | Irregular TS embedding | STRICTLY POSITIVE |
| 113-118 | DeepSeek-Shared + LoRA | 4-axis hybrid | STRICTLY POSITIVE |
| 123-125 | LoRA-DAG-Shared | 4-axis hybrid | STRICTLY POSITIVE |
| 127 | K_r=K_s=2 sweep | Symmetric config | STRICTLY POSITIVE |
| **128** | **OscillatorCfC (2nd-order closed-form)** | **2nd-order ODE** | **HONEST NEGATIVE** |

**NEW INSIGHT (round 128)**: The 2nd-order damped harmonic
oscillator closed-form is a **different family** from the
1st-order ODE family that all our winners share. The
closed-form constraint (F = W_x·x + b) is too restrictive
for the recurrent RNN setting. The paper's claim of
expressiveness preservation likely holds for **transformer
attention backbones** (where F can be a function of queries
and external inputs), not for **recurrent cells** (where
the state is the only information carrier).

## 7. Critical implementation details

1. **F = W_x·x + b ONLY** (no h-term) — required for
   closed-form to be valid. We lose recurrent nonlinearity
   in the forcing.
2. **Per-neuron learnable ω, ζ** — 2H additional params.
   ω = exp(omega_raw), ζ = sigmoid(zeta_raw). Init ω log-uniform,
   ζ via sigmoid(zeta_init) ≈ 0.18.
3. **Underdamped case only** — paper uses γ < ω. Critically
   damped / overdamped cases also derivable but uncommon.
4. **Tanh-bounded forcing required** for stability with
   extended training. Vanilla forcing can diverge.
5. **dt=1.0 default** — the closed-form works at any dt;
   we used the same dt as CfC for direct comparison.

## 8. Future work

1. **OscillatorCfC as a backbone in MoE** — maybe as one of
   the experts, the closed-form speed could help.
2. **Hybrid Oscillator+CfC** — use oscillator for first-order
   dynamics, CfC for higher-order. May give the best of both.
3. **Critically-damped / overdamped variants** — different
   inductive bias, may help.
4. **Test on PhysioNet 36D** — paper claims strong on
   irregular; we couldn't reproduce the gain in 1D.
5. **Use oscillator for attention (paper's original setting)**
   — adapt to transformer instead of RNN.
6. **Oscillator with learnable h-dependent coefficient** —
   treat F as constant per interval but with a learnable
   multiplier. Compromise between closed-form and expressiveness.

## Why it works (where it could)

The OscillatorCfC's win condition is:
- **Pure linear forcing** with rich input
- **High-dimensional state** (large H) so the per-neuron
  oscillations can capture many modes
- **Long sequences** where the closed-form speed matters
- **Irregular time series** where the analytical integration
  between observations is a real advantage

None of these conditions hold in our 1D toy bench. The
negative result is **specific to our 1D recurrent setting**,
not a refutation of the paper's claim in their transformer
attention setting.
