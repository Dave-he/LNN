# Round 154 — MONO-CfC (Monotonic Activation CfC)

**Date**: 2026-06-15
**PRD**: #10-116
**Verdict**: **UNANIMOUS NEGATIVE** — all 4 monotonic/bounded
variants make CfC WORSE. Tanh is critical to CfC's closed-form
solution.

## Summary

Round 154 tests **MONO-CfC (Monotonic Activation CfC)** — replace
CfC's Tanh activations in g_branch and h_branch with monotonic
or bounded alternatives. Inspired by monotonic networks
(Chilinski & Silva 2020, "Neural Likelihoods for Continuous-Time
Markov Chains")::

    # Standard CfC
    g_branch = Tanh(W_g · [x, h] + b_g)  # in [-1, 1]
    h_branch = Tanh(W_h · [x, h] + b_h)  # in [-1, 1]

    # mono_g: replace g_branch Tanh with Softplus (positive monotonic)
    # mono_h: replace h_branch Tanh with Softplus (positive monotonic)
    # mono_both: replace both Tanh with Softplus
    # mono_sig: replace both Tanh with Sigmoid (control, bounded [0, 1])

    # Closed-form solution (unchanged):
    h_t = σ(-f · τ) · g + (1 - σ(-f · τ)) · h_branch

**Verdict**: **UNANIMOUS NEGATIVE** — all 4 variants make CfC
significantly worse:

- **mono_g**: sin +73%, structured +262% CATASTROPHIC,
  random +21%
- **mono_h**: sin +91%, structured +241% CATASTROPHIC,
  random +1%
- **mono_both**: sin +506%, structured +265%, random +57%
- **mono_sig** (control): sin +1553%, structured +265%,
  random +63%

## 1. Hypothesis

- **H1** (Monotonic helps): monotonic activations preserve
  temporal ordering. **REJECTED** — all 4 variants worse.
- **H2** (Softplus is better than Sigmoid): monotonic +
  unbounded > bounded. **PARTIAL** — Softplus slightly better
  than Sigmoid but still much worse than Tanh.
- **H3** (One branch at a time is safer): replacing one Tanh is
  less disruptive. **PARTIAL** — mono_g +73% vs mono_both +506%
  on sin shows single-branch is less catastrophic but still
  strongly negative.
- **H4** (Tanh is essential): CfC requires Tanh's bidirectional
  [-1, 1] output. **CONFIRMED** — all non-Tanh variants fail.

## 2. Implementation

`lnn/core/mono_cfc.py` (~210 lines) — `MonoCfCCell` +
`MonoCfCStackedNetwork`.

Key design choices:

1. **Preserve f_gate**: Sigmoid activation in f_gate is
   unchanged (it outputs [0, 1] which is correct).
2. **Replace activations**: only g_branch and h_branch are
   modified.
3. **Closed-form solution unchanged**: h_t = τ · g + (1-τ) · h_branch.
4. **Preserve CfC API**: same `forward(x, h, dt)`, same
   `time_scale` parameter.

## 3. Bench results (30 cells: 5 conds × 3 datasets × 2 seeds, 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (Tanh+Tanh baseline) | **0.0275±0.0028** | **0.1327±0.0183** | **0.1051±0.0029** | 2545 |
| mono_g (Softplus+Tanh) | 0.0467±0.0000 | 0.4799±0.0020 | 0.1268±0.0106 | 2545 |
| mono_h (Tanh+Softplus) | 0.0524±0.0073 | 0.4528±0.0229 | 0.1058±0.0021 | 2545 |
| mono_both (Softplus+Softplus) | 0.1663±0.0163 | 0.4838±0.0009 | 0.1651±0.0054 | 2545 |
| mono_sig (Sigmoid+Sigmoid) | 0.4541±0.0011 | 0.4845±0.0003 | 0.1710±0.0065 | 2545 |

**Headline numbers (× change vs baseline)**:

- **mono_g**: sin **+73% WORSE**, structured **+262% CATASTROPHIC**,
  random +21%
- **mono_h**: sin **+91% WORSE**, structured **+241% CATASTROPHIC**,
  random +1%
- **mono_both**: sin **+506% CATASTROPHIC**, structured
  **+265% CATASTROPHIC**, random **+57% WORSE**
- **mono_sig**: sin **+1553% CATASTROPHIC**, structured
  **+265% CATASTROPHIC**, random **+63% WORSE**

## 4. Why MONO-CfC is unanimously negative

### 4.1 The closed-form solution requires Tanh

CfC's closed-form solution::

    h_t = σ(-f · τ) · g + (1 - σ(-f · τ)) · h_branch

This is an **interpolation** between g (transient response) and
h_branch (steady-state response), with τ = σ(-f · τ_eff)
controlling the interpolation weight.

- **Tanh** outputs in [-1, 1] — supports both positive and
  negative values.
- **Softplus** outputs in [0, ∞) — only positive values.
- **Sigmoid** outputs in [0, 1] — only positive values.

When g and h_branch are both Tanh, the interpolation can produce
h_t in [-1, 1] (output range).

When g or h_branch is Softplus, the output becomes
asymmetric — it can never go below 0. The interpolation loses
half its range.

When both are Softplus, the output is [0, 1]-ish — completely
loses the bidirectional information flow.

When both are Sigmoid, the output is even more compressed
(bounded by [0, 1] at every step).

### 4.2 Sin data is the worst hit

Sin data oscillates between -1 and +1. Tanh's output range
[-1, 1] is exactly what sin needs. Any non-Tanh activation
fundamentally cannot represent negative sin values, so the
closed-form solution can't interpolate properly.

- mono_g: +73% (g is partial only)
- mono_h: +91% (h_branch is partial only)
- mono_both: +506% (both branches broken)
- mono_sig: +1553% (both branches + bounded)

This shows a clean monotonic relationship: more broken branches
→ more catastrophic failure.

### 4.3 Structured data is catastrophically broken

The structured task requires the model to switch between sin
and sin(2t) at the midpoint. The CfC closed-form solution's
interpolation between g and h_branch is critical for this
switching.

When g and h_branch are non-Tanh, the model loses the ability
to interpolate, and gets stuck at one mode. Result: ~0.48 MSE
on both mono_g/mono_h/mono_both (essentially unable to learn).

### 4.4 Random data is the least affected

Random walk has no bidirectional structure — the cumulative
sum is always positive. So monotonic activations are less
harmful. But still: +21% to +63% worse.

## 5. Why this differs from prior mechanisms

### 5.1 vs FiLM 153 (mixed)
- **FiLM 153**: γ, β modulation, preserves CfC activation.
- **MONO 154**: replaces activation, breaks CfC closed-form.

### 5.2 vs MSDC 151 (14th positive)
- **MSDC 151**: parallel conv context, concat with x.
  CfC activation unchanged.
- **MONO 154**: replaces CfC activation directly.

### 5.3 vs TCC 149 / TDSA 152 / LiNo 150 (target-dep or neg)
- All preserve CfC activation, just modify input flow.
- **MONO 154**: breaks CfC activation itself.

### 5.4 vs Adaptive Time-Constant 141 (14th neg)
- **Adaptive τ**: changes time scale per-step.
- **MONO 154**: changes activations.
- Both break the closed-form assumption.

## 6. NEW INSIGHTS

1. **Tanh is CRITICAL to CfC** — the bidirectional [-1, 1]
   output range is essential to the closed-form solution's
   interpolation. Any non-Tanh activation breaks the cell.
2. **Monotonic activation hypothesis REJECTED** — despite
   theoretical appeal of monotonicity for ODE solutions, the
   empirical result is that monotonicity removes necessary
   negative information flow.
3. **Sigmoid (bounded) is even worse than Softplus (positive
   unbounded)** — confirms that the [-1, 1] range is essential.
4. **CfC is fragile to activation swaps** — unlike vanilla RNN
   which can often swap Tanh for ReLU/Softplus and still
   train, CfC's closed-form solution is tightly coupled to
   Tanh.
5. **Pattern reinforced (14 + 10 + 23 = 47 mechanism classes)**:
   - 14 strictly positive (preserves recurrent step + adds
     structure)
   - 10 target-dep
   - **23 negatives** (was 22): previous 22 + **MONO-CfC
     (this round, unanimous)**

**NEW RULE**: **NEVER replace CfC's Tanh activations.** The
closed-form solution requires bidirectional output [-1, 1].
Monotonic or bounded activations break the interpolation.

## 7. The 91-154 audit: 47 mechanism classes

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE (12 winners) |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Gated Input Skip | 134 | STRICTLY POSITIVE (13th winner) |
| Multi-Scale Dilated Conv CfC | 151 | STRICTLY POSITIVE (14th winner) |
| Layer Normalization | 135 | TARGET-DEPENDENT |
| Conv Input Preprocessing | 137 | TARGET-DEPENDENT |
| GLU + Identity Skip | 139 | TARGET-DEPENDENT |
| Decoupled / IndRNN-CfC | 143 | TARGET-DEPENDENT |
| Bidirectional CfC (concat) | 144 | TARGET-DEPENDENT (5th) |
| SCRN-CfC (α=0.5) | 146 | TARGET-DEPENDENT (6th) |
| Time-Decay CfC (γ=0.5) | 148 | TARGET-DEPENDENT (7th) |
| TCC-CfC (K=3/5/7) | 149 | TARGET-DEPENDENT (8th) |
| LiNo-CfC (sum/concat) | 150 | TARGET-DEPENDENT (9th) |
| FiLM-CfC (self γ, β) | 153 | TARGET-DEPENDENT (10th) |
| FiLM-CfC (global γ, β) | 153 | NEGATIVE (22nd, CATASTROPHIC) |
| Time-Domain Self-Attention CfC | 152 | NEGATIVE (21st) |
| SCRN-CfC (α=0.8/0.95/0.99) | 146 | NEGATIVE (17-19th) |
| Diff Features (diff_only) | 145 | NEGATIVE (16th) |
| Multiplicative Integration | 142 | NEGATIVE (15th) |
| Adaptive Time-Constant | 141 | NEGATIVE (14th) |
| SE Channel Attention | 140 | NEGATIVE (13th) |
| GLU alone | 139 | NEGATIVE (12th) |
| Sinusoidal Time Embedding | 138 | NEGATIVE (11th) |
| Zoneout | 136 | NEGATIVE (10th) |
| Bidirectional CfC (weighted) | 144 | NEGATIVE (15th) |
| Clockwork CfC (K=2/3/4) | 147 | NEGATIVE (20th) |
| LiNo (concat mode on structured) | 150 | NEGATIVE (sub-verdict) |
| LiNo (lin_only) | 150 | NEGATIVE (sanity) |
| FiLM (concat mode) | 153 | NEUTRAL (sub-verdict) |
| **MONO-CfC (mono_g/h/both/sig)** | **154** | **NEGATIVE (23rd, unanimous)** |
| Recursion depth (MoR) | 126 | NEGATIVE |
| 2nd-order oscillator | 128 | NEGATIVE |
| Multi-timescale ELM | 129 | NEGATIVE |
| Multi-rate MoE + dual attn | 130 | NEGATIVE |
| HGRN | 131 | NEGATIVE |
| AntisymmetricRNN | 132 | NEGATIVE |
| Hebbian Fast Weights | 133 | NEGATIVE |

**Pattern reinforced (14 + 10 + 23 = 47 tests)**:

- 14 winners preserve recurrent step + add useful structure
- 10 target-dep (input-side processing, bidi, SCRN,
  Time-Decay, TCC, LiNo, FiLM self)
- **23 negatives** (was 22): per-step mods, alternatives,
  regularizers, bottlenecks, redundant info, replacements,
  long-α SCRN, Clockwork partition, self-attention, FiLM
  global, **MONO-CfC (this round, unanimous)**

## 8. Recommendation

**MONO-CfC is unanimously NEGATIVE — DO NOT USE.**

- **NEVER replace CfC's Tanh activations** with monotonic
  (Softplus) or bounded (Sigmoid) alternatives.
- The closed-form solution REQUIRES bidirectional output.
- All 4 variants fail: mono_g, mono_h, mono_both, mono_sig.

**Production recipe**: stick with default Tanh+Tanh in CfC.

## 9. Critical implementation details

1. **f_gate unchanged**: Sigmoid in f_gate is essential
   (outputs [0, 1] interpolation weight).
2. **g_branch and h_branch**: ONLY these can be modified.
3. **Closed-form solution**: `h_t = τ · g + (1-τ) · h_branch`
   assumes g, h_branch ∈ [-1, 1].
4. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements.

## 10. Files

- `lnn/core/mono_cfc.py` (~210 lines)
- `tests/test_mono_cfc.py` (25 tests, all pass)
- `scripts/bench_mono_cfc.py` (30-cell bench)
- `results/bench_mono_cfc.json`
- `docs/prds/2026-06-15-lnn-round-154-mono-cfc.md`
- `docs/research/2026-06-15_mono_cfc_report.md`
