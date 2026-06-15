# Round 129 — ELMCfC (arXiv:2605.12049 Spieler, Martius, Levina 2026)

**Date**: 2026-06-15
**PRD**: #10-91
**Commit**: TBD
**Verdict**: **HONEST NEGATIVE-WITH-NUANCE** — 14th negative
in 91-129 audit.

## Summary

Tested the **Expressive Leaky Memory (ELM)** neuron from
arXiv:2605.12049 (Spieler, Martius, Levina — 12 May 2026,
"Scaling Laws and Tradeoffs in Recurrent Networks of Expressive
Neurons"). The ELM neuron is a cortical-inspired recurrent
cell with multi-timescale leaky memory units, dendritic
branch structure, nonlinear MLP update proposals, and a
temporal high-pass filter on the output.

We adapted the ELM neuron to a recurrent cell for CfC:
- Each of H logical neurons has d_m memory units (Pareto recipe: d_m ~ √N_rec = 4 for H=16)
- Per-memory-unit learnable timescale κ_m
- Tanh-bounded MLP for update proposal
- High-pass filtered output: a = ReLU(b + w_r^T m - r)

**Verdict: HONEST NEGATIVE-WITH-NUANCE** — ELMCfC loses on
ALL 3 datasets, primarily because the high-pass filter
actively destroys the DC component of our targets.

## 1. Hypothesis

The paper claims a 3-axis scaling law (N units × k_e per-unit
complexity × k_c per-unit connectivity) with a closed-form
information-theoretic bound. The Pareto recipe is
d_m ~ √N, d_mlp = 2·d_m, d_tree = 2·d_mlp. The hypothesis:
adding per-neuron multiple memory units with multiple
timescales would give the cell more "effective capacity" per
logical neuron, helping on structured_irr (regime switch).

## 2. The ELM cell (simplified)

State: m ∈ [B, H, d_m] (memory), r ∈ [B, H] (EMA readout)
Input projection: in_proj([x, h_prev]) → [B, H, d_m]
MLP update: Δm = tanh(MLP([proj, m]))  # 2*d_m → d_mlp → d_m
Leaky integration: m_new = κ_m ⊙ m + (1 - κ_λ) ⊙ Δm
EMA readout: r_new = κ_r · r + (1 - κ_r) · w_r^T m
High-pass output: a = ReLU(b + w_r^T m - r)

We use d_m=4 (Pareto recipe for H=16), d_mlp=8.

## 3. Bench results (12 cells, 30 epochs, 2 seeds)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc**     | **0.0094**±0.0019 | **0.0053**±0.0010 | **0.0013**±0.0004 | 2545 |
| elm_cfc     | 0.0916±0.0149 | 0.1064±0.0249 | 0.1996±0.0864 | 3977 |

**ELMCfC loses on ALL 3 datasets** with 1.56× more params:
- sin: 0.0916 vs 0.0094 (**9.7× worse**)
- structured: 0.1064 vs 0.0053 (**20.1× worse**)
- random: 0.1996 vs 0.0013 (**153× worse**)

### Ablation: removing the high-pass filter (sin_irr only)

| Cond | s0 | s1 | n_params |
|------|----|----|----------|
| cfc | 0.0077 | 0.0079 | 2545 |
| elm_cfc (with high-pass) | 0.1064 | 0.0767 | 3977 |
| **elm_no_hp (no high-pass)** | **0.0140** | **0.0203** | 3977 |

**Removing the high-pass filter recovers ~50% of the loss**:
- 0.0916 → 0.0172 average on sin
- Still 2-3× worse than CfC

## 4. Why it fails

### 4.1 The high-pass filter is the main culprit

The high-pass output `a = ReLU(b + w_r^T m - r)` subtracts the
EMA of the readout. This is a "novelty detector" — it suppresses
the slow-varying component of the signal. **All 3 of our
targets have significant DC content**:
- sin: y = sin(t), zero mean but oscillates between ±1
- structured: y = sin(t) + sin(2t) + 0.3, has constant offset
- random: y = cumsum(noise), has large DC drift

The high-pass filter removes the constant offset, which the
network then cannot recover. **This is an architectural
mismatch between the cell's inductive bias and our 1D targets.**

### 4.2 The multi-timescale + MLP update doesn't help in 1D

Even without the high-pass filter, ELMCfC is 2-3× worse than
CfC. The reason: our 1D targets don't have a use for the
**multiple internal timescales per neuron** that ELM provides.
CfC's single τ per neuron is sufficient.

The ELM's dendritic branch structure (which we skipped) and
MLP-based update proposal add capacity that's hard to train
in 30 epochs with batch_size=8. The paper trains on SHD-Adding
(700×1000) and Enwik8 (10⁸ bytes) — much more data than our
1D toy bench.

### 4.3 The paper's setting is different

The paper uses ELM neurons for:
- SHD-Adding (spike pattern classification)
- Enwik8 (character-level LM)
- NeuronIO (somatic voltage prediction)

All are tasks where the **high-pass filter makes sense**
(cortical-inspired, the paper notes ELM's biological motivation
in adaptive gain control). Our 1D regression tasks don't have
the same inductive bias.

## 5. The 91-129 audit: 5th neuron-family tested

**Pattern (91-129)**: 25 structural mechanisms tested.
- **12 STRICTLY POSITIVE winners**: 99, 102, 105, 107, 113, 114, 116, 118, 123, 124, 125, 127
- **13 negatives/target-dep**: 108, 109, 110, 112, 115, 117, 119, 120, 121, 122, 126, 128, **129**

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Recursion depth (MoR) | 126 | NEGATIVE (5-axis Pareto failure) |
| 2nd-order oscillator (closed-form) | 128 | NEGATIVE (F constraint) |
| **Multi-timescale ELM** | **129** | **NEGATIVE (high-pass filter, no 1D use)** |

**NEW INSIGHT (round 129)**: The ELM paper's biological
motivation (cortical adaptive gain control) is **specific to
its target tasks** (SHD, Enwik8, NeuronIO). The high-pass
filter is a strong inductive bias that helps on spike-based
classification and language modeling but actively HURTS on
**continuous-valued regression** with DC content. The ELM
neuron's other innovations (multi-timescale memory, MLP
update) don't add value in our 1D setting.

## 6. Critical implementation details

1. **Skipped dendritic branches** — used linear input
   projection instead of `branch_sum`. The paper's d_tree
   branches are biologically motivated but our 1D setting
   doesn't need them.
2. **Pareto recipe d_m=4** for H=16. d_mlp=8 = 2·d_m per the
   paper. Larger d_m (e.g. 8) didn't help (tested in
   `test_elm_different_d_m_values`).
3. **κ_m, κ_λ initialized via sigmoid(-2) ≈ 0.12** — slow
   decay, similar to slow τ in LTC.
4. **κ_r fixed at exp(-1/1.0) ≈ 0.37** — readout EMA.
5. **MLP input is 2·d_m** (proj + prev_memory per the paper's
   `[b_t, κ_m ⊙ m_{t-1}]`).
6. **High-pass output**: `a = ReLU(b + w_r^T m - r)`. The
   ReLU is the paper's choice; we could also use tanh or
   identity, but ReLU is the "novelty detector" form.

## 7. Future work

1. **Test ELM on PhysioNet 36D** — paper's NeuronIO is
   physiological, may be a better match.
2. **ELM as a MoE expert** — the multi-timescale memory
   may help in regime-switched data when combined with
   expert routing.
3. **ELM with no high-pass + tanh output** — may recover
   more of the gap.
4. **ELM with longer training** — paper uses 100+ epochs;
   our 30 may be too short.
5. **ELM with larger d_m** — Pareto recipe may be wrong
   for our small setting.

## Why it works (where it could)

The ELM's win condition is:
- **Spike-based or high-frequency data** (where the high-pass
  filter is a feature, not a bug)
- **Large training data** (the paper uses orders of magnitude
  more than our 1D bench)
- **Tasks that benefit from novelty detection** (anomaly
  detection, regime identification)
- **Long sequences** (where multi-timescale memory helps)

None of these conditions hold in our 1D continuous regression
bench. The negative result is **specific to our 1D
continuous-valued regression setting**, not a refutation of
the paper's claim in their biological/spike-based setting.
