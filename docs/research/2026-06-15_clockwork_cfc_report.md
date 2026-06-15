# Round 147 — Clockwork CfC (Koutnik 2014 CW-RNN)

**Date**: 2026-06-15
**PRD**: #10-109
**Verdict**: **HONEST NEGATIVE** — 20th negative in 91-147 audit. ALL variants LOSE on ALL datasets.

## Summary

Round 147 tests the **Clockwork RNN (CW-RNN)** idea from Koutnik
et al. 2014 applied to CfC. The hidden state is partitioned into
K modules, each updating only at its assigned period (2^k)::

    # At step t, only module k updates if t mod 2^k == 0
    for k in range(K):
        if t mod (2 ** k) == 0:
            h_k = CfCCell_k(x_t, h_k)
        else:
            h_k = h_k   # carry forward
    h_combined = concat([h_0, h_1, ..., h_{K-1}])

**Verdict**: ALL THREE variants LOSE on ALL THREE datasets:

- **cw_k2** (periods 1, 2): sin 2.95×, structured 4.74×, random 5.69× worse
- **cw_k3** (periods 1, 2, 4): sin 3.69×, structured 9.26×, random **20.7×** worse
- **cw_k4** (periods 1, 2, 4, 8): sin 4.71×, structured **12.85×**, random 19.6× worse

**HEADLINE**: Clockwork partition is CATASTROPHIC for CfC in
T=32 sequences. The slower modules get too few gradient updates
(4 updates per sequence for K=4, period 8) and the carry-forward h
creates discontinuities that don't align with smooth data.

## 1. Hypothesis

- **H1** (Smooth data): slow modules capture long-term trends.
  **REJECTED for all K** (sin 2.95-4.71× worse).
- **H2** (Structured data): slow modules capture regime patterns.
  **REJECTED for all K** (structured 4.74-12.85× worse).
- **H3** (Random data): slow modules "smooth out" noise.
  **REJECTED for all K** (random 5.69-20.7× worse).
- **H4** (Different K): K=2 minimal, K=4 maximum.
  **REJECTED**: K=2 is the best, but still loses on all datasets.

## 2. Implementation

`lnn/core/clockwork_cfc.py` (~140 lines) — `ClockworkCfCCell` +
`ClockworkCfCStackedNetwork`.

Key design choices:

1. **Auto-equal module sizes**: hidden_size // K per module, last
   `rem` modules absorb remainder.
2. **Periods**: 2^0, 2^1, ..., 2^{K-1} (so K=3 → 1, 2, 4).
3. **Carry-forward h**: when module doesn't update, h stays the same.
4. **NaN handling**: zero-fill input per step.
5. **Per-layer Clockwork**: each layer has its own K modules.

## 3. Bench results (24 cells: 4 conds × 3 datasets × 2 seeds, 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | **0.0094±0.0019** | **0.0053±0.0010** | **0.0013±0.0004** | 2545 |
| cw_k2 | 0.0277±0.0006 | 0.0251±0.0001 | 0.0074±0.0028 | 1777 |
| cw_k3 | 0.0347±0.0044 | 0.0491±0.0074 | 0.0269±0.0105 | 1525 |
| cw_k4 | 0.0443±0.0042 | 0.0681±0.0016 | 0.0255±0.0047 | 1393 |

**Headline numbers (× worse than baseline)**:

- **cw_k2**: sin **2.95×**, structured 4.74×, random 5.69×
- **cw_k3**: sin 3.69×, structured 9.26×, random **20.7×**
- **cw_k4**: sin 4.71×, structured **12.85×**, random 19.6×

## 4. Why Clockwork fails for CfC in T=32 sequences

### 4.1 Too few params per module

With hidden=16 and K=4, each module has only 4 hidden units. The
CfC's f_gate, g_branch, h_branch are Linear(input+hidden, hidden)
= Linear(6, 4) for module 0 of layer 0. That's not enough
capacity to learn anything useful per module.

The 1393-1777 param count for clockwork is 30-45% smaller than the
baseline 2545, so clockwork starts at a disadvantage.

### 4.2 Too few gradient updates for slow modules

For T=32 with K=4:
- Module 0 (period 1): 32 updates per sequence
- Module 1 (period 2): 16 updates
- Module 2 (period 4): 8 updates
- Module 3 (period 8): 4 updates (only!)

With 30 epochs of training, module 3 sees only 4 * 30 = 120
gradient updates total. That's not enough to learn anything.

### 4.3 Carry-forward h creates discontinuities

When module 1 (period 2) doesn't update, its h stays the same
for 2 timesteps, then jumps. This is a discrete "phase change"
that doesn't align with smooth data like sin_irr, where the
target evolves continuously.

The baseline CfC can smoothly track changes every timestep; the
clockwork partition forces discrete updates that hurt the
smooth-data fit.

## 5. Why this is different from SCRN (146, target-dep)

SCRN (146) had a parallel slow context stream — the slow context
ALSO evolves every step (just with EMA decay). So the slow context
gets gradient updates at every step, just weighted by α.

Clockwork's slow modules get NO updates at all on most steps.
That's a fundamentally different (and worse) design for T=32
sequences.

## 6. NEW INSIGHTS

1. **Multi-timescale partition fails for T=32** because slow
   modules get too few gradient updates. The audit's T=64/T=128
   extension would likely be needed for Clockwork to work.
2. **Per-step carry-forward is a per-step modification** that
   loses. Even though the partition is structural, the carry-
   forward is a per-step "decision" (whether to update or not),
   and per the audit pattern (ATC 141, MI 142, zoneout 136, etc.),
   per-step modifications lose.
3. **Capacity matters more than partition** in 1D. The baseline
   CfC with 2545 params wins despite being "unstructured".
4. **Pattern reinforced (13 + 6 + 20 = 39 tests)**:
   - 13 winners preserve recurrent step + add useful structure
   - 6 target-dep: input-side processing that PRESERVES x
   - 20 negatives: per-step mods, alternatives, regularizers,
     bottlenecks, redundant info, replacements, long-α SCRN,
     **clockwork partition**

**NEW RULE**: Multi-timescale PARTITION (with carry-forward) is
NOT the same as multi-timescale PARALLEL streams (with EMA
decay). Partition fails for short sequences (T<64) due to
insufficient gradient updates for slow modules.

## 7. The 91-147 audit: 40 mechanism classes

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE (12 winners) |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Gated Input Skip | 134 | STRICTLY POSITIVE (13th winner) |
| **Layer Normalization** | 135 | **TARGET-DEPENDENT** |
| **Conv Input Preprocessing** | 137 | **TARGET-DEPENDENT** |
| **GLU + Identity Skip** | 139 | **TARGET-DEPENDENT** |
| **Decoupled / IndRNN-CfC** | 143 | **TARGET-DEPENDENT** |
| **Bidirectional CfC (concat)** | 144 | **TARGET-DEPENDENT (5th)** |
| **SCRN-CfC (α=0.5)** | 146 | **TARGET-DEPENDENT (6th)** |
| **Clockwork CfC (K=2/3/4)** | **147** | **NEGATIVE (20th)** |
| SCRN-CfC (α=0.8/0.95/0.99) | 146 | NEGATIVE (17-19th) |
| Diff Features (diff_only) | 145 | NEGATIVE (16th) |
| Diff Features (concat) | 145 | NEUTRAL |
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

**Pattern reinforced (13 + 6 + 20 = 39 + this 20th = 40 tests)**:
- 13 winners preserve recurrent step + add useful structure
- 6 target-dep: input-side processing (all 6 preserve x) OR
  bidirectional structural addition
- **20 negatives**: per-step mods, alternatives, regularizers,
  bottlenecks, redundant info, replacements, long-α SCRN,
  **clockwork partition (this round)**

## 8. Recommendation

**Clockwork CfC is the 20th NEGATIVE in the 91-147 audit.**

- **DO NOT use Clockwork CfC** for T≤64 sequences — slow modules
  get too few gradient updates and carry-forward h creates
  discontinuities.
- **The Clockwork idea may work for T≥128** (where slow modules
  get more gradient updates), but we have not tested that.
- **Stick with the 6 target-dep winners** for production when
  smooth/structured data is expected (LN, conv, GLU+skip,
  decoupled, bidi_concat, scrn_05).

## 9. Critical implementation details

1. **Auto-equal module sizes**: hidden_size // K per module, last
   `rem` modules absorb remainder (each gets +1).
2. **Periods**: 2^0, 2^1, ..., 2^{K-1}.
3. **Carry-forward h**: when module doesn't update, h stays the
   same. NO gradient flows through carry-forward.
4. **NaN handling**: zero-fill input per step.
5. **Per-layer Clockwork**: each layer has its own K modules.
6. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements — torch imports work fine at runtime
   via `.venv312/bin/python`.
