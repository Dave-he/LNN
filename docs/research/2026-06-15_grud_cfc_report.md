# Round 148 — GRU-D / Time-Decay CfC (Che 2018, Jia & Benson 2019)

**Date**: 2026-06-15
**PRD**: #10-110
**Verdict**: **TARGET-DEPENDENT (7th)** — heavy decay wins on sin, loses on structured/random.

## Summary

Round 148 tests the **GRU-D / CT-RNN / ODE-RNN** idea of exposing
the time delta Δt to the cell and applying a learnable per-feature
decay to the hidden state between observations::

    decay_t = exp(-γ * Δt)        # in (0, 1]
    h_t     = h_{t-1} * decay_t   # time-aware decay
    h_t     = CfCCell(x_t, h_t)   # standard CfC update

**Verdict**: TARGET-DEPENDENT (7th in 91-148 audit):

- **grud_g0 (γ≈0)**: EXACTLY matches cfc baseline — sanity check passes
- **grud_g005 (γ≈0.05, light decay)**: neutral on sin/structured,
  slightly better on random (-23%)
- **grud_g05 (γ≈0.5, heavy decay)**:
  - sin: **0.0060 vs 0.0094 — 36% better (STRICTLY POSITIVE)**
  - structured: 0.0068 vs 0.0053 — 28% worse
  - random: 0.0027 vs 0.0013 — 108% worse

## 1. Hypothesis

- **H1** (Sin data): per-feature decay should help smooth periodic
  data. **CONFIRMED for heavy decay** (γ=0.5 gives -36%).
- **H2** (Structured data): per-feature decay should help on
  regime-change data. **REJECTED** — heavy decay LOSES by 28% on
  structured (decay wipes the long-term memory needed to remember
  the regime).
- **H3** (Random data): per-feature decay should hurt noisy data
  (decay wipes useful memory). **CONFIRMED for heavy decay** (+108%).
- **H4** (Different γ): init γ=0 (no decay) vs γ=0.05 (light)
  vs γ=0.5 (heavy). **CONFIRMED** — γ=0.5 has the strongest effect.

## 2. Implementation

`lnn/core/grud_cfc.py` (~140 lines) — `TimeDecayCfCCell` +
`TimeDecayCfCStackedNetwork`.

Key design choices:

1. **softplus γ parameterization**: `γ = softplus(gamma_param) ≥ 0`.
   Init at gamma_param = -3.0 (γ ≈ 0.05, very light decay).
2. **Per-feature γ**: one γ per hidden unit. So different features
   can decay at different rates.
3. **Time-aware dt input**: dt (B, T, 1) is an optional input. We
   use dt[:, t, :] as the per-step time delta. dt=1.0 for regular TS.
4. **NaN handling**: zero-fill input AND zero-fill dt (use dt=0 =
   decay=1 = no decay on missing data).
5. **Preserves CfC**: h goes through the standard CfC update after
   decay. This is a STRUCTURAL addition (preserves the recurrent
   step), not a per-step modification.

## 3. Bench results (24 cells: 4 conds × 3 datasets × 2 seeds, 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | 0.0094±0.0019 | **0.0053±0.0010** | **0.0013±0.0004** | 2545 |
| grud_g0 (γ≈0) | 0.0094±0.0019 | 0.0053±0.0010 | 0.0013±0.0004 | 2577 |
| grud_g005 (γ≈0.05) | 0.0095±0.0016 | 0.0054±0.0006 | 0.0010±0.0001 | 2577 |
| **grud_g05 (γ≈0.5)** | **0.0060±0.0010** | 0.0068±0.0014 | 0.0027±0.0004 | 2577 |

**Headline numbers (× change vs baseline)**:

- **grud_g0**: NO CHANGE (sanity check, γ=0 → decay=1)
- **grud_g005**: sin 1.0× (neutral), structured 1.0× (neutral), random -23% (slight improvement)
- **grud_g05**: sin **-36%** (better!), structured +28% (worse), random +108% (worse)

## 4. Why heavy decay (γ=0.5) helps sin but hurts structured/random

For T=32 with γ=0.5:
- decay per step = exp(-0.5) = 0.61
- After 32 steps, h is decayed by 0.61^32 ≈ 6e-7 (essentially zero)

This forces the model to rely on RECENT input rather than long-term
memory:

- **sin**: highly periodic, no need for long-term memory. Recent
  input tells the model where on the sinusoid we are. **Heavy decay
  works.**
- **structured**: regime change at t=T/2 requires long-term memory
  to know "we are in the slow regime". Heavy decay wipes this. **Loss.**
- **random**: cumulative noise — heavy decay interferes with smooth
  integration. **Loss.**

## 5. Why this differs from Clockwork 147 (NEGATIVE) and SCRN 146 (target-dep)

### 5.1 Clockwork 147 (NEGATIVE 20th)
- Binary carry-forward (h stays the same for K-1 steps).
- Slow modules get only 4 gradient updates per T=32 sequence.
- Result: ALL variants LOSE on ALL datasets (max 20.7× worse).

### 5.2 SCRN 146 (target-dep 6th, α=0.5)
- Parallel slow context stream (separate from main h).
- Fixed α (no per-feature learning, no time-awareness).
- Result: α=0.5 WINS on sin/structured, LOSES on random.

### 5.3 GRU-D / Time-Decay 148 (target-dep 7th)
- Decay applied to MAIN h (no parallel stream).
- Learnable per-feature γ.
- Time-aware (depends on actual dt).
- Result: γ=0.5 WINS on sin, LOSES on structured/random.

**Key difference from SCRN 146**: SCRN's slow context is a
*parallel additive* stream — even at α=0.5, the slow context
retains memory of the past. Time-Decay is *multiplicative* — it
directly scales down h. So Time-Decay has a more aggressive
memory effect than SCRN.

**Why sin benefits more from Time-Decay than SCRN**: Time-Decay
at γ=0.5 gives decay=0.61 per step, which essentially makes the
model "forget" the past quickly. For periodic data, this is good
because the past is predictable from the current input. SCRN at
α=0.5 keeps the slow context from 16 steps ago, which is useful
for non-periodic structured data.

## 6. NEW INSIGHTS

1. **Time-aware decay is a valid mechanism** but its optimal γ
   depends on the data structure:
   - γ=0 (no decay): safe, equivalent to baseline
   - γ=0.05 (light): safe, neutral or slightly positive
   - γ=0.5 (heavy): only good for periodic data
2. **Sin benefits from HEAVY decay** (-36% with γ=0.5) — this is
   the first strictly positive result on sin since the audit began.
3. **Multiplicative decay (this round) vs additive slow context
   (SCRN 146)**: they target different use cases. SCRN is good for
   regime-change data, Time-Decay is good for periodic data.
4. **Pattern reinforced (13 + 7 + 20 = 40 mechanism classes)**:
   - 13 winners preserve recurrent step + add useful structure
   - **7 target-dep**: input-side processing, bidi, SCRN,
     **Time-Decay (this round)**
   - 20 negatives: per-step mods, alternatives, regularizers,
     bottlenecks, redundant info, replacements, long-α SCRN,
     Clockwork partition

**NEW RULE**: Heavy time decay (γ ≥ 0.5) is only safe for
periodic data. For regime-change or noisy data, use light decay
(γ ≤ 0.1) or skip the mechanism entirely.

## 7. The 91-148 audit: 41 mechanism classes

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
| **Time-Decay CfC (γ=0.5)** | **148** | **TARGET-DEPENDENT (7th)** |
| **Time-Decay CfC (γ=0.05)** | **148** | **neutral-safe** |
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
| Clockwork CfC (K=2/3/4) | 147 | NEGATIVE (20th) |
| Recursion depth (MoR) | 126 | NEGATIVE |
| 2nd-order oscillator | 128 | NEGATIVE |
| Multi-timescale ELM | 129 | NEGATIVE |
| Multi-rate MoE + dual attn | 130 | NEGATIVE |
| HGRN | 131 | NEGATIVE |
| AntisymmetricRNN | 132 | NEGATIVE |
| Hebbian Fast Weights | 133 | NEGATIVE |

**Pattern reinforced (13 + 7 + 20 = 40 mechanism classes)**:

- 13 winners preserve recurrent step + add useful structure
- **7 target-dep**: input-side processing (LN 135, conv 137, GLU+skip
  139, decoupled/IndRNN 143, bidi_concat 144), SCRN α=0.5 (146),
  **Time-Decay γ=0.5 (148, this round)**
- 20 negatives: per-step mods, alternatives, regularizers,
  bottlenecks, redundant info, replacements, long-α SCRN,
  Clockwork partition

## 8. Recommendation

**Time-Decay CfC is the 7th TARGET-DEPENDENT in the 91-148 audit.**

- **DO use Time-Decay CfC (γ ≈ 0.5)** for PERIODIC data (sin-like).
  We see 36% improvement on sin_irr.
- **DO NOT use Time-Decay CfC (γ ≥ 0.5)** for regime-change or
  noisy data. Use light decay (γ ≤ 0.1) or no decay.
- **Time-Decay γ=0.05 is a safe default** if you're unsure.
- **Production recipe**: detect data periodicity and enable heavy
  decay only when periodic.

## 9. Critical implementation details

1. **softplus γ**: `γ = softplus(gamma_param) ≥ 0`.
2. **dt = 1.0 for regular TS**: matches standard time step.
3. **NaN handling**: zero-fill dt, then decay=1 (no decay) on NaN.
4. **Per-feature γ**: each hidden unit has its own decay rate.
5. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements — torch imports work fine at runtime
   via `.venv312/bin/python`.
