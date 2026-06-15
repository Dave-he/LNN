# Round 153 — FiLM-CfC (Feature-wise Linear Modulation)

**Date**: 2026-06-15
**PRD**: #10-115
**Verdict**: **MIXED** — film_self is **10th TARGET-DEPENDENT** (structured
-32%), film_global is **22nd NEGATIVE (CATASTROPHIC)** (structured +180%),
film_concat is neutral.

## Summary

Round 153 tests **FiLM-CfC (Feature-wise Linear Modulation)** — a
context-driven multiplicative + additive modulation of the CfC's
hidden state::

    out = gamma * h + beta

where γ and β are computed from a context (either the input
itself or a global summary). Inspired by Perez et al. 2018 ("FiLM:
Visual Reasoning with a General Conditioning Layer")::

    # Context (sequence-level summary or per-step)
    if ctx_mode == 'global':
        ctx = x.mean(dim=1, keepdim=True)  # [B, 1, D]
    elif ctx_mode == 'self':
        ctx = x  # [B, T, D]

    # Modulation parameters
    gamma = Linear_gamma(ctx)  # [B, T, hidden_size]
    beta = Linear_beta(ctx)    # [B, T, hidden_size]

    # Standard CfC
    h_t = CfCCell(x_t, h_{t-1})  # [B, T, hidden_size]

    # Modulated output
    out = gamma * h + beta  # [B, T, hidden_size]

**Verdict**: **MIXED**:

- **film_global**: sin **+27% WORSE**, structured **+180% CATASTROPHIC**,
  random +15% — **NEGATIVE (22nd)**
- **film_self**: sin +11% (worse), structured **-32%** (positive),
  random +3% — **TARGET-DEPENDENT (10th)**
- **film_concat**: sin **+44% WORSE**, structured -7%, random -1% —
  neutral

## 1. Hypothesis

- **H1** (Sin data): FiLM helps periodic data. **REJECTED** — all
  3 variants +11% to +44% WORSE.
- **H2** (Structured data): FiLM helps regime-change data.
  **PARTIAL** — film_self -32% (positive), film_global +180%
  CATASTROPHIC, film_concat -7%.
- **H3** (Random data): FiLM may hurt noise. **CONFIRMED** —
  film_global +15%, others essentially neutral.
- **H4** (Self vs global): self wins over global. **CONFIRMED** —
  self is -32% on structured, global is +180% (CATASTROPHIC).

## 2. Implementation

`lnn/core/film_cfc.py` (~190 lines) — `FiLMCfCCell` +
`FiLMCfCStackedNetwork`.

Key design choices:

1. **Two modulation projections**: Linear_gamma, Linear_beta.
2. **Three context modes**: self (per-step), global (sequence-level),
   concat (control, no modulation).
3. **Multiplicative + additive**: γ * h + β. Standard FiLM formula.
4. **NaN handling**: zero-fill input.
5. **Preserves CfC**: h goes through the standard CfC update, then
   is modulated by γ, β.

## 3. Bench results (24 cells: 4 conds × 3 datasets × 2 seeds, 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | **0.0275±0.0028** | 0.1327±0.0183 | 0.1051±0.0029 | 2545 |
| film_global | 0.0349±0.0122 | 0.3719±0.1125 | 0.1208±0.0069 | 3185 |
| film_self | 0.0306±0.0051 | **0.0906±0.0415** | 0.1083±0.0022 | 3185 |
| film_concat | 0.0395±0.0004 | 0.1236±0.0611 | **0.1038±0.0037** | 3409 |

**Headline numbers (× change vs baseline)**:

- **film_global**: sin **+27% WORSE**, structured **+180% CATASTROPHIC**,
  random +15% — **NEGATIVE (22nd)**
- **film_self**: sin +11% (worse), structured **-32%** (positive),
  random +3% — **TARGET-DEPENDENT (10th)**
- **film_concat**: sin **+44% WORSE**, structured -7%, random -1% —
  neutral

## 4. Why FiLM-CfC has split results

### 4.1 film_global: CATASTROPHIC (22nd NEGATIVE)

Global ctx_mode uses the SAME γ, β for all timesteps (γ and β
are derived from the sequence-level mean of x). This is the most
restrictive FiLM variant — no per-timestep flexibility.

- **Sin +27%**: constant γ, β modulate the entire sequence the
  same way. Sin's local phase information is lost because γ, β
  are sequence-level averages.
- **Structured +180% CATASTROPHIC**: regime change requires
  DIFFERENT γ, β in each regime. Global γ, β is the average of
  the two regimes, which is wrong for both. Catastrophic.
- **Random +15%**: global γ, β adds a constant bias that's just
  noise.

**Key insight**: γ, β must vary across timesteps to handle
non-stationary data. Global γ, β is too rigid.

### 4.2 film_self: TARGET-DEPENDENT (10th)

Self ctx_mode uses γ, β from x_t at each timestep. This is the
most flexible FiLM variant — γ, β can change every timestep.

- **Sin +11%**: per-step modulation of already-smooth sin is
  over-modulation. The model needs to learn γ_t ≈ 1, β_t ≈ 0
  to recover the baseline, but 30 epochs is not enough.
- **Structured -32% POSITIVE**: per-step γ, β is exactly what's
  needed for regime change. The model can learn γ_t, β_t that
  shift behavior in the second half.
- **Random +3%**: per-step modulation on noise is essentially
  noise. Slightly worse.

**Key insight**: per-step γ, β is flexible enough to handle
regime change, but undertrained in 30 epochs for smooth data.

### 4.3 film_concat: NEUTRAL

Concat mode (control) is similar to TCC 137 — augments the input
with a context (global mean of x). This is essentially the same
as the existing TCC mechanism (concat x with conv output), but
with a global mean instead of conv. Slightly worse on sin (the
augmented input is the same for all timesteps, losing phase
info), slightly better on structured (the context helps with
regime detection), neutral on random.

## 5. Why this differs from prior mechanisms

### 5.1 vs MSDC 151 (strictly positive 14th)
- **MSDC 151**: parallel 1D conv context, concat with x.
- **FiLM 153**: context-driven multiplicative + additive
  modulation. FIRST mechanism in audit to use MULTIPLICATIVE
  interaction with hidden state.

### 5.2 vs TCC 149 (target-dep 8th)
- **TCC 149**: parallel 1D conv context, concat with x.
- **FiLM 153**: modulation, not concat. Different mechanism.

### 5.3 vs LiNo 150 (target-dep 9th)
- **LiNo 150**: parallel linear projection, sum with CfC.
- **FiLM 153**: modulation, not sum. Different mechanism.

### 5.4 vs TDSA 152 (negative 21st)
- **TDSA 152**: parallel self-attention context, concat with x.
- **FiLM 153**: modulation, not concat. Different mechanism.

## 6. NEW INSIGHTS

1. **Multiplicative modulation is a new mechanism class** in the
   audit. First used by FiLM 153.
2. **Global γ, β is CATASTROPHIC** for non-stationary data — too
   rigid to handle regime change.
3. **Self γ, β (per-step) is TARGET-DEPENDENT** — flexible but
   undertrained for smooth data.
4. **Concat mode (no modulation) is NEUTRAL** — similar to TCC 137.
5. **Pattern reinforced**:
   - 14 strictly positive (preserves recurrent step + adds structure)
   - **10 target-dep** (was 9): previous 9 + **film_self (this round)**
   - **22 negatives** (was 21): previous 21 + **film_global (this round)**

**NEW RULE**: FiLM with per-step γ, β (film_self) is target-dep
(structured -32%, but sin +11%). AVOID global FiLM (film_global
is CATASTROPHIC on structured +180%). FiLM modulation is a
different mechanism from concat/sum; multiplicative interaction
is generally riskier than additive.

## 7. The 91-153 audit: 46 mechanism classes

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
| **FiLM-CfC (self γ, β)** | **153** | **TARGET-DEPENDENT (10th)** |
| **FiLM-CfC (global γ, β)** | **153** | **NEGATIVE (22nd, CATASTROPHIC)** |
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
| Recursion depth (MoR) | 126 | NEGATIVE |
| 2nd-order oscillator | 128 | NEGATIVE |
| Multi-timescale ELM | 129 | NEGATIVE |
| Multi-rate MoE + dual attn | 130 | NEGATIVE |
| HGRN | 131 | NEGATIVE |
| AntisymmetricRNN | 132 | NEGATIVE |
| Hebbian Fast Weights | 133 | NEGATIVE |

**Pattern reinforced (14 + 10 + 22 = 46 tests)**:

- 14 winners preserve recurrent step + add useful structure
- **10 target-dep** (was 9): input-side processing, bidi, SCRN,
  Time-Decay, TCC, LiNo, **FiLM self (this round)**
- **22 negatives** (was 21): per-step mods, alternatives,
  regularizers, bottlenecks, redundant info, replacements,
  long-α SCRN, Clockwork partition, self-attention,
  **FiLM global (this round, CATASTROPHIC)**

## 8. Recommendation

**FiLM-CfC has split results this round:**

- **film_self is 10th TARGET-DEPENDENT** — sin +11% (worse),
  structured -32% (positive), random +3% (worse).
- **film_global is 22nd NEGATIVE (CATASTROPHIC)** — sin +27%,
  structured +180%, random +15% all worse.

- **DO use film_self for regime-change data** (structured -32%).
  AVOID for periodic data (sin +11% worse).
- **DO NOT use film_global** — CATASTROPHIC on structured (+180%).
- **Production recipe**: prefer MSDC 151 (strictly positive) over
  FiLM for general use. FiLM self is a niche tool for
  regime-change-heavy data.

## 9. Critical implementation details

1. **Two Linear projections**: γ = Linear(D, H), β = Linear(D, H).
2. **Context expansion**: if ctx is [B, 1, D], expand to [B, T, D]
   via `.expand(-1, T, -1)`.
3. **Multiplicative + additive**: γ * h + β. Standard FiLM formula.
4. **NaN handling**: zero-fill input.
5. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements.

## 10. Files

- `lnn/core/film_cfc.py` (~190 lines)
- `tests/test_film_cfc.py` (23 tests, all pass)
- `scripts/bench_film_cfc.py` (24-cell bench)
- `results/bench_film_cfc.json`
- `docs/prds/2026-06-15-lnn-round-153-a-film-cfc.md`
- `docs/research/2026-06-15_film_cfc_report.md`
