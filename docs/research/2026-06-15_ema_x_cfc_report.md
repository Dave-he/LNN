# Round 156 — EMA-X-CfC (Input EMA Augmentation)

**Date**: 2026-06-15
**PRD**: #10-118
**Verdict**: **ONE NEW STRICTLY POSITIVE WINNER** (ema_diff),
one TARGET-DEPENDENT (ema_concat), two NEGATIVES (ema_gate,
ema_ema_only).

## Summary

Round 156 tests **EMA-X-CfC** — augment CfC input with an
Exponential Moving Average (EMA) of the input, providing
explicit access to a smoothed / low-pass-filtered version of x::

    ema_t   = β · ema_{t-1} + (1 - β) · x_t
    aug_x_t = f_concat(x_t, ema_t)  # 4 variants

β = 0.9 (fixed hyperparameter). Four variants:
- **ema_concat**: aug_x = [x_t, ema_t], 2D input.
- **ema_gate**: aug_x = α·x_t + (1-α)·ema_t, D input, learned α.
- **ema_diff**: aug_x = [x_t, ema_t - x_t], 2D input.
- **ema_ema_only**: aug_x = ema_t only (control, replace x).

**Verdict**:

- **ema_diff**: sin **-11%**, structured **-42%**, random -1% —
  **17th STRICTLY POSITIVE**
- **ema_concat**: sin +5% (worse), structured **-29%**, random -1%
  — **12th TARGET-DEPENDENT**
- **ema_gate**: sin +49% WORSE, structured +92% CATASTROPHIC,
  random -1% — **25th NEGATIVE**
- **ema_ema_only**: sin +85% WORSE (var), structured +170%
  CATASTROPHIC, random +18% — **26th NEGATIVE**

## 1. Hypothesis

- **H1** (EMA helps periodic data): low-pass x is similar to
  the smoothed target. **PARTIAL** — ema_diff -11%, ema_concat
  +5% (slightly worse).
- **H2** (EMA helps regime change): smoother signal easier to
  predict. **CONFIRMED** — ema_diff -42%, ema_concat -29%.
- **H3** (EMA neutral on noise): smoothing helps noise. **PARTIAL**
  — random -1% to +18% (one variant +18%).
- **H4** (Diff (residual) is best): x - ema_x is the
  high-frequency signal. **CONFIRMED** — ema_diff is the ONLY
  strictly positive variant.

## 2. Implementation

`lnn/core/ema_x_cfc.py` (~280 lines) — `EMAXCfCCell` +
`EMAXCfCStackedNetwork`.

Key design choices:

1. **EMA state per cell, but only used at first layer** — the
   EMA's dim is the cell's input_size; for non-first layers,
   pass zeros to satisfy the cell's interface.
2. **β = 0.9** — fixed hyperparameter (long EMA window).
3. **Diff = high-pass** — `x - ema_x` is the high-frequency
   component of x, complementing ema_x's low-frequency content.
4. **Closed-form solution unchanged** — h_t = τ·g + (1-τ)·h_branch.

## 3. Bench results (30 cells: 5 conds × 3 datasets × 2 seeds, 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | 0.0275±0.0028 | 0.1327±0.0183 | 0.1051±0.0029 | 2545 |
| ema_concat | 0.0289±0.0033 | 0.0936±0.0116 | 0.1035±0.0036 | 3409 |
| ema_gate | 0.0411±0.0063 | 0.2543±0.0435 | 0.1035±0.0033 | 2547 |
| **ema_diff** | **0.0244±0.0019** | **0.0773±0.0514** | **0.1036±0.0024** | 3409 |
| ema_ema_only | 0.0510±0.0293 | 0.3578±0.1024 | 0.1237±0.0031 | 2545 |

**Headline (× change vs baseline)**:

- **ema_concat**: sin +5%, structured **-29%**, random -1%
- **ema_gate**: sin +49%, structured +92%, random -1%
- **ema_diff**: sin **-11%**, structured **-42%**, random -1%
- **ema_ema_only**: sin +85%, structured +170%, random +18%

## 4. Why ema_diff is a STRICTLY POSITIVE winner

### 4.1 The diff signal is a learnable high-pass filter

`diff_t = ema_t - x_t = (β · ema_{t-1} + (1-β) · x_t) - x_t
        = β · (ema_{t-1} - x_t) = -β · (x_t - ema_{t-1})`

The diff signal encodes how much x has changed since the EMA
was last updated. It's a high-pass filter on x.

By passing BOTH x_t (low + high freq) and (ema_t - x_t) (just
high freq), the model gets explicit access to:
- Original signal (all freq)
- High-frequency signal (only above the EMA cutoff)

This is a richer representation than TCC/MSDC (which add a
single conv context).

### 4.2 Sin data -11%

Sin's high-freq signal is cos (derivative of sin). The diff
signal tracks cos, providing explicit phase-velocity
information.

### 4.3 Structured data -42% — LARGEST among this round

The diff signal SPIKES at the regime switch (between sin and
sin(2t)), because the EMA lags behind the new regime. The
spike provides an explicit "regime change" marker — similar
to DELTA 155's Δh signal but on the input side.

### 4.4 Random data -1% (essentially neutral)

Random walk has unpredictable changes; diff is just noise.

## 5. Why ema_concat is TARGET-DEPENDENT

ema_concat passes [x, ema] (both low-pass and raw). The
concat doubles the input dim (2D=4), giving the model:
- Original x (all freq)
- Smoothed x (low-pass)

For periodic data (sin), the model already learns to use
both implicitly; adding ema_concat is redundant (+5%).

For regime-change data (structured), the ema signal helps
detect the smoothed background, but the diff signal is
better (-29% vs -42% for ema_diff).

## 6. Why ema_gate and ema_ema_only are NEGATIVE

### ema_gate: α-blend is hard to learn

α = sigmoid(0) = 0.5 at init → input is 50% x + 50% ema.
This is a high-pass filter on the input (removes low freq).
The model needs to learn α → 1 to recover the baseline, but
30 epochs is not enough.

### ema_ema_only: replacing x with ema loses information

Replacing x entirely with ema removes the high-freq
component. The model has to recover the high-freq from h
recurrence, which is harder.

## 7. Why this differs from prior mechanisms

### 7.1 vs DELTA-CfC 155 (15th, 16th positive)
- **DELTA 155**: hidden state deltas Δh.
- **EMA-X 156**: input EMA, diff = ema - x (high-pass on x).
- Both detect regime changes, but on different signals.

### 7.2 vs DiffCfC 145 (16th negative)
- **DiffCfC 145**: input deltas Δx, Δ²x (high-pass on x).
- **EMA-X ema_diff 156**: input ema-x (high-pass on x).
- Both are high-pass on x, but ema_diff uses low-pass cutoff
  at β=0.9 (long window), while DiffCfC uses Δx (1-step
  diff). Different cutoff, different signal.

### 7.3 vs TCC 149 (8th target-dep)
- **TCC 149**: parallel 1D conv context.
- **EMA-X 156**: input EMA state.
- TCC is a one-shot conv; EMA is a recurrent state.

### 7.4 vs MSDC 151 (14th positive)
- **MSDC 151**: parallel 1D convs at multiple dilations.
- **EMA-X 156**: input EMA at fixed β.
- Both are low-pass filters, but MSDC is one-shot; EMA is
  recurrent.

## 8. NEW INSIGHTS

1. **ema_diff is a NEW STRICTLY POSITIVE** mechanism. Passing
   [x, ema-x] gives both original and high-pass signals.
2. **Diff > Concat for EMA augmentation** — the high-pass
   signal is more useful than the low-pass signal.
3. **Gated α-blend is hard to learn** — ema_gate catastrophic
   despite having fewer parameters.
4. **Replacing x with ema (ema_ema_only) is CATASTROPHIC** —
   loses the high-freq information.
5. **Pattern reinforced (17 + 12 + 26 = 55 mechanism classes)**:
   - **17 strictly positive** (was 16): previous 16 +
     **ema_diff (this round)**
   - **12 target-dep** (was 11): previous 11 + **ema_concat
     (this round)**
   - **26 negatives** (was 24): previous 24 + **ema_gate +
     ema_ema_only (this round)**

**NEW RULE**: **For input-side processing, the high-pass signal
(ema_diff = ema - x) is more useful than the low-pass signal
(ema alone) or both combined.** Pass [x, ema-x] for clean
improvement.

## 9. The 91-156 audit: 55 mechanism classes

| Family | Rounds | Verdict |
|--------|--------|---------|
| 1st-order ODE + MoE | 76-78, 113-118, 123-125, 127 | STRICTLY POSITIVE (12 winners) |
| Diagnostics + gates | 91-99, 81-90 | STRICTLY POSITIVE |
| Irregular TS embedding | 102-103 | STRICTLY POSITIVE |
| Gated Input Skip | 134 | STRICTLY POSITIVE (13th winner) |
| Multi-Scale Dilated Conv CfC | 151 | STRICTLY POSITIVE (14th winner) |
| DELTA-CfC (concat) | 155 | STRICTLY POSITIVE (15th winner) |
| DELTA-CfC (concat_input) | 155 | STRICTLY POSITIVE (16th winner) |
| **EMA-X-CfC (diff)** | **156** | **STRICTLY POSITIVE (17th winner)** |
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
| DELTA-CfC (proj) | 155 | TARGET-DEPENDENT (11th) |
| **EMA-X-CfC (concat)** | **156** | **TARGET-DEPENDENT (12th)** |
| FiLM-CfC (global γ, β) | 153 | NEGATIVE (22nd, CATASTROPHIC) |
| DELTA-CfC (gated) | 155 | NEGATIVE (24th) |
| **EMA-X-CfC (gate)** | **156** | **NEGATIVE (25th)** |
| **EMA-X-CfC (ema_only)** | **156** | **NEGATIVE (26th)** |
| Time-Domain Self-Attention CfC | 152 | NEGATIVE (21st) |
| MONO-CfC (all 4 variants) | 154 | NEGATIVE (23rd, unanimous) |
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

**Pattern reinforced (17 + 12 + 26 = 55 tests)**:

- 17 winners preserve recurrent step + add useful structure
- 12 target-dep
- 26 negatives

## 10. Recommendation

**EMA-X-CfC: ONE new winner.**

- **DO use ema_diff** for general improvement (-11% sin,
  -42% structured).
- **DO use ema_concat** for regime-change data only (-29%
  structured, +5% sin worse).
- **DO NOT use ema_gate** (+92% structured CATASTROPHIC).
- **DO NOT use ema_ema_only** (+170% structured CATASTROPHIC).

**Production recipe**:
1. For regime-change-heavy data: **DELTA-CfC (concat_input)**
   from round 155 (still best: -50% structured).
2. For general improvement: **ema_diff** from this round
   (-11% sin, -42% structured).
3. For minimal params: stick with CfC baseline.

## 11. Critical implementation details

1. **EMA state per cell** — each cell's ema has dim = cell's
   input_size. For non-first layers, pass zeros.
2. **β = 0.9** — fixed; ablate later if needed.
3. **Diff = ema - x** — high-pass filter.
4. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements.

## 12. Files

- `lnn/core/ema_x_cfc.py` (~280 lines)
- `tests/test_ema_x_cfc.py` (27 tests, all pass)
- `scripts/bench_ema_x_cfc.py` (30-cell bench)
- `results/bench_ema_x_cfc.json`
- `docs/prds/2026-06-15-lnn-round-156-ema-x-cfc.md`
- `docs/research/2026-06-15_ema_x_cfc_report.md`
