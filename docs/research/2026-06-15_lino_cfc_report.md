# Round 150 — Linear-Nonlinear CfC (LiNo-CfC)

**Date**: 2026-06-15
**PRD**: #10-112
**Verdict**: **TARGET-DEPENDENT (9th)** — sin wins -18% to -20%, structured neutral/sum and broken/concat, random neutral.

## Summary

Round 150 tests **Linear-Nonlinear CfC (LiNo-CfC)** — a parallel
linear projection stream + nonlinear CfC stream, summed (or
concatenated) at the output. Inspired by the LiNo framework (PKU/HK
PolyU, Jan 2025) and DLinear (Zeng et al. 2022 AAAI)::

    # Linear stream: per-step linear projection (no recurrence)
    h_lin = x @ W_lin + b_lin  # [B, T, hidden_size]

    # Nonlinear stream: standard CfC
    h_nl = CfCCell(x, h)  # [B, T, hidden_size]

    # Combine: sum (LiNo original) or concat
    h = h_lin + h_nl  # or concat

The key idea: many time series have a strong linear trend component
captured by a simple linear projection, while the nonlinear residual
requires CfC. Combining both covers both modes.

**Verdict**: TARGET-DEPENDENT (9th in 91-150 audit):

- **lino_sum**: sin **-18%**, structured neutral, random neutral
- **lino_concat**: sin **-20%**, structured **+88% WORSE**, random -1%

## 1. Hypothesis

- **H1** (Sin data): linear stream should help smooth periodic data
  (sin has linear trends locally). **CONFIRMED** (-18% to -20%).
- **H2** (Structured data): linear stream should help regime-change
  data. **REJECTED** — sum is neutral, concat BREAKS (+88%).
- **H3** (Random data): linear stream should hurt noisy data.
  **REJECTED** — random is essentially neutral across all conditions.
- **H4** (Concat vs sum): sum is LiNo's original formulation.
  **CONFIRMED** — sum is safer (concat breaks on structured).

## 2. Implementation

`lnn/core/lino_cfc.py` (~190 lines) — `LinearNonlinearCfCCell` +
`LinearNonlinearCfCStackedNetwork`.

Key design choices:

1. **Linear stream**: a single nn.Linear (input_size → hidden_size).
   Applied to x at every step. NO RECURRENCE.
2. **Nonlinear stream**: standard CfC cell. n_tau=1.
3. **Sum combination** (LiNo spirit): h = h_lin + h_nl. Element-wise.
4. **Concat control**: lino_concat doubles the input dim to the head.
5. **NaN handling**: zero-fill input.
6. **Preserves CfC**: h goes through the standard CfC update. The
   linear stream is ADDITIVE (parallel), not a replacement.
7. **Per-layer**: each layer has its own linear stream.

## 3. Bench results (24 cells: 4 conds × 3 datasets × 2 seeds, 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** (baseline) | 0.0275±0.0028 | **0.1327±0.0183** | 0.1051±0.0029 | 2545 |
| **lino_sum** | **0.0225±0.0058** | 0.1333±0.0122 | 0.1054±0.0028 | 2865 |
| **lino_concat** | **0.0220±0.0002** | 0.2496±0.0659 | **0.1042±0.0025** | 3905 |
| lino_lin_only | 0.1492±0.0095 | 0.4441±0.0006 | 0.1199±0.0029 | 337 |

**Headline numbers (× change vs baseline)**:

- **lino_sum**: sin **-18%**, structured 0%, random 0%
- **lino_concat**: sin **-20%**, structured **+88% WORSE**, random -1%
- **lino_lin_only**: sin **+442% WORSE**, structured **+235% WORSE**,
  random +14% (sanity check confirms CfC matters)

## 4. Why LiNo-CfC helps sin and breaks on structured

### 4.1 Sin (lino_sum -18%, lino_concat -20%)

Sin is highly smooth and periodic. Locally, sin has a strong linear
trend (constant derivative over short windows). The linear stream
captures this trend directly without needing the CfC to learn it.
Result: linear + CfC beats CfC alone by 18-20%.

### 4.2 Structured (lino_sum neutral, lino_concat +88% WORSE)

Structured has a regime change at t=T/2 (sin → sin(2t)). The linear
stream cannot represent a jump discontinuity — it's literally a
linear projection. So:

- **lino_sum**: linear stream contributes nothing useful at the
  regime boundary; CfC ignores the misleading linear contribution.
  Result: NEUTRAL.
- **lino_concat**: linear stream is concatenated with CfC, doubling
  the input dim. The model now has to learn to "ignore" the linear
  stream's misleading info at the boundary. Result: BROKEN (+88%).

### 4.3 Random (all neutral)

Random is cumulative noise. Linear projection of noise is still
noise — it doesn't add information but doesn't destroy it either.
The CfC is still doing all the real work. Result: NEUTRAL across
all LiNo variants.

### 4.4 Lin-only (sanity check, all WORSE)

Pure linear projection (no CfC) fails everywhere:
- sin +442% (linear can't capture nonlinear periodic patterns)
- structured +235% (linear can't represent regime changes)
- random +14% (linear is essentially a moving average, can't track
  drift)

This confirms the **CfC is doing the heavy lifting**; the linear
stream is supplementary at best.

## 5. Why this differs from TCC 149 (target-dep 8th)

Both rounds add a parallel context stream to the CfC:

- **TCC 149**: parallel 1D temporal conv. K=3 wins on sin, K=7 wins
  on structured.
- **LiNo 150**: parallel linear projection. Wins on sin only.

The conv in TCC has a **receptive field** — K=3 sees 3 steps, K=7
sees 7 steps. This allows it to "anticipate" regime changes by
seeing mixed slow/fast signals BEFORE the boundary. The linear
projection has **no receptive field** — it only sees x_t, not its
neighbors. So:

- TCC conv can detect "approaching regime change" via the windowed
  mean. Linear cannot.
- TCC conv provides smoothing that's useful for noise AND regime
  detection. Linear projection is just a per-step projection.

**Conv > Linear** for parallel context streams on non-trivial data.

## 6. Why this differs from DLinear (Zeng 2022 AAAI)

DLinear uses linear projection + moving-average decomposition:

- DLinear decomp = linear(moving_avg(x)) + linear(x - moving_avg(x))
- LiNo-CfC: linear(x) + CfC(x)

Key difference: DLinear uses **two linear projections** (one for
trend, one for residual), while LiNo-CfC uses **one linear + one
nonlinear** (CfC). CfC has much more capacity than linear, so the
comparison is not apples-to-apples.

DLinear claims that simple linear models outperform transformers on
many TS tasks. Our lino_sum result confirms the trend component IS
useful (sin -18%) but the linear stream alone (lino_lin_only) is
insufficient for any nontrivial task.

## 7. NEW INSIGHTS

1. **Linear projection is a valid parallel context stream** for
   periodic data, providing a modest -18% to -20% improvement.
2. **Linear projection is NOT useful for regime-change data** — the
   stream cannot represent jump discontinuities.
3. **Concat is dangerous when streams are heterogeneous** —
   lino_concat breaks on structured (+88%). Sum is safer because
   the linear stream is additive and can be ignored by CfC.
4. **Conv > Linear** for parallel context: TCC 149 wins more
   broadly (sin -33%, structured -34%) than LiNo 150 (sin -18%
   only).
5. **Pattern reinforced (13 + 9 + 20 = 42 mechanism classes)**:
   - 13 winners preserve recurrent step + add useful structure
   - **9 target-dep**: input-side processing, bidi, SCRN, Time-Decay,
     TCC, **LiNo (this round)**
   - 20 negatives: per-step mods, alternatives, regularizers,
     bottlenecks, redundant info, replacements, long-α SCRN,
     Clockwork partition

**NEW RULE**: LiNo-CfC (linear projection + CfC, summed) is best
for smooth periodic data (-18% on sin). AVOID concat mode (breaks
on regime-change data). AVOID linear-only (loses everywhere).

## 8. The 91-150 audit: 42 mechanism classes

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
| **Time-Decay CfC (γ=0.5)** | 148 | **TARGET-DEPENDENT (7th)** |
| **TCC-CfC (K=3/5/7)** | 149 | **TARGET-DEPENDENT (8th)** |
| **LiNo-CfC (sum/concat)** | **150** | **TARGET-DEPENDENT (9th)** |
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
| LiNo (concat mode on structured) | **150** | NEGATIVE (21st — sub-verdict) |
| LiNo (lin_only) | **150** | NEGATIVE (sanity check) |
| Recursion depth (MoR) | 126 | NEGATIVE |
| 2nd-order oscillator | 128 | NEGATIVE |
| Multi-timescale ELM | 129 | NEGATIVE |
| Multi-rate MoE + dual attn | 130 | NEGATIVE |
| HGRN | 131 | NEGATIVE |
| AntisymmetricRNN | 132 | NEGATIVE |
| Hebbian Fast Weights | 133 | NEGATIVE |

**Pattern reinforced (13 + 9 + 20 = 42 tests)**:

- 13 winners preserve recurrent step + add useful structure
- **9 target-dep**: input-side processing, bidi, SCRN, Time-Decay,
  TCC, **LiNo (this round)**
- 20 negatives: per-step mods, alternatives, regularizers,
  bottlenecks, redundant info, replacements, long-α SCRN,
  Clockwork partition, LiNo (concat mode)

## 9. Recommendation

**LiNo-CfC is the 9th TARGET-DEPENDENT in the 91-150 audit.**

- **DO use LiNo-CfC sum mode for PERIODIC data** (sin-like). -18%
  on sin.
- **DO use sum mode for regime-change data** (no win, but no loss).
- **DO NOT use LiNo-CfC concat mode** for regime-change data. +88%
  on structured.
- **DO NOT use linear-only** (lino_lin_only loses everywhere).
- **Production recipe**:
  - Periodic → lino_sum (-18%)
  - Regime-change → lino_sum (neutral, safe)
  - Noisy → lino_sum (neutral, safe)
  - AVOID lino_concat for regime-change (breaks)
  - AVOID lino_lin_only (insufficient)

## 10. Critical implementation details

1. **Linear stream**: nn.Linear(input_size, hidden_size) per layer.
2. **Sum combination**: h = h_lin + h_nl (element-wise).
3. **NaN handling**: zero-fill input. Both linear and nonlinear
   streams handle NaN.
4. **Concat mode**: lino_concat doubles the head input dim,
   allowing the head to "mix" linear and CfC streams more flexibly.
   This is also its failure mode — the model can't learn to ignore
   the linear stream on regime-change data.
5. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements.

## 11. Files

- `lnn/core/lino_cfc.py` (~190 lines)
- `tests/test_lino_cfc.py` (20 tests, all pass)
- `scripts/bench_lino_cfc.py` (24-cell bench)
- `results/bench_lino_cfc.json`
- `docs/prds/2026-06-15-lnn-round-150-a-lino-cfc.md`
- `docs/research/2026-06-15_lino_cfc_report.md`
