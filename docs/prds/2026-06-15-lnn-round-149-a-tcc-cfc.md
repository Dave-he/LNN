# PRD #10-111 — Temporal Conv Concat CfC (TCC-CfC) (Round 149)

**Date**: 2026-06-15
**Round**: 149
**Verdict target**: TARGET-DEPENDENT (8th) or STRICTLY POSITIVE (14th) or NEGATIVE (21st)

## 1. Motivation

The 91-148 audit shows 7 target-dep winners that follow a clear
pattern: **preserve x, add parallel context**:

- LN 135: input LN (preserves x, normalizes)
- Conv 137: input conv (preserves x, but REPLACES with conv output — target-dep)
- GLU+skip 139: input GLU
- decoupled/IndRNN 143: decoupled h
- bidi_concat 144: bidi h concat
- SCRN α=0.5 146: parallel slow context
- Time-Decay γ=0.5 148: multiplicative decay on h

Round 149 tests a structurally different idea: **parallel temporal
convolution stream concatenated with x** (TCC-CfC).

Standard CfC: `h_t = CfCCell(x_t, h_{t-1})`.

TCC-CfC: at each step t, a 1D convolution over the time axis
produces a parallel context vector c_t. The augmented input is
`concat(x_t, c_t)`. The CfC cell sees this enriched input::

    # 1D conv over the time axis
    c = Conv1D(x, kernel_size=K)  # shape [B, T, D]
    # Concatenate with x
    aug_x = concat(x, c, dim=-1)  # shape [B, T, 2D]
    # Standard CfC with augmented input
    h = CfCCell(aug_x, h)

This is **different from**:
- **Conv preprocessing 137 (target-dep)**: 137 REPLACES x with
  the conv output. TCC PRESERVES x and adds c as a parallel stream.
- **QuITE 102 (strictly positive)**: QuITE uses attention to
  embed irregular TS. TCC uses simple 1D conv.
- **Gated Input Skip 134 (strictly positive 13th)**: GIS is a
  single-step skip (skip = 1). TCC uses multi-step kernel (K=3, 5, etc.).

## 2. Mechanism

Standard CfC: `h_t = CfCCell(x_t, h_{t-1})`.

TCC-CfC adds a parallel convolution stream::

    # Pre-compute 1D conv over the time axis
    # pad to maintain sequence length
    x_padded = pad(x, (K-1, 0))  # left-pad for causal conv
    c = Conv1D(x_padded)  # [B, T, D]  -- output of conv at each t

    # Concatenate with x for the input
    aug_x = concat([x, c], dim=-1)  # [B, T, 2D]

    # Standard CfC step
    h_t = CfCCell(aug_x_t, h_{t-1})

This is a **structural addition**: it preserves both x and h, and
adds a parallel context stream.

## 3. Hypotheses

- **H1** (Sin data): conv kernel should help periodic data
  (sin has local smoothness, conv captures this). **EXPECTED: positive.**
- **H2** (Structured data): conv kernel should help regime-change
  data (local context + regime marker). **EXPECTED: neutral or positive.**
- **H3** (Random data): conv kernel should hurt noisy data
  (averaging kills high-freq noise info). **EXPECTED: negative.**
- **H4** (Different K): K=3 vs K=5 vs K=7. Larger K = more
  smoothing. **EXPECTED: K=3 sweet spot.**

## 4. Implementation

`lnn/core/tcc_cfc.py` (~120 lines) — `TemporalConvConcatCfCCell` +
`TemporalConvConcatCfCStackedNetwork`.

Key design choices:

1. **Causal left-padding**: pad only on the left (K-1, 0) so the
   conv at step t only sees x_{t-K+1..t}. No future leakage.
2. **Single 1D conv layer**: kernel size K, stride 1, no bias on
   the conv (or with bias — implementation choice).
3. **Concat with x**: aug_x = concat(x, c) at each step. The conv
   output goes through CfC alongside x, not replacing it.
4. **NaN handling**: zero-fill input per step.
5. **Preserves CfC**: h goes through the standard CfC update with
   augmented input. This is a STRUCTURAL addition (preserves the
   recurrent step), not a per-step modification.

## 5. Bench

24 cells: 4 conds × 3 datasets × 2 seeds, 30 epochs:
- cfc (baseline)
- tcc_k3 (K=3 conv)
- tcc_k5 (K=5 conv)
- tcc_k7 (K=7 conv)

## 6. Why this might win (mechanism reasoning)

The audit pattern: input-side processing that PRESERVES x wins.
TCC preserves x (just adds a parallel conv stream to the input
dimension). The conv provides:
- Local context: helps with smooth data (sin_irr)
- Receptive field: K=3 means the cell sees x_{t-2..t} as additional
  input, not just x_t
- Parallel stream: doesn't replace x, doesn't modify h

The risk: TCC doubles the input dimension, so the CfC cell has
2D input instead of D. This means more parameters in the first
linear layer. The cell might overfit on small data.

## 7. Critical implementation details

1. **Causal conv**: left-pad with (K-1, 0) so position t sees only
   x_{t-K+1..t}. NO future leakage.
2. **Single conv layer**: simple, no nested convs.
3. **NaN handling**: zero-fill input before conv. The conv will
   produce some output even with NaN inputs (after zero-fill).
4. **Concat dim**: dim=-1, so aug_x has shape [B, T, 2D].
5. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements.
