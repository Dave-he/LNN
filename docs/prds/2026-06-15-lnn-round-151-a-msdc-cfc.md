# PRD #10-113 — Multi-Scale Dilated Conv CfC (MSDC-CfC) (Round 151)

**Date**: 2026-06-15
**Round**: 151
**Verdict target**: TARGET-DEPENDENT (10th) or STRICTLY POSITIVE (14th) or NEGATIVE (21st)

## 1. Motivation

The 91-150 audit shows 9 target-dep winners that follow a clear
pattern: **preserve x, add parallel context**.

Rounds 149 (TCC) and 137 (Conv preprocessing) tested 1D conv as a
parallel context stream. TCC 149 used a **single kernel size K**
(3, 5, or 7). Round 137 used a **single K conv to REPLACE x**.

What's NOT been tested: **multi-scale parallel context** —
multiple 1D convs with different dilations running in parallel,
summed to a single context vector, then concatenated with x.

Multi-scale dilated convolutions (WaveNet/Oord 2016, TCN/Bai 2018,
Inception/Szegedy 2015) are a well-established mechanism for
capturing different temporal scales simultaneously:
- Dilation 1: local details
- Dilation 2: medium-range patterns
- Dilation 4: longer-range context

Round 151 tests **Multi-Scale Dilated Conv CfC (MSDC-CfC)** —
three parallel 1D convs with kernel=2, dilations 1/2/4, summed to
a context vector, concatenated with x as input to CfC::

    # Three parallel 1D convs with kernel=2, dilations 1/2/4
    c1 = Conv1D_d1(x_padded)  # [B, D, T]
    c2 = Conv1D_d2(x_padded)
    c3 = Conv1D_d4(x_padded)
    # Sum (or concat) to form context
    c = c1 + c2 + c3  # [B, D, T]
    # Concatenate with x
    aug_x = concat([x, c], dim=-1)  # [B, T, 2D]
    # Standard CfC with augmented input
    h = CfCCell(aug_x, h)

This is **different from**:
- **TCC 149 (target-dep 8th)**: TCC uses a single K, no dilation.
  MSDC uses three dilations, summed.
- **Conv preprocessing 137 (target-dep)**: 137 REPLACES x with conv.
  MSDC PRESERVES x (concats).
- **LiNo 150 (target-dep 9th)**: LiNo uses linear projection (no
  receptive field). MSDC uses conv (has receptive field).
- **WaveNet (Oord 2016)**: WaveNet stacks dilated convs SERIALLY
  with residual connections. MSDC runs them in PARALLEL and sums.

## 2. Mechanism

Standard CfC: `h_t = CfCCell(x_t, h_{t-1})`.

MSDC-CfC: parallel 1D dilated convs summed, then concat with x::

    # Three parallel 1D convs (kernel=2, dilations 1/2/4)
    c1 = Conv1D(d=1)(x_padded)  # receptive field 1
    c2 = Conv1D(d=2)(x_padded)  # receptive field 3
    c3 = Conv1D(d=4)(x_padded)  # receptive field 5
    # Sum (or concat) to form context
    c = c1 + c2 + c3  # [B, T, D]
    # Concatenate with x
    aug_x = concat([x, c], dim=-1)  # [B, T, 2D]
    # Standard CfC with augmented input
    h = CfCCell(aug_x, h)

This is a **structural addition** (preserves x, adds multi-scale
context).

## 3. Hypotheses

- **H1** (Sin data): multi-scale conv should help periodic data
  (different scales capture different harmonics). **EXPECTED:
  positive.**
- **H2** (Structured data): multi-scale conv should help
  regime-change data (long-range dilation 4 sees boundary). **EXPECTED:
  positive.**
- **H3** (Random data): conv smoothing destroys noise. **EXPECTED:
  negative.**
- **H4** (Sum vs concat): sum is more compact, concat gives more
  capacity. **EXPECTED: similar.**

## 4. Implementation

`lnn/core/msdc_cfc.py` (~150 lines) — `MultiScaleDilatedConvCfCCell` +
`MultiScaleDilatedConvCfCStackedNetwork`.

Key design choices:

1. **Three parallel 1D convs**: kernel_size=2, dilations 1/2/4.
2. **Causal padding**: left-pad with (dilation, 0).
3. **Sum to context**: c = c1 + c2 + c3 (saves params vs concat).
4. **Concat with x**: aug_x = concat([x, c], dim=-1), shape [B, T, 2D].
5. **Standard CfC**: takes aug_x as input, h is unchanged.
6. **NaN handling**: zero-fill input.

## 5. Bench

24 cells: 4 conds × 3 datasets × 2 seeds, 30 epochs:
- cfc (baseline)
- msdc_sum (3 dilations summed, then concat with x)
- msdc_concat (3 dilations concatenated with x → 4D)
- msdc_single (single dilation d=4 only, control)

## 6. Why this might win (mechanism reasoning)

The audit pattern: input-side processing that PRESERVES x wins.
MSDC preserves x (concat), preserves h (CfC stream is recurrent
step), and adds a multi-scale context stream.

TCC 149 used single K. MSDC uses multi-scale. Multi-scale should
generalize TCC's wins:
- TCC K=3 wins on sin: MSDC's d=1 + d=2 covers K=3's receptive
  field
- TCC K=7 wins on structured: MSDC's d=4 has receptive field 5
  (close to K=7's 7)
- TCC all K bad on random: MSDC likely also bad

Risk: three convs add params and may overfit on T=32 data. Sum
combination reduces param count.

## 7. Critical implementation details

1. **Causal padding**: `pad = (dilation, 0)` ensures position t
   sees only x_{t-dilation}.
2. **Sum combination**: c1 + c2 + c3 reduces params vs concat.
3. **NaN handling**: zero-fill input before conv.
4. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements.
