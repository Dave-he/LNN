# PRD #10-112 — Linear-Nonlinear CfC (LiNo-CfC) (Round 150)

**Date**: 2026-06-15
**Round**: 150
**Verdict target**: TARGET-DEPENDENT (9th) or STRICTLY POSITIVE (14th) or NEGATIVE (21st)

## 1. Motivation

The 91-149 audit shows 8 target-dep winners that follow a clear
pattern: **preserve x, add parallel context**.

Recent work from PKU/HK PolyU (LiNo framework, Jan 2025) and
DLinear (Zeng et al. 2022 AAAI) suggest that **many time series
have a strong linear trend component** that can be captured by a
simple linear projection, while the nonlinear residual requires
a more sophisticated model. Combining both gives complementary
strengths.

Round 150 tests **Linear-Nonlinear CfC (LiNo-CfC)** — a parallel
linear stream + nonlinear CfC stream, summed at the output::

    # Linear stream: per-step linear projection (no recurrence)
    h_lin = Linear(x)  # x @ W_lin + b_lin, shape [B, T, hidden_size]

    # Nonlinear stream: standard CfC
    h_nl = CfCNetwork(x)  # [B, T, hidden_size]

    # Combine: sum (LiNo spirit)
    h = h_lin + h_nl

This is **different from**:
- **Conv preprocessing 137 (target-dep)**: 137 REPLACES x with
  the conv output. LiNo PRESERVES both linear and nonlinear streams.
- **TCC 149 (target-dep)**: TCC concats x with conv. LiNo sums
  linear projection with CfC.
- **Gated Input Skip 134 (strictly positive 13th)**: GIS is a
  single-step skip. LiNo is a parallel stream architecture.
- **Bidirectional CfC 144 (target-dep 5th)**: bidi processes the
  same input forward + backward. LiNo processes the same input
  with two different model classes (linear vs nonlinear).
- **DLinear (Zeng 2022)**: DLinear uses a linear projection +
  moving average decomp. LiNo uses a linear projection + CfC.

## 2. Mechanism

Standard CfC: `h_t = CfCCell(x_t, h_{t-1})`.

LiNo-CfC: parallel linear + nonlinear streams, summed::

    # Linear stream: per-step linear projection
    # W_lin: (input_size, hidden_size), b_lin: (hidden_size,)
    h_lin = x @ W_lin + b_lin  # [B, T, hidden_size]

    # Nonlinear stream: standard CfC
    h_nl = CfCStackedNetwork(x)  # [B, T, hidden_size]

    # Combine: sum
    h = h_lin + h_nl  # [B, T, hidden_size]

This is a **structural addition**: it preserves the recurrent step
(the CfC stream), preserves x (the linear stream reads x directly),
and adds a parallel linear context stream.

## 3. Hypotheses

- **H1** (Sin data): linear stream should help with smooth
  periodic data (sin has linear trends locally). **EXPECTED: positive.**
- **H2** (Structured data): linear stream should help with
  regime-change data (linear extrapolation + regime detection).
  **EXPECTED: positive.**
- **H3** (Random data): linear stream should hurt noisy data
  (linear projection of noise is still noise). **EXPECTED: negative.**
- **H4** (Concat vs sum): sum is LiNo's original formulation.
  Concat would give more capacity but more params. **EXPECTED:
  sum is sufficient.**

## 4. Implementation

`lnn/core/lino_cfc.py` (~120 lines) — `LinearNonlinearCfCCell` +
`LinearNonlinearCfCStackedNetwork`.

Key design choices:

1. **Linear stream**: a single nn.Linear (input_size → hidden_size).
   Applied to x at every step. No recurrence.
2. **Nonlinear stream**: standard CfC cell. n_tau=1 (default).
3. **Sum combination**: h = h_lin + h_nl. Element-wise addition.
4. **NaN handling**: zero-fill input.
5. **Preserves CfC**: h goes through the standard CfC update.
   The linear stream is ADDITIVE (parallel).
6. **Per-layer**: each layer has its own linear stream.

## 5. Bench

24 cells: 4 conds × 3 datasets × 2 seeds, 30 epochs:
- cfc (baseline)
- lino_sum (linear + CfC, summed)
- lino_concat (linear + CfC, concatenated — control)
- lino_lin_only (just linear, no CfC — sanity check)

## 6. Why this might win (mechanism reasoning)

The audit pattern: input-side processing that PRESERVES x wins.
LiNo preserves x (the linear stream reads x directly), preserves
h (the CfC stream is the recurrent step), and adds a parallel
linear stream.

The risk: the linear stream's projection has a different number
of parameters than the CfC stream's first layer. Sum requires
matching dimensions. If the linear stream is too "weak" (too few
params), the CfC dominates. If too "strong" (too many params),
the model is essentially linear.

## 7. Critical implementation details

1. **Linear stream**: nn.Linear(input_size, hidden_size) per layer.
2. **Sum combination**: h = h_lin + h_nl (element-wise).
3. **NaN handling**: zero-fill input. Both linear and nonlinear
   streams handle NaN.
4. **Concat control**: lino_concat doubles the input dim to the
   head, more capacity.
5. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements.
