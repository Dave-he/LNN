# Round 212 — Research Report & Idea

**Date**: 2026-06-16
**Round**: 212
**Audit context (91-211)**: 49 strictly positive + 27 target-dep
+ 58 negatives = 134 mechanism classes.

## TL;DR

Two consecutive SPs from the **3-scale spectral** axis (r210
simple average, r211 learned per-scale weights). Both r210
(simple avg) and r211 (adaptive) win all 3 datasets. The 3rd
scale (quarter FFT) is the breakthrough — it captures coarse
regime structure invisible to single-scale spectral.

## What's the next idea?

**4-scale spectral gating** — push scale count from 3 to 4,
adding an "eighth" FFT (hidden_size//16+1 frequencies).
The reasoning:
- 1 scale: r200 (TD, struct hurt)
- 2 scales: r209 (TD, struct hurt +19.5%)
- 3 scales: r210/r211 (SP, struct -62% to -70%)
- 4 scales: r212 (TBD — does pushing to 4 scales help or hurt?)

The hypothesis: r209 2-scale had struct regression because the
2nd scale (half) doesn't capture regime structure; r210 3-scale
fixed this by adding quarter. Pushing to 4 adds another ultra-
coarse scale for very long-range dependencies.

## Research question

Does a 4th scale (eighth FFT) provide additional benefit, or
does it just dilute the contributions of the 3 effective scales?

## Implementation plan

1. **File**: `lnn/core/learned_beta_ps_ln_khlfft_4spectral_cfc.py`
2. **Cell**: `FourScaleSpectralCfCCell` with 4 scale masks
   (full, half, quarter, eighth)
3. **Stacked**: `FourScaleSpectralCfCStackedNetwork` with
   Kh=[5,3,2] ladder (same as r210)
4. **Factory**: `make_lbps_lnkhlfft_4spectral_5_3_2`
5. **Tests**: 10 unit tests (forward, NaN, gradient, stacked,
   determinism, smoke_learns_sin, long_seq, zero_input,
   weights_sum, weights_bounded)
6. **Bench**: 18 cells (3 conds × 3 datasets × 2 seeds × 30 epochs)
   comparing cf baseline vs r210 vs r212
7. **Expected outcome**: SP if all 3 improve; TD if 1 hurts;
   NEG if 2+ hurt

## Pre-mortem: why 4-scale might fail

1. **Hidden=16 may be too small for 4 effective scales** —
   scale 4 only has 2 frequencies for hidden=16. This is
   borderline trivial.
2. **Diminishing returns** — 3 scales already capture the
   useful structure.
3. **Optimization difficulty** — 4 mask linear layers may
   be harder to train than 3.

## What we'll measure

- Per-dataset test MSE for 3 conds × 3 datasets × 2 seeds
- Stability (no NaN, no explosion)
- Training time relative to r210

## Reference

CfC with FFT: r200 (Sonnet 2026 FNO-style spectral gating on CfC)
3-scale: r210 (SP 48th)
3-scale adaptive: r211 (SP 49th)
