# Round 160 — MultiBeta-H-CfC (Multi-Scale Hidden State EMA) Report

**Date**: 2026-06-15
**Round**: 160
**PRD**: #10-122
**Audit context (91-160)**: 22 strictly positive + 17 target-dep +
30 negatives = 69 mechanism classes.
**Verdict**: **1 NEW STRICTLY POSITIVE WINNER (mbh_diff_2)**
with **BEST sin EVER (-32%)** in the 91-160 audit, beating
round 159's eh_diff (-16% sin) by 16pp.

## What was tested

The CROSS of round 158 + round 159: apply K=2 or K=3 multi-scale
EMA pattern to the **hidden state h** (vs input x in round 158).
This tests whether multi-scale h-side EMA strictly improves over
single-scale h-side EMA.

Mechanism::

    # At step t, for each k in 0..K-1:
    ema_h_k,t = beta_k * ema_h_k,t-1 + (1 - beta_k) * h_t
    aug_h_t = f_concat(h_t, ema_h_1,t, ..., ema_h_K,t)  # variants

K=2: β ∈ {0.7, 0.95} (short, long)
K=3: β ∈ {0.5, 0.9, 0.99} (short, medium, long)

Variants (mirror round 158):
- mbh_diff_2:   aug_h = [h_t, ema_h_1-h_t, ema_h_2-h_t] (3H input)
- mbh_concat_2: aug_h = [h_t, ema_h_1, ema_h_2] (3H input)
- mbh_diff_3:   aug_h = [h_t, ema_h_1-h_t, ema_h_2-h_t, ema_h_3-h_t] (4H input)
- mbh_concat_3: aug_h = [h_t, ema_h_1, ema_h_2, ema_h_3] (4H input)

## Bench (30 cells: 5 conds × 3 datasets × 2 seeds, 30 epochs)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc** | 0.0275±0.0028 | 0.1327±0.0183 | 0.1051±0.0029 | 2545 |
| **mbh_diff_2** | **0.0186±0.0029 (-32%)** | **0.0556±0.0124 (-58%)** | **0.1031±0.0029 (-2%)** | 5617 |
| mbh_concat_2 | 0.0251±0.0053 (-9%) | 0.4225±0.0033 (+218%) | 0.1047±0.0011 (-0%) | 5617 |
| mbh_diff_3 | 0.0198±0.0026 (-28%) | 0.0664±0.0307 (-50%) | 0.1044±0.0022 (-1%) | 7153 |
| mbh_concat_3 | 0.0203±0.0017 (-26%) | 0.2897±0.1467 (+118%) | 0.1058±0.0005 (+1%) | 7153 |

## Headlines

### **mbh_diff_2 — 22nd STRICTLY POSITIVE WINNER**

- **Sin -32%** — **BEST sin EVER** in the 91-160 audit!
  Beats:
  - Round 159's eh_diff (-16%) by 16pp
  - Round 158's mb_concat_2 (-13%) by 19pp
  - Round 156's ema_diff (-11%) by 21pp
- Structured -58% (good but not as good as round 159's -77%)
- Random -2% (preserved)

### mbh_diff_3 — 17th TARGET-DEPENDENT

- Sin -28%
- Structured -50% (high variance ±31%)
- Random -1%
- K=3 with high variance — K=2 is the sweet spot for h-side

### mbh_concat_2 — 31st NEGATIVE

- Sin -9% (only marginal improvement)
- Structured **+218%** CATASTROPHIC
- Low-pass multi-scale on h is unstable

### mbh_concat_3 — 32nd NEGATIVE

- Sin -26%
- Structured **+118%** CATASTROPHIC (high variance ±147%)
- K=3 low-pass is even worse than K=2

## Why mbh_diff_2 achieves BEST EVER sin (-32%)

### 1. K=2 is the sweet spot for h-side multi-scale
Unlike x-side (round 158, K=3 best), h-side prefers K=2. With
K=2, the two EMAs (β=0.7 short, β=0.95 long) capture:
- Short-window smoothing: tracks recent h changes
- Long-window smoothing: provides regime-level reference

The diff signal `(ema_h_1 - h, ema_h_2 - h)` exposes BOTH:
- "How much h is deviating from short-term average" (transient)
- "How much h is deviating from long-term average" (regime)

For sinusoidal data, this gives the model explicit access to
**phase information** (the diff signal oscillates with the
sinusoid).

### 2. Sin is periodic with single dominant frequency
For sin, the optimal h-side processing is to track:
- Phase (current angle)
- Amplitude (current value)
- Smoothed reference (long-term)

K=2 diff EMA provides exactly this decomposition. K=3 introduces
redundancy (the 3rd EMA doesn't add new information for sin).

### 3. h-side multi-scale is better than h-side single-scale
mbh_diff_2 (sin -32%) STRICTLY OUTPERFORMS round 159's eh_diff
(sin -16%) on sin. The 2nd EMA (β=0.95) adds value over single
β=0.9 EMA.

### 4. Structured -58% is a regression from single-scale -77%
For STRUCTURED data (regime changes), single-scale h-side
EMA (round 159) is BETTER than multi-scale h-side EMA. Why?
- Multi-scale introduces 2-3 EMA states, each with their own
  "regime detection" logic
- The 3-EMA configuration is over-parameterized for simple
  regime detection
- For structured (only 1 regime change at t=16), the simpler
  single-EMA detects it cleanly; multi-EMA "averages out" the
  detection

## Cross-round progression (sin, EMA-based)

| Round | Mechanism | sin | structured | random |
|-------|-----------|-----|------------|--------|
| 156 | ema_diff | -11% | -42% | -1% |
| 157 | lb_diff | -11% | -63% | -1% |
| 158 | mb_concat_2 | -13% | -60% | -2% |
| 158 | mb_diff_3 | -5% | -65% | -2% |
| 159 | eh_diff | -16% | **-77%** | -2% |
| **160** | **mbh_diff_2** | **-32%** | -58% | -2% |

**mbh_diff_2 achieves the BEST sin EVER**. The cross-product of
multi-scale (round 158) + h-side (round 159) gives a different
result than either alone: BEST sin but WORSE structured than
the simpler h-side (round 159).

## Why K=2 wins for h-side, K=3 wins for x-side

### X-side (round 158): K=3 diff best for structured (-65%)
X-side has high-frequency noise + low-frequency content. K=3
EMAs at different cutoffs (0.5, 0.9, 0.99) capture:
- Short-term spikes (β=0.5)
- Medium-term patterns (β=0.9)
- Long-term regime (β=0.99)

### H-side (round 160): K=2 diff best for sin (-32%)
H-side is already smoothed (it's a recurrent state, not raw
input). Adding K=3 EMAs to h creates redundant signals:
- β=0.5 EMA: tracks raw h changes (no smoothing value)
- β=0.9 EMA: similar to original h (redundant)
- β=0.99 EMA: long-term regime

K=2 (β=0.7, 0.95) hits the sweet spot — enough to capture
multiple time-scales without redundancy.

## NEW INSIGHTS

1. **mbh_diff_2 is the 22nd STRICTLY POSITIVE winner** with
   **BEST sin EVER (-32%)** — first mechanism to break the
   -20% sin barrier.
2. **Multi-scale h-side prefers K=2** (vs x-side's K=3) —
   different optimal K for different signal types.
3. **mbh_diff_2 trades structured for sin** (structured -58%
   vs round 159's -77%, sin -32% vs -16%) — multi-scale helps
   sin but doesn't help structured.
4. **Low-pass multi-scale h is catastrophic** — concat mode
   with 2-3 EMAs loses high-frequency information critical
   for regime detection.
5. **K=2 high-pass > K=3 high-pass on h-side** — adding
   3rd EMA doesn't help, just adds variance.

**NEW RULE**: **Multi-scale h-side diff EMA prefers K=2 over
K=3 for sin data.** For max sin improvement, use mbh_diff_2
(best sin EVER -32%). For max structured improvement, use
single-scale h-side EMA (round 159's eh_diff, -77%).

## Pattern reinforced (22 + 17 + 30 = 69 mechanism classes)

- **22 strictly positive** (was 21): previous 21 + **mbh_diff_2 (this round)**
- **17 target-dep** (was 16): previous 16 + **mbh_diff_3 (this round)**
- **30 negatives** (was 28): +2 (mbh_concat_2, mbh_concat_3)

## Why this differs from prior mechanisms

- **MultiBeta 158** (19th, 20th positive): K=2/K=3 on input x.
  **MultiBeta-H 160**: K=2/K=3 on h. Different optimal K
  (h-side prefers K=2, x-side prefers K=3).
- **EMA-H 159** (21st positive): single β=0.9 on h.
  **MultiBeta-H 160**: K=2 on h. MultiBeta-H OUTPERFORMS
  EMA-H on sin (-32% vs -16%) but UNDERPERFORMS on structured
  (-58% vs -77%).
- **Multi-timescale τ 76** (7th winner): multi-timescale τ in
  CfC recurrence. **MultiBeta-H 160**: multi-timescale h-EMA.
  Different mechanism (parameter vs state).

## Critical implementation details

1. **K=2: β ∈ {0.7, 0.95}** (short, long).
2. **K=3: β ∈ {0.5, 0.9, 0.99}** (short, medium, long).
3. **2 modes (diff/concat)** — diff for high-pass, concat for
   low-pass.
4. **Per-layer cell input size** — layer 0 receives input_size,
   layer 1+ receives hidden_size.
5. **NaN handling at the cell** — `torch.nan_to_num` for x, h,
   and all EMAs.
6. **Pyright false positives** on `import torch` are pre-existing
   per standing requirements.

## Files

- `lnn/core/multi_beta_h_cfc.py` (~280 lines)
- `tests/test_multi_beta_h_cfc.py` (24 tests, all pass)
- `scripts/bench_multi_beta_h_cfc.py` (30-cell bench)
- `results/bench_multi_beta_h_cfc.json`
- `docs/prds/2026-06-15-lnn-round-160-multi-beta-h-cfc.md`
- `docs/research/2026-06-15_multi_beta_h_cfc_report.md`

## Conclusion

Round 160 is a **WIN for sin** (best ever -32%): mbh_diff_2
achieves the best sin ever in the 91-160 audit, breaking the
-20% sin barrier for the first time. The 22nd STRICTLY POSITIVE
winner.

The cross-product of multi-scale + h-side reveals a new
**interaction effect**: multi-scale helps sin (which is
periodic) but doesn't help structured (which is regime-change).
The optimal K also flips: K=2 for h-side, K=3 for x-side.

Next ideas:
1. **Stacked EMA-X + EMA-H**: combine input-side EMA (rounds
   156-158) with h-side EMA (rounds 159-160) → could be the
   best of both worlds.
2. **AdaptiveK-H-CfC**: dynamically choose K based on data
   characteristics.
3. **Per-feature β on h-side**: combine round 157 (per-feature
   learned β) with h-side (round 159).
