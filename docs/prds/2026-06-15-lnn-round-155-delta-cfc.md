# PRD #10-117 — Round 155: DELTA-CfC (Hidden State Delta Augmentation)

**Date**: 2026-06-15
**Round**: 155
**Audit context**: 91-154 audit, 14 strictly positive + 10
target-dep + 23 negatives = 47 mechanism classes.

## Hypothesis

The CfC's closed-form solution produces h_t at each step, but
loses explicit information about **how much h has changed**
since the previous step. The temporal derivative Δh_t = h_t -
h_{t-1} carries information about:
- regime switches (large Δh)
- noise level (small Δh ≈ noise)
- confidence in h (small Δh → stable)

By **augmenting the hidden state output with Δh**, the next
layer / downstream head can use this information.

This is **structurally different from**:
- **DiffCfC (round 145)**: input-side deltas Δx_t, Δ²x_t.
  Different mechanism dimension.
- **TDSA 152 / MSDC 151 / TCC 149**: parallel context, concat
  with x.
- **FiLM 153**: γ, β modulation.
- **SCRN 146 / Time-Decay 148 / Clockwork 147**: alternative
  memory structures.

## Mechanism

```
h_t       = CfC(x_t, h_{t-1})            # standard
delta_t   = h_t - h_{t-1}                # temporal derivative
h_aug_t   = concat([h_t, delta_t])       # 2*hidden_size output
```

Variants:
1. **delta_concat**: append Δh to h, doubling the hidden dim.
2. **delta_proj**: project the concat back to hidden_size via
   a Linear layer (preserves param count).
3. **delta_gated**: h_out = (1-α) * h + α * delta, where α is
   learned scalar per dimension.
4. **delta_concat_input**: feed Δh to the NEXT layer's input
   (no change to current layer's output dim).

## Bench plan

32 cells: 5 conds × 3 datasets × 2 seeds, 30 epochs:

| cond            | mechanism              |
|-----------------|------------------------|
| cfc              | baseline (Tanh+Tanh)   |
| delta_concat     | h + Δh, doubled dim    |
| delta_proj       | h + Δh, projected back |
| delta_gated      | (1-α)·h + α·Δh         |
| delta_concat_input| next layer sees Δh     |

3 datasets: sin_irr, structured_irr, random_irr.

## Expected outcomes

- **If Δh helps**: STRICTLY POSITIVE (15th winner) — explicit
  change signal aids regime detection.
- **If Δh hurts**: NEGATIVE (24th) — CfC's closed-form already
  encodes temporal info via τ.
- **If mixed**: TARGET-DEPENDENT (11th).

## Risk

Δh has 0 mean over time (since h_t is a stable process), so it
may be uninformative and just add noise. The projection variant
(linear) might learn to discard it.

## Files to create

- `lnn/core/delta_cfc.py` (~220 lines)
- `tests/test_delta_cfc.py` (~25 tests)
- `scripts/bench_delta_cfc.py` (~250 lines, 30-cell bench)
- `results/bench_delta_cfc.json`
- `docs/research/2026-06-15_delta_cfc_report.md`
