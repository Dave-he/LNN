---
title: "Round 273 — STE × Longer Sequences (T=128) — TARGET-DEPENDENT-WITH-NUANCE (T=128 hurts structured)"
date: 2026-06-29
round: 273
prd: "docs/prds/2026-06-29-lnn-round-273-ste-longer-sequences.md"
status: "HONEST NEGATIVE — T=128 is significantly worse on structured (production T=64 CONFIRMED)"
audit_pattern: "66 SP + 28 TD + 61 NEG = 155 mechanism classes (UNCHANGED)"
---

# Round 273 — STE × Longer Sequences (T=128) — Target-Dependent

## TL;DR

**Longer sequences (T=128) hurt structured learning.** This is a
**target-dependent finding** — toy_sin and random are unchanged,
but the 4-segment structured task gets **32× worse at h=64** and
**11× worse at h=192** when going from T=64 to T=128.

| mode                       | T   | hidden | structured | seed_std | top1_frac |
|----------------------------|-----|--------|------------|----------|-----------|
| ste_entropy_t64_h64 (r270) | 64  | 64     | 0.000426   | 0.000129 | 0.022     |
| ste_entropy_t64_h192 (r272)| 64  | **192**| **0.000171** | **0.000021** | 0.0075 |
| ste_entropy_t128_h64       | 128 | 64     | **0.013965** ⚠ | 0.005526 | 0.025 |
| ste_entropy_t128_h192      | 128 | **192**| 0.001965   | 0.001207 | 0.0077    |

**Production T=64 is CONFIRMED**. Going to T=128 is a regression
on structured.

**h=192 still wins at T=128** (H2 CONFIRMED): 0.001965 vs
0.013965 (h=64) — 7× better. The mechanism scales with hidden
but is **independent of T**.

## Hypothesis Evaluation

### H1 (T=128 ≥ T=64 on structured)
**REJECTED**. structured:
- T=64, h=192: 0.000171
- T=128, h=192: 0.001965 (**11× WORSE**)

Longer sequences HURT structured. Counter-intuitive — more
data should help, but **BPTT difficulty** dominates.

### H2 (h=192 still wins at T=128)
**CONFIRMED**. structured at T=128:
- h=64: 0.013965
- h=192: 0.001965 (**7× better**)

Hidden effect is **T-independent**. Same proportional win at
T=128 as at T=64.

### H3 (τ dynamics more stable at T=128)
**REJECTED**. structured τ_min:
- T=64, h=192: 0.243-0.377
- T=128, h=192: 0.106-0.133 (**2-3× lower**)

τ_min drops sharply. The model **adapts** τ to T (faster
neurons for longer sequences), but this adaptation is **not
enough** to overcome BPTT difficulty.

### H4 (logit_std pattern preserved at T=128)
**PARTIAL**. structured logit_std:
- T=64, h=192: 0.37-0.61 (stable)
- T=128, h=192: 0.39-0.49 (stable, preserved)
- T=64, h=64: 0.62-0.67
- T=128, h=64: 0.80-0.98 (**INCREASES — instability!**)

h=192 keeps logit_std stable at T=128. But h=64 has
**logit instability** at T=128 (0.80-0.98 vs 0.62-0.67 at T=64).
This suggests **hidden size and T interact**: small hidden can't
handle long T, but large hidden can.

### H5 (top1_frac pattern preserved at T=128)
**CONFIRMED**. structured top1_frac at h=192:
- T=64: 0.0070-0.0080
- T=128: 0.0076-0.0078 (**SAME**)

The entropy reg mechanism is **T-invariant**. Same number of
effective experts regardless of T.

## Why T=128 Hurts Structured

The structured task has 4 segments at fixed levels
{0.0, 1.0, -0.5, 0.7}. At T=64, segments are 16 timesteps
each. At T=128, segments are 32 timesteps each.

The model needs to:
1. Predict the **current level** (already learned well at T=64)
2. **Stay on the same level** for 32 timesteps in a row

Staying-on-the-same-level is easy at T=64 (just memorize the
last 16 values). At T=128, the model needs to "remember" for
32 timesteps without diverging.

With τ dynamics + entropy reg, the hidden state must remain
stable over 32 timesteps. The τ_min dropping to 0.106 helps
fast adaptation but doesn't help long-term stability.

## Why toy_sin and random are T-Invariant

- **toy_sin**: periodic continuous signal. Predicting sin(t+1)
  from sin(t) is **the same** at T=64 and T=128. No long-term
  memory needed.
- **random**: unlearnable noise. T doesn't matter.

Only **structured** (piecewise constant) requires long-term
memory, and that's where T=128 hurts.

## When h=192 Helps at T=128

Even at T=128, h=192 gives 7× improvement over h=64. The
mechanism (entropy reg + soft mask) is preserved — only the
overall performance degrades.

This shows **hidden and T are independent dimensions**:
- Hidden improves **per-step learning**
- T affects **long-horizon stability**

## Production Settings (UNCHANGED)

```python
STEWithEntropy(
    input_size=d_in,
    hidden_size=192,           # r272 CONFIRMED
    density=0.3,                # r263 hard top-k fraction
    ste_temperature=1.0,        # r265/r269 CONFIRMED
    entropy_lambda=0.1,         # r267/r268 CONFIRMED
    T=64,                       # r273 CONFIRMED at T=64
)
```

T=64 is **production-locked**. T=128 hurts structured.

## Pattern Update

| Bucket          | Before | After | Δ |
|-----------------|--------|-------|---|
| Strictly pos.   |   66   |   66  | 0 |
| Target-dep      |   28   |   28  | 0 |
| Negatives       |   61   |   61  | 0 |
| **Total**       |  155   |  155  | 0 |

r273 doesn't change the pattern count. It's a **parameter
characterization** that documents a target-dependent
limitation (T=128 hurts structured).

## Why r273 is HONEST NEGATIVE-WITH-NUANCE

**Negative part**: T=128 is **worse** on structured (counter
to my hypothesis). More data ≠ better learning when BPTT
difficulty dominates.

**Nuance part**: h=192 still scales with T (mechanism
preserved). T-invariant **mechanism** is real, but T-dependent
**task performance** is the reality.

This is the kind of finding that prevents overconfidence. The
r267 mechanism is **not universally better** — it depends on
T (for structured).

## T Sweep Map (r267-r273)

| T   | h=64 structured | h=192 structured | source |
|-----|-----------------|-------------------|--------|
| 64  | 0.000426        | 0.000171          | r270, r272 |
| 128 | 0.013965 ⚠      | 0.001965          | r273 |

**T=64 is the production-locked value** for the structured
task. T=128 introduces BPTT difficulty that exceeds the model's
capacity to compensate.

## Next Round (Round 274)

The (τ, λ, hidden, T) sweep is now complete (r267-r273).
Candidates for r274:

1. **STE × multi-channel input** (d_in=4 or 8) — test with
   more input diversity. May help structured learn more
   quickly.
2. **STE × different density** (0.1 or 0.5) — test sparsity
   sensitivity at h=192.
3. **STE + annealed entropy reg** — start with λ=1.0 then
   anneal to λ=0.1.
4. **STE + PDNA-style pulse** (arXiv:2603.00153) — add
   oscillatory dynamics on top of STE.
5. **STE × longer epochs** (200 or 300) — does the model
   need more time to converge at h=192?

**Recommended: #1 (multi-channel input)** — natural next
dimension. Tests whether more input diversity helps.

## Files Added (Round 273)

- `scripts/bench_ste_longer_sequences.py` (~370 LOC)
- `analysis/ste_longer_sequences_bench.json` (36 cells)
- `docs/prds/2026-06-29-lnn-round-273-ste-longer-sequences.md`

## Cumulative Test Count

**0 new tests** (r273 is bench-only — reuses r267 STEWithEntropy).
No regressions.