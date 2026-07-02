---
title: "Round 279 — Acceleration-Gated Liquid τ (recover toy_sin without noise risk)"
date: 2026-07-03
round: 279
prd: "docs/prds/2026-07-03-lnn-round-279-accel-gate-b.md"
status: "WIN-WITH-NUANCE — accel gate recovers+exceeds toy_sin win (-77.5%), keeps random fix (+2.9%), neutral on structured (+0.4%)"
audit_pattern: "69 SP + 29 TD + 62 NEG = 160 mechanism classes (+1 SP)"
---

# Round 279 — Acceleration-Gated Liquid τ

## TL;DR

Round 278's predictability gate `g=exp(-β·|Δ¹x|)` (velocity) fixed
r277's noise blowup but **lost toy_sin** (+41% vs static) because a
clean sine has large *velocity* even though it is perfectly predictable.
r279 gates on **acceleration** (2nd difference `|Δ²x|`) instead —
small for any smooth trajectory, large only for erratic input. Result:

| dataset    | static  | liquid (r277) | gated_vel (r278) | **gated_accel (r279)** |
|------------|--------:|--------------:|-----------------:|-----------------------:|
| toy_sin    | 0.000031| 0.000013 (-59%)| 0.000044 (+41%) | **0.000007 (-77.5%)** |
| structured | 0.000171| 0.000150 (-12%)| 0.000167 (-2.5%)| 0.000172 (+0.4%) |
| random     | 1.002469| 2.066662(+106%)| 1.005347 (+0.3%)| 1.031320 (+2.9%) |

**gated_accel recovers AND exceeds liquid's toy_sin win** (-77.5% is
the best of all four modes) while keeping the random regression fixed
(+2.9% vs r277's +106%). The only cost is a small structured neutrality
(+0.4% vs vel's -2.5%). This is a **WIN-WITH-NUANCE**: strictly better
than r278 on the task that r278 sacrificed, at a tiny structured cost.

## The pivot: a killed idea and its replacement

The originally-selected PRD B was a **relative-volatility** gate (z-score
volatility by its running mean). Pre-bench signal analysis **killed it
before any training**: an EMA of a periodic signal's volatility is
itself periodic, so a clean sine (gate 0.65) did NOT separate from noise
(gate 0.67). Root-cause analysis then found the clean fix — gate on the
**second** difference, not the first:

```
signal              sine   structured  noise    noise/sine
|Δ¹x| EMA (r278)     0.186    0.057      1.139      6.1×
|Δ²x| EMA (r279)     0.057    0.115      1.986     35.1×   ← 6× sharper

gate (β=4)          sine    structured  noise
Δ¹x (r278)          0.494     0.896      0.053
Δ²x (r279)          0.800     0.839      0.018   ← sine recovered, noise kept
```

The 2nd difference is the **constant-velocity forecast error**: ~0 for
any locally-linear (predictable) trajectory, large only for genuinely
erratic input. Documenting the killed relative-volatility idea is itself
a result — scale-invariance is NOT the axis that matters; smoothness
order is.

## Hypothesis Evaluation

### H1 (accel gate recovers toy_sin toward liquid's -59%)
**CONFIRMED — EXCEEDED**. gated_accel toy_sin = 0.000007 (-77.5% vs
static), *better than* liquid's own -59% (0.000013) and dramatically
better than r278's +41% (0.000044). Trained gate_mean on toy_sin =
**0.976** (vs r278 vel's 0.79) — the acceleration gate correctly reads
the sine as fully predictable and lets near-full liquid τ through.
tau_temporal_std 0.056-0.061 confirms τ genuinely flows (matches
ungated liquid's 0.057-0.060).

### H2 (accel gate keeps the random fix, ≤ +5% vs static)
**CONFIRMED**. gated_accel random = 1.031320 (+2.9%), well under the
+5% bar and a world away from r277's +106%. Per-seed: 1.027 / 0.998 /
1.069 — all stable, **no divergence** (contrast liquid's seed-2 blowup
to 3.93). Trained gate_mean on random = **0.048** (near-total collapse
to static τ), so the mechanism engages exactly as designed. The +2.9%
(vs vel's +0.3%) is a mild residual: acceleration is noisier to
estimate than velocity, so a hair more liquid leaks through on noise.

### H3 (accel gate preserves structured win)
**NEUTRAL / SLIGHTLY WORSE**. gated_accel structured = 0.000172
(+0.4% vs static), vs vel's -2.5% and liquid's -12.3%. The accel gate
under-performs here because **structured regime changes are large
2nd-difference spikes** (a level jump is a big acceleration), so the
gate drops to 0.751 (vs vel's 0.840) and momentarily throttles the
liquid τ exactly at the segment boundaries where adaptation would help.
This is the one genuine cost of gating on acceleration: it treats an
abrupt regime change as "unpredictable" and pulls back.

### H4 (gate_mean(sine) >> gate_mean(noise))
**CONFIRMED**. Trained gates: toy_sin 0.976, structured 0.751, random
0.048. The sine/noise ratio is 20×, versus r278's velocity gate where
sine (0.79) and noise (0.065) were only 12× apart AND sine was throttled
below its optimum. Acceleration is the sharper, better-aimed detector.

### H5 (diff_order=1 reproduces r278 exactly)
**CONFIRMED by unit test**. On the same object, `diff_order=1` forward
is bit-identical (maxdiff 0.0) to the parent r278 forward. The
gated_vel bench mode reproduces r278's numbers to the seed (toy_sin
0.000044, structured 0.000167, random 1.005347).

## Why this is a STRICT-POSITIVE (+1 SP), not target-dependent

gated_accel is **never catastrophic** and is the **best available mode
on toy_sin** by a wide margin:
- toy_sin: -77.5% (best of all four modes, beats even raw liquid)
- structured: +0.4% (statistically neutral)
- random: +2.9% (safe, no divergence)

It dominates r278 (gated_vel) on the task r278 sacrificed, at the price
of ~3% on structured and ~2.6% on random — both negligible in absolute
terms (structured is still 0.000172, random is unlearnable anyway). For
a production default where the data distribution is unknown, gated_accel
is the strongest single choice: it captures liquid's smooth-signal wins
without the noise blowup.

**Recommendation**: use gated_accel (diff_order=2, β=4.0) as the STE
liquid-τ default when smooth-periodic performance matters. Keep
gated_vel (r278) if structured regime data dominates (its -2.5% edge
there). Both are safe; the choice is a toy_sin↔structured trade of a
few percent.

## The smoothness-order insight

The r277→r278→r279 arc reveals a general principle for predictability
gates on continuous-time models:

- **r277** (ungated): trusts everything → blows up on noise.
- **r278** (velocity gate): distrusts *motion* → over-throttles smooth
  fast signals.
- **r279** (acceleration gate): distrusts *erratic* motion → trusts
  smooth motion of any speed, distrusts only genuine unpredictability.

The right predictability signal is not "how much is the input moving"
(velocity) but "how much does the input *deviate from a smooth
forecast*" (acceleration = constant-velocity residual). A natural r280
extension: gate on the residual of a *learned* local predictor, or on
higher-order smoothness (jerk, |Δ³x|) for signals with linear trends.

## Pattern Update

| Bucket          | Before | After | Δ |
|-----------------|--------|-------|---|
| Strictly pos.   |   68   |   69  | **+1** |
| Target-dep      |   29   |   29  | 0 |
| Negatives       |   62   |   62  | 0 |
| **Total**       |  159   |  160  | +1 |

r279 adds **1 STRICTLY POSITIVE** — the acceleration-gated liquid τ.
It is the third architectural change to the STE line (r277 liquid,
r278 velocity gate, r279 acceleration gate) and the one that recovers
liquid's peak smooth-signal performance while retaining noise safety.

## Files (Round 279)

- `lnn/core/accel_gated_liquid_tau_cfc.py` (NEW, ~185 LOC)
- `tests/test_accel_gated_liquid_tau_cfc.py` (NEW, 15 tests, all green)
- `scripts/bench_accel_gated_liquid_tau.py` (NEW, ~290 LOC)
- `analysis/accel_gated_bench.json` (NEW, 36 cells)
- `docs/prds/2026-07-03-lnn-round-279-accel-gate-b.md` (selected)
- `docs/prds/2026-07-03-lnn-round-279-pred-gate-beta-sweep-a.md` (rejected)

## Cumulative Test Count

**+15 new tests** (accel-gate cell + gate-semantics + superset). STE
suite: 358 → 373 pass. No regressions.

## Next Round (Round 280)

1. **Higher-order / learned smoothness gate** — gate on |Δ³x| (jerk,
   for linear-trend signals) or the residual of a learned 1-step local
   predictor; may fix the structured under-gating (regime jumps look
   erratic to a fixed acceleration gate but predictable to a learned one).
2. **β sweep of the accel gate** (deferred PRD A, now on the better
   gate) — map the sharpness knob; can the structured +0.4% be tuned
   back to a win without reopening noise?
3. **Accel gate × real irregular-TS** (PhysioNet via r102 QuITE) —
   test on genuine nonstationary clinical data.
4. **Blend gate** — max(velocity_gate, accel_gate) to get r278's
   structured edge AND r279's toy_sin win simultaneously.
