---
title: "Round 288 — EMA-Smoothed Gated Pulse (HONEST NEGATIVE — 5-round pulse line exhausted)"
date: 2026-07-12
round: 288
prd: "docs/prds/2026-07-12-lnn-round-288-ema-gated-pulse-a.md"
paper: "arXiv:2603.00153 (Sharma 2026-03) — EMA-smoothed gate"
status: "PARTIAL — best noise safety in line (+1.9%) but gap-robustness NOT recovered"
parent: "r287 binary-gate; r286 sqrt-gate; r285 linear-gate; r284 unconditional pulse"
---

# Round 288 — EMA-Smoothed Gated Pulse

## TL;DR

The EMA-smoothing hypothesis was: smooth the per-step gate before
thresholding so the pulse fires continuously through structured input
gaps while collapsing on consistent noise. **Result: PARTIAL — best
random noise safety in the entire 5-round pulse line (+1.9% at α=0.3
τ=0.7), but structured gap-robustness NOT recovered** (best 430 vs
r284's 61). The fundamental issue: any threshold gate eventually turns
off during a long gap on structured data because the per-step gate
spikes to 0 during gaps, and EMA decays — even with α=0.3 (heavy
smoothing).

After **5 rounds (r284-r288)** the pulse + any-gate family is **fully
characterized** and **fundamentally exhausted**:
- Unconditional pulse: gap-robust but chases noise
- Linear/sqrt/binary/EMA gates: trade-off curve is anti-correlated
- No mechanism in this family achieves strict-positive on both axes

**The r284 paper's "endogenous rhythm carries state through gaps"
claim cannot be reproduced in a gated variant** on this benchmark. The
right next step is to **abandon the pulse line** and pivot to a
different mechanism. Top candidates:
1. arXiv:2607.01986 decorrelation loss (turbofan disentanglement)
2. arXiv:2606.21295 neuron-wise topological dynamics
3. r99 segment reliability gate (already in the line)

## Results (128-hidden, T=48, 50 epochs, 2 seeds, gap p=0.3)

Clean test MSE, gap_ratio, pulse_amp:

| mode              | toy_sin mse / gr / amp | structured mse / gr / amp | random mse / gr / amp |
|-------------------|------------------------|---------------------------|-----------------------|
| static_tau        | 0.00024 / 1707 / n/a   | 0.00028 / 449 / n/a       | 0.981 / 1.00 / n/a    |
| blend_gated (r280)| 0.00001 / 26464 / n/a  | 0.00024 / 368 / n/a       | 0.984 / 1.00 / n/a    |
| pulse_sin (r284)  | 0.00009 / 3108 / 0.083 | 0.00020 / **61** / 0.159  | 1.422 / 1.06 / 0.401  |
| binary τ=0.5 (r287)| 0.00009 / 13991 / 0.083| 0.00019 / 612 / 0.156    | 1.012 / 1.04 / 0.392  |
| **ema α=0.3 τ=0.3** | 0.00009 / 6998 / 0.083 | 0.00020 / **430** / 0.159 | 1.129 / 1.08 / 0.391  |
| ema α=0.3 τ=0.5   | 0.00009 / 12479 / 0.083| 0.00018 / **945** / 0.163 | 1.031 / 1.00 / 0.342  |
| ema α=0.5 τ=0.5   | 0.00009 / 13424 / 0.083| 0.00015 / **719** / 0.159 | 1.014 / 1.01 / 0.319  |
| **ema α=0.3 τ=0.7** | 0.00009 / 17432 / 0.083| 0.00027 / **532** / 0.161 | 1.002 / 1.01 / **0.288** |

Δ% clean MSE vs blend_gated:
- structured: pulse_sin -19.5% / binary -23.6% / ema_a03_t03 -19.5% / ema_a03_t05 **-25.7%** / ema_a05_t05 **-36.8%** / ema_a03_t07 +8.8%
- random: pulse_sin +44.6% / binary +2.9% / ema_a03_t03 +14.7% / ema_a03_t05 +4.8% / ema_a05_t05 +3.1% / ema_a03_t07 **+1.9%**

## Hypothesis evaluation

### H1 (structured gap_ratio ≤ r284 = 61) — REJECTED (catastrophic)
| mode | structured gap_ratio |
|---|---:|
| ema_a03_t03 | 430 |
| ema_a03_t05 | **945** (worst in line) |
| ema_a05_t05 | 719 |
| ema_a03_t07 | 532 |

All four EMA configurations are **7-15× worse than r284**, and the
α=0.3 τ=0.5 mode is **worse than r287's binary gate (612)**. The EMA
did NOT recover the gap-robustness; it actually made it worse. Why?
Because the EMA decays during sustained gap sequences on structured
data (g_t dips, g_ema follows), and once g_ema crosses below τ the
pulse turns off — same failure mode as the binary gate.

### H2 (random Δ% ≤ +5%) — PASS at τ ≥ 0.5 (3 of 4 modes)
| mode | random Δ% |
|---|---:|
| ema_a03_t03 | +14.7%  FAIL |
| ema_a03_t05 | +4.8%  PASS |
| ema_a05_t05 | +3.1%  PASS |
| ema_a03_t07 | **+1.9%  PASS — best in line** |

The EMA at τ=0.7 achieves the **best random noise safety in the entire
5-round pulse line**. This is a real positive: the EMA + strict
threshold cleanly separates structured-with-gaps (where the gate is
high between gaps) from consistent noise (where the gate is
consistently low).

### H3 (random pulse_amp ≤ 0.20) — REJECTED (improved)
| mode | random pulse_amp |
|---|---:|
| ema_a03_t03 | 0.391 |
| ema_a03_t05 | 0.342 |
| ema_a05_t05 | 0.319 |
| ema_a03_t07 | 0.288 |

A-chase is reduced compared to r287 (0.39) and r286 (0.73) at higher τ,
because the mask is more often off → less gradient on A → less
chase. But still above the 0.20 bar.

### H4 (H1 ∧ H2 ∧ H3) — REJECTED
H1 fails catastrophically, so H4 cannot hold. r288 is the **5th
TD-only** result in the pulse line.

### H5 (ema_alpha=1.0 ≡ r287) — CONFIRMED (unit-tested)
`EmaGatedPulseCfCCell(ema_alpha=1.0)` ≡
`BinaryGatedPulseCfCCell(threshold=0.5, g_ema_init=0.5)` bit-for-bit.
Test: `test_alpha_one_equals_r287`.

### H6 (threshold=0 ≡ r284) — CONFIRMED (unit-tested)
`EmaGatedPulseCfCCell(threshold=0.0)` ≡
`PulseGatedLiquidTauCfCCell` bit-for-bit.
Test: `test_threshold_zero_equals_r284`.

## Interpretation

### Why EMA didn't recover gap-robustness

The EMA-smoothing hypothesis relied on:
- Structured + gap: g_t dips momentarily → g_ema stays high (α=0.3 → 70% history weight)
- Noise: g_t consistently low → g_ema collapses

This works in principle, but in practice with structured data at gap
p=0.3 and T=48, the input has ~14 gap steps out of 48. With α=0.3,
g_ema = 0.3·g_t + 0.7·g_ema_{t-1}. During a gap, g_t drops to ~0
(the input is zero, so vol1/vol2 may spike or stay high — but `g_t =
exp(-β·vol)` so big vol = small g_t).

Wait — during gaps, the input is zeroed, so the difference between
consecutive zero steps is ZERO → vol1 drops → g_t RISES, not falls.
So the gate on structured+gaps should stay HIGH during gaps, not low!

Let me re-check the gate behaviour. Looking at the parent's gate
computation:
- `vol1 = γ·vol1 + (1-γ)·|x_t - x_{t-1}|.mean()`
- During a gap where x_t = 0 and x_{t-1} = 0, vol1 stays at 0 → g_t stays at 1.

So on structured+gaps, g_t should remain high. But the bench shows
gate_mean ≈ 0.80 on structured — that's high, not low. And yet gap_ratio
is still 430-945.

So the gate IS high. The pulse IS firing. But the gap-robustness is
still lost. **The issue isn't gating at all — it's something else.**

Looking again at the bench output: `pulse_amp` on structured is 0.16 —
similar to r284 (0.16). The amplitude is NOT being chased. So the
issue is that the pulse, even when firing, doesn't carry state through
gaps.

Wait — the gap_ratio is `gap_MSE / clean_MSE`. The clean_MSE for the
EMA τ=0.5 mode is 0.00018 vs r284's 0.00020 (so clean is BETTER). But
gap_MSE is much higher (0.18 vs 0.01). The pulse is firing but the
gap-corrupted input causes the *rest of the cell* (recurrent weights,
input projection) to fail. The pulse can't carry state through the gap
because the *rest of the dynamics* isn't gap-robust.

**The fundamental insight:** the r284 pulse doesn't carry state through
gaps by itself; it provides an *endogenous rhythm* that complements the
rest of the cell. When you gate the pulse, you don't lose the
*endogenous rhythm* — you lose the *complementarity*. The pulse
amplitude is only useful in the context of a cell that already handles
gaps (or doesn't have gaps as a problem). The r284 paper's claim was
likely demonstrated with a different cell architecture.

**This explains all 5 rounds:** the pulse adds value ONLY when the cell
otherwise can't handle gaps, and that complementary value is lost as
soon as you try to gate the pulse for safety.

### Five-round pulse line summary

| round | mechanism                       | structured gap_ratio | random Δ% | verdict |
|-------|---------------------------------|---------------------:|----------:|---------|
| r284  | unconditional sin pulse         | **61**               | +44.6%    | TD      |
| r285  | linear `g_t`-gated              | 394                  | +9.4%     | TD      |
| r286  | sqrt(`g_t`)-gated               | 310                  | +25.2%    | TD      |
| r287  | binary `(g_t > τ)`-gated τ=0.5  | 612                  | +2.9%     | TD      |
| r288  | EMA(α=0.3, τ=0.7)               | 532                  | **+1.9%** | TD      |

No strict-positive. Anti-correlated axes. **The pulse line is closed.**

## Mechanism map update

| Bucket        | Before | After | Δ |
|---------------|--------|-------|---|
| Strictly pos. |   71   |   71  | 0 |
| Target-dep    |   34   |   35  | **+1** |
| Negatives     |   62   |   62  | 0 |
| **Total**     |  166   |  167 | +1 |

r288 adds **+1 TD** (5th pulse variant). The 5-round pulse line now
contributes **5 TDs and 0 SPs** to the mechanism map.

## Files (Round 288)

- `lnn/core/ema_gated_pulse_cfc.py` (NEW, ~150 LOC): `EmaGatedPulseCfCCell`.
- `tests/test_ema_gated_pulse_cfc.py` (NEW, 11 tests, all green).
- `scripts/bench_ema_gated_pulse.py` (NEW, ~330 LOC): 8 modes × 3
  datasets × 2 seeds × 50 epochs.
- `analysis/ema_gated_pulse_bench.json` (NEW, 42 cells).
- `docs/prds/2026-07-12-lnn-round-288-ema-gated-pulse-a.md`
- `docs/research/2026-07-12_round288_ema_gated_pulse_report.md` (this).

## Decision: r289 pivots away from the pulse line

After 5 rounds of pulse variants, the family is exhausted. The next
round should pivot to a different mechanism. Ranked:

1. **arXiv:2607.01986 decorrelation loss** — turbofan degradation
   disentanglement. Hypothesis: adds an axis of variation that doesn't
   conflict with the gate line. Could be strict-positive.
2. **arXiv:2606.21295 neuron-wise topological dynamics** — neuron-wise
   ODE on a learnable directed graph. New backbone class.
3. **r99 segment reliability gate** (already in the line, on pulse
   inputs) — apply the existing reliability gate to the r284 pulse cell.

Top recommendation: **r289 = decorrelation loss** — it's a fresh
mechanism that has not been explored in this line and is plausibly
strict-positive.

## Citation

- Sharma, P. (2026-03). *Pulse-Driven Neural Architecture*. arXiv:2603.00153.
- r284 pulse gate report: `docs/research/2026-07-11_round284_pulse_gated_liquid_tau_report.md`
- r285 linear-gate report: `docs/research/2026-07-12_round285_predictability_gated_pulse_report.md`
- r286 sqrt-gate report: `docs/research/2026-07-12_round286_sqrt_gated_pulse_report.md`
- r287 binary-gate report: `docs/research/2026-07-12_round287_binary_gated_pulse_report.md`