---
title: "Round 287 — Binary-Gated Pulse Amplitude (HONEST NEGATIVE — gating breaks continuity)"
date: 2026-07-12
round: 287
prd: "docs/prds/2026-07-12-lnn-round-287-binary-gated-pulse-a.md"
paper: "arXiv:2603.00153 (Sharma 2026-03) — extension with additive threshold gate"
status: "FAIL H1 (catastrophic), PASS H2 at τ ∈ {0.5, 0.7} — gating breaks the 'continuous endogenous rhythm' premise"
parent: "r286 sqrt-gate; r285 linear-gate; r284 unconditional pulse"
---

# Round 287 — Binary-Gated Pulse Amplitude

## TL;DR

Four rounds of pulse variants (r284 / r285 / r286 / r287) **all fail
to achieve strict-positive default**. This round's key finding is more
informative than the previous three combined: **the binary gate
successfully kills noise chasing (H2 ✓ at τ=0.5: random Δ% +2.9%) but
catastrophically destroys gap-robustness (H1 ✗: structured gap_ratio
611 vs r284's 61).** The mechanism is clear: the pulse's "endogenous
rhythm carries state through gaps" claim requires **continuous drive**,
but ANY gate (multiplicative or threshold) interrupts the pulse when
`g_t` momentarily dips during a gap, breaking the rhythm. Gating the
pulse amplitude is the wrong intervention axis entirely.

## Results (128-hidden, T=48, 50 epochs, 2 seeds, gap p=0.3)

Clean test MSE, gap_ratio, pulse_amp:

| mode              | toy_sin mse / gr / amp | structured mse / gr / amp | random mse / gr / amp |
|-------------------|------------------------|---------------------------|-----------------------|
| static_tau        | 0.00024 / 1707 / n/a   | 0.00028 / 449 / n/a       | 0.981 / 1.00 / n/a    |
| blend_gated (r280)| 0.00001 / 26464 / n/a  | 0.00024 / 368 / n/a       | 0.984 / 1.00 / n/a    |
| pulse_sin (r284)  | 0.00009 / 3108 / 0.083 | 0.00020 / **61** / 0.159  | 1.422 / 1.06 / 0.401  |
| **binary τ=0.3**  | 0.00009 / 11180 / 0.083| 0.00017 / **544** / 0.157 | 1.063 / 1.16 / 0.581  |
| **binary τ=0.5**  | 0.00009 / 13991 / 0.083| 0.00019 / **612** / 0.156 | 1.012 / 1.04 / 0.392  |
| **binary τ=0.7**  | 0.00009 / 16514 / 0.083| 0.00033 / **381** / 0.158 | 1.014 / 1.00 / 0.270  |

Δ% clean MSE vs blend_gated:
- structured: pulse_sin -19.5% / τ=0.3 **-30.1%** / τ=0.5 -23.6% / τ=0.7 +35.1%
- random: pulse_sin +44.6% / τ=0.3 +8.1% / τ=0.5 **+2.9%** / τ=0.7 +3.1%

## Hypothesis evaluation

### H1 (structured gap_ratio ≤ r284 = 61) — REJECTED (catastrophic)
| τ | structured gap_ratio |
|---|---------------------:|
| 0.3 | 544 |
| 0.5 | 612 |
| 0.7 | 381 |

All three thresholds are **6-10× worse than r284 (61)** and 1.5-1.7×
worse than the linear-gate r285 (394). **The binary gate destroys
gap-robustness even more than the linear gate.** This is the **most
instructive finding of r284-r287**: gating the pulse amplitude in any
form is fatal to the gap-robustness claim.

### H2 (random Δ% ≤ +5%) — PASS at τ ∈ {0.5, 0.7}
| τ | random Δ% |
|---|----------:|
| 0.3 | +8.1%  FAIL |
| 0.5 | **+2.9%  PASS** |
| 0.7 | **+3.1%  PASS** |

**The binary gate successfully kills noise chasing.** This is the first
time in the r284-r287 line that the noise safety bar is hit. The
mechanism: when `g_t ≈ 0.1 < τ`, the pulse is exactly zero, so A
contributes nothing to noise, and the optimizer cannot chase noise.

### H3 (random pulse_amp ≤ 0.20) — REJECTED (but improved)
| τ | random pulse_amp |
|---|-----------------:|
| 0.3 | 0.581 |
| 0.5 | 0.392 |
| 0.7 | 0.270 |

A-chase is reduced at higher τ (less firing → less gradient on A), but
still well above the 0.20 bar. The optimizer still grows A on the
structured steps where the pulse fires.

### H4 (H1 ∧ H2 ∧ H3) — REJECTED
H1 catastrophically fails, so H4 cannot hold. r287 is the **4th
TD-only** result in the pulse line.

### H5 (threshold=0 ≡ r284) — CONFIRMED (unit-tested)
`BinaryGatedPulseCfCCell(threshold=0)` ≡ `PulseGatedLiquidTauCfCCell`
bit-for-bit. Test: `test_threshold_zero_equals_r284`.

### H6 (threshold=10 ≡ r280) — CONFIRMED (unit-tested)
`BinaryGatedPulseCfCCell(threshold=10)` ≡ `BlendGatedLiquidTauCfCCell`
bit-for-bit. Test: `test_threshold_high_equals_blend`.

## Interpretation

### The fundamental failure mode of pulse + gate

The r287 result *combined* with r284-r286 reveals the structural
problem: **the r284 paper's claim — "endogenous oscillatory rhythm
carries state through input gaps" — requires a continuous drive that
gates fundamentally interrupt.**

- r284 (no gate): pulse always on → continuous rhythm → structured
  gap_ratio=61. But optimizer chases noise (+44.6%).
- r285 (linear gate): pulse attenuated every step → no continuous
  rhythm → gap_ratio=394. Optimizer chases A to compensate (0.71).
- r286 (sqrt gate): same issue, slightly less aggressive (gap_ratio=310).
- **r287 (binary gate): pulse fires only when g_t > τ → rhythm is
  *intermittent* → gap_ratio=544-612, even WORSE than multiplicative
  gates**. The intermittent firing is worse than the constant
  attenuation because the pulse contribution has discontinuities.

The figure of merit on the **noise** axis (random Δ%) follows the
opposite trajectory — but the gap-robustness floor is too high
(gap_ratio ≥ 380 in all gated variants) to admit any gated pulse as
strict-positive.

### What the gate actually does well

The binary gate at τ=0.5 hits **+2.9% on random** — the first
pulse-line result to satisfy the noise safety bar. The amplitude on
random drops to 0.39 (close to r284's 0.40 but now with proper noise
suppression). So the **safety axis** of the binary gate is a real,
positive contribution. The problem is purely on the **gap-robustness
axis**.

### Mechanism map: 4 rounds of pulse variants, no strict-positive

| round | mechanism                       | structured gap_ratio | random Δ% | noise-amp | verdict |
|-------|---------------------------------|---------------------:|----------:|----------:|---------|
| r284  | unconditional sin pulse         | **61**               | +44.6%    | 0.40      | TD      |
| r285  | linear `g_t`-gated              | 394                  | +9.4%     | 0.71      | TD      |
| r286  | sqrt(`g_t`)-gated               | 310                  | +25.2%    | 0.73      | TD      |
| r287  | binary `(g_t > τ)`-gated τ=0.5  | 612                  | **+2.9%** | 0.39      | TD      |

The **gap-robustness axis collapses monotonically** with gating
strength (r284 < r286 < r285 < r287). The **noise axis improves
monotonically** with gating strength (r287 < r285 < r286 < r284).
The two axes are **anti-correlated** under any single-knob
intervention. No multiplicative or threshold gate can achieve
strict-positive on both axes.

## r288 candidates

Two distinct directions from this finding:

### A. EMA-smoothed gate (intervention on the gate, not the pulse)
The gate fluctuates during gaps because `g_t` depends on `|x_t - x_{t-1}|`
which spikes when input drops out. **Smooth the gate** with an EMA:
`g_eff_t = α · g_t + (1-α) · g_eff_{t-1}`. Then threshold `g_eff`.
Hypothesis: g_eff stays high during gaps, so the pulse fires
continuously through them on structured, AND g_eff collapses on noise
because random input keeps g_t consistently low → binary mask=0.

### B. Abandon the pulse line; switch to a different mechanism
After 4 rounds (r284-r287), the pulse + gate family is fully
characterized: anti-correlated trade-off, no strict-positive. Move
to a different mechanism entirely. Top candidates from the LNN
backlog:
- **arXiv:2607.01986** decorrelation loss (turbofan disentanglement)
- **arXiv:2606.21295** neuron-wise topological dynamics
- **r99 segment reliability gate** (already in the line, extends to
  pulse inputs)
- **r100 SNNL** for expert disentanglement (already in the line)

Recommendation: try **A** (EMA-smoothed gate) once more because it's a
single-cell delta on r287 and might recover H1 without breaking H2; if
that fails, switch to **B**.

## Mechanism map update

| Bucket        | Before | After | Δ |
|---------------|--------|-------|---|
| Strictly pos. |   71   |   71  | 0 |
| Target-dep    |   33   |   34  | **+1** |
| Negatives     |   62   |   62  | 0 |
| **Total**     |  165   |  166 | +1 |

r287 adds **+1 TD** (4th pulse variant in the line).

## Files (Round 287)

- `lnn/core/binary_gated_pulse_cfc.py` (NEW, ~150 LOC): `BinaryGatedPulseCfCCell`.
- `tests/test_binary_gated_pulse_cfc.py` (NEW, 12 tests, all green).
- `scripts/bench_binary_gated_pulse.py` (NEW, ~310 LOC): 6 modes × 3
  datasets × 2 seeds × 50 epochs.
- `analysis/binary_gated_pulse_bench.json` (NEW, 36 cells).
- `docs/prds/2026-07-12-lnn-round-287-binary-gated-pulse-a.md`
- `docs/research/2026-07-12_round287_binary_gated_pulse_report.md` (this).

## Citation

- Sharma, P. (2026-03). *Pulse-Driven Neural Architecture*. arXiv:2603.00153.
- r284 pulse gate report: `docs/research/2026-07-11_round284_pulse_gated_liquid_tau_report.md`
- r285 linear-gate report: `docs/research/2026-07-12_round285_predictability_gated_pulse_report.md`
- r286 sqrt-gate report: `docs/research/2026-07-12_round286_sqrt_gated_pulse_report.md`