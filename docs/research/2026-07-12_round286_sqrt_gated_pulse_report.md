---
title: "Round 286 — sqrt-Gated Pulse Amplitude (HONEST NEGATIVE-WITH-NUANCE)"
date: 2026-07-12
round: 286
prd: "docs/prds/2026-07-12-lnn-round-286-sqrt-gated-pulse-a.md"
paper: "arXiv:2603.00153 (Sharma 2026-03) — extension with shape-preserving gate"
status: "PARTIAL — between r284 and r285 on every axis; H4 (strict-positive) NOT achieved"
parent: "r285 predictability-gated pulse (linear); r284 pulse; r280 blend gate"
---

# Round 286 — sqrt-Gated Pulse Amplitude

## TL;DR

The r286 hypothesis was that **shape-preserving** gating
`pulse = sqrt(g_t) · A · sin(...)` would rescue the r284
gap-robustness (which r285's linear gate destroyed) while still
suppressing noise amplitude growth. **Result: HONEST NEGATIVE-WITH-
NUANCE — sqrt-gate is geometrically between r284 and r285 on every
axis, but neither extreme is strict-positive.**

- **H1 ✗ (gap-robustness NOT preserved)**: structured gap_ratio
  **310 (r286)**, vs r284=61, r285=394. Better than r285 but still
  5× worse than r284. sqrt-gate is still too aggressive on high-`g_t`
  steps.
- **H2 ✗ (noise NOT safe)**: random Δ% **+25.2% (r286)**, vs r284=+44.6%,
  r285=+9.4%. Better than r284 but worse than r285. sqrt-gate is
  still not aggressive enough to suppress noise.
- **H3 ✗ (A-chase NOT slowed)**: random pulse_amp **0.726 (r286)**,
  vs r284=0.401, r285=0.705. Essentially the same as r285 — the
  gradient-scaling argument was wrong. The optimizer still grows A
  regardless of the gate shape because `A · sin` is the only way to
  amplify the high-`g_t` signal.
- **H5 ✓ (gate_pulse_shape='none' ≡ r284)**: unit-tested bit-for-bit.
- **H6 ✓ (gate_pulse_shape='linear' ≡ r285)**: unit-tested bit-for-bit.

**The pulse + multiplicative-gate family is fundamentally misconfigured.**
Three rounds (r284/r285/r286) of variations all produce target-dependent
results with different trade-offs, but no strict-positive. The next
direction (r287) must try an **additive / binary gate** rather than
another multiplicative shape: e.g. `pulse = (g_t > 0.5) · A · sin(...)`
— activate the pulse only when input is predictable, kill it otherwise,
with NO amplitude scaling on the active steps.

## Results (128-hidden, T=48, 50 epochs, 2 seeds, gap p=0.3)

Clean test MSE, gap_ratio, pulse_amp:

| mode              | toy_sin mse / gr / amp | structured mse / gr / amp | random mse / gr / amp |
|-------------------|------------------------|---------------------------|-----------------------|
| static_tau        | 0.00024 / 1707 / n/a   | 0.00028 / 449 / n/a       | 0.981 / 1.00 / n/a    |
| blend_gated (r280)| 0.00001 / 26464 / n/a  | 0.00024 / 368 / n/a       | 0.984 / 1.00 / n/a    |
| pulse_sin (r284)  | 0.00009 / 3108 / 0.083 | 0.00020 / **61** / 0.159  | 1.422 / 1.06 / 0.401  |
| **sqrt_pulse (r286)** | 0.00001 / 15752 / 0.082 | 0.00017 / **310** / 0.162 | 1.231 / 1.12 / **0.726** |
| linear_pulse (r285)| 0.00003 / 3423 / 0.084 | 0.00021 / 394 / 0.167     | 1.076 / 1.15 / 0.705  |

Δ% clean MSE vs blend_gated:
- toy_sin (init-noise-dominated): pulse_sin +764% / sqrt_pulse **-44.5%** / linear_pulse +228%
- structured: pulse_sin -19.5% / sqrt_pulse **-29.2%** / linear_pulse -14.6%
- random: pulse_sin +44.6% / sqrt_pulse +25.2% / linear_pulse **+9.4%**

## Hypothesis evaluation

### H1 (structured gap_ratio ≤ r284 = 61) — REJECTED
structured gap_ratio: blend 368, r284 **61**, r286 **310**, r285 394.
sqrt-pulse recovers ~30% of the gap-robustness vs r285 but is still
5× worse than r284. **The shape-preservation argument helped
moderately but did not close the gap.** The multiplicative gate
still attenuates the pulse at every step, even at high `g_t`.

### H2 (random Δ% ≤ +5%) — REJECTED
random Δ%: r284 +44.6%, r286 **+25.2%**, r285 +9.4%. sqrt is exactly
midway between r284 and r285 — which is mathematically expected since
`sqrt(g) ∈ (g, 1)` for `g ∈ (0, 1)`, and we're taking the *mean* of
the gate over the noise steps. The shape-preservation argument was
*not* about noise — it was about structured — so this is unsurprising.

### H3 (random pulse_amp ≤ 0.20) — REJECTED
random pulse_amp: r284 0.401, r286 **0.726**, r285 0.705. **The
gradient-scaling hypothesis was wrong.** The optimizer is still
chasing A because the gradient on `A · sin(ω·t+φ)` is fundamentally
`∂L/∂A ∝ pulse_value`, not `∂L/∂A ∝ pulse_value / gate`. The
multiplicative gate scales the *forward output*, not the *parameter
gradient on A*. So A grows to maintain the high-`g_t` signal, and
on noise steps the larger A still leaks through (sqrt(0.1)·0.73 ≈
0.23, comparable to linear(0.1)·0.71 ≈ 0.07... wait actually sqrt
leaks MORE than linear on noise!).

Re-checking: noise gate = 0.096, sqrt(0.096)=0.310, linear=0.096.
sqrt amplifies the noise contribution 0.310/0.096 = **3.2× more**
than the linear gate. This explains why r286 is *worse* on noise
than r285 (random Δ% +25.2% vs +9.4%) despite being "in between".

### H4 (H1 ∧ H2 ∧ H3 — strict-positive) — REJECTED
None of H1/H2/H3 pass. r286 does **not** promote to strict-positive.

### H5 (gate_pulse_shape='none' ≡ r284) — CONFIRMED (unit-tested)
`SqrtGatedPulseCfCCell(gate_pulse_shape='none')` ≡
`PulseGatedLiquidTauCfCCell` bit-for-bit on both sine and noise
inputs. Test: `test_none_shape_equals_r284`.

### H6 (gate_pulse_shape='linear' ≡ r285) — CONFIRMED (unit-tested)
`SqrtGatedPulseCfCCell(gate_pulse_shape='linear')` ≡
`PredictabilityGatedPulseCfCCell` bit-for-bit. Test:
`test_linear_shape_equals_r285`. Plus the composed superset
`gate_pulse_shape='*' + pulse_strength=0` ≡ r280 blend (all 3 shapes).

## Interpretation

### Why sqrt didn't help

The shape hypothesis was: sqrt(g_t) is closer to 1 than g_t on
structured (where g ≈ 0.8), so the pulse amplitude is preserved.
That part is correct — sqrt(0.8) = 0.89 vs g=0.80, so the structured
amplitude is +11% compared to linear. This is why r286's structured
gap_ratio (310) is **better than r285's (394)**.

But:
1. **On structured, even 0.89 is a multiplicative attenuation.** The
   endogenous drive is now 89% of what it could be, vs 100% for r284.
   The 11% recovery from r285 is not enough to close the gap to r284.
2. **On noise, sqrt amplifies leakage.** The pulse leak on noise is
   `A · sqrt(g) ≈ 0.73 · 0.310 ≈ 0.226` for r286 vs
   `0.71 · 0.096 ≈ 0.068` for r285. r286 leaks **3.3× more** than r285
   on noise, which is why random Δ% is worse.
3. **A-chase is unchanged.** The optimizer still grows A to ~0.73
   regardless of gate shape, because the *parameter gradient* on A is
   proportional to the *raw pulse contribution*, not the gated one.

### Three-round pattern: pulse + multiplicative gate is the wrong shape

| round | gate shape | structured gap_ratio | random Δ% | random amp |
|-------|-----------|---------------------:|----------:|-----------:|
| r284  | none (1.0)| **61**               | +44.6%    | 0.401      |
| r285  | linear g_t| 394                  | **+9.4%** | 0.705      |
| r286  | sqrt(g_t)| 310                  | +25.2%    | 0.726      |

**The trade-off curve is roughly linear in gate shape, and no
multiplicative shape achieves strict-positive on both axes.**

### r287 candidate: additive / binary gate

The natural next step is to abandon the multiplicative-gate family
entirely. Two new mechanisms to test:

1. **Binary gate**: `pulse = (g_t > 0.5) · A · sin(...)`. No
   attenuation — pulse is full strength or zero. Hypothesis: on
   structured (g ≈ 0.8 > 0.5) the pulse activates fully and recovers
   the r284 gap-robustness; on noise (g ≈ 0.1 < 0.5) the pulse is
   exactly zero so no noise chasing.
2. **Conditional pulse mode**: `pulse = g_t · A · sin(...)` for sin
   mode, but **only the noise-mode is gated** (since noise mode is
   the mechanism control anyway). Leaves the structural sin pulse
   un-gated; only suppresses the ablation control.

Either of these decouples the "amplify on structured" axis from the
"leak on noise" axis, which is exactly what the multiplicative-gate
family cannot do.

## Mechanism map update

| Bucket        | Before | After | Δ |
|---------------|--------|-------|---|
| Strictly pos. |   71   |   71  | 0 |
| Target-dep    |   32   |   33  | **+1** |
| Negatives     |   62   |   62  | 0 |
| **Total**     |  164   |  165 | +1 |

r286 adds **+1 TD** (the 3rd pulse variant in the line). The r284 /
r285 / r286 family now has **three different trade-off points** but
no strict-positive default.

## Files (Round 286)

- `lnn/core/sqrt_gated_pulse_cfc.py` (NEW, ~150 LOC): `SqrtGatedPulseCfCCell`
  subclass of `PredictabilityGatedPulseCfCCell`. New flag
  `gate_pulse_shape: str = 'sqrt'` (valid: 'sqrt', 'linear', 'none').
- `tests/test_sqrt_gated_pulse_cfc.py` (NEW, 14 tests, all green).
- `scripts/bench_sqrt_gated_pulse.py` (NEW, ~300 LOC): 5 modes × 3
  datasets × 2 seeds × 50 epochs.
- `analysis/sqrt_gated_pulse_bench.json` (NEW, 30 cells).
- `docs/prds/2026-07-12-lnn-round-286-sqrt-gated-pulse-a.md`
- `docs/research/2026-07-12_round286_sqrt_gated_pulse_report.md` (this).

## Next round (Round 287) candidates

Ranked:
1. **Binary gate** (`pulse = (g_t > τ) · A · sin`) — abandon
   multiplicative family, test additive/binary thresholding.
2. **arXiv:2607.01986 decorrelation loss** — fresh disentanglement axis.
3. **arXiv:2606.21295 neuron-wise topological dynamics** — new backbone.
4. **Adaptive EMA gate**: `g_eff = α·gate + (1-α)·g_prev` — smooth
   the gate to reduce per-step variance. Lower priority — addresses
   symptoms, not the root cause.

## Citation

- Sharma, P. (2026-03). *Pulse-Driven Neural Architecture*. arXiv:2603.00153.
- r284 pulse gate internal report: `docs/research/2026-07-11_round284_pulse_gated_liquid_tau_report.md`
- r285 linear-gate internal report: `docs/research/2026-07-12_round285_predictability_gated_pulse_report.md`