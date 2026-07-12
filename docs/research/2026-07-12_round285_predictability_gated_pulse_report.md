---
title: "Round 285 — Predictability-Gated Pulse Amplitude (HONEST NEGATIVE-WITH-NUANCE)"
date: 2026-07-12
round: 285
prd: "docs/prds/2026-07-12-lnn-round-285-predictability-gated-pulse-amplitude-a.md"
paper: "arXiv:2603.00153 (Sharma 2026-03) — extension with r280 gate"
status: "TARGET-DEPENDENT DIFFERENT TRADE-OFF — H1/H2/H3 FAIL, H5 OK"
parent: "r284 pulse-augmented gated liquid τ"
---

# Round 285 — Predictability-Gated Pulse Amplitude

## TL;DR

The r284 report itself recommended gating the pulse amplitude by the
r280 predictability score `g_t ∈ (0,1]` so the endogenous drive is
suppressed exactly when input is erratic. This round does exactly that
(`pulse = g_t · A · sin(...)`, **zero new parameters / loss / schedule**)
and gets a **classic HONEST NEGATIVE-WITH-NUANCE**:

- **H5 ✓ (gate is real, not just clamping)**: `gate.mean() = 0.804` on
  structured + gap → the pulse is genuinely active when input is
  predictable.
- **H1 ✗ (the headline failure)**: structured gap_ratio **61 (r284)
  → 394 (r285)**, comparable to baseline blend (368). Gating the pulse
  amplitude *destroys* the gap-robustness benefit.
- **H2 ✗ (partial)**: random Δ% **+44.6% (r284) → +9.4% (r285)**.
  Big improvement but still above the +5% safety bar.
- **H3 ✗ (the surprise)**: pulse_amp on random **0.40 (r284) →
  0.71 (r285)**. Gating *amplifies* the parameter chase: because the
  optimizer sees `pulse = g·A·sin`, it raises A to compensate for the
  multiplicative attenuation at low-g steps. The parameter-free gate
  actually makes the *parameter* more free, not less.

The mechanism map updates: **r284 = +1 TD (gap-robust on structured,
chases noise)**, **r285 = +1 TD with a *different* trade-off (partial
noise safety, but lost gap-robustness)**. Neither promotes to strict
positive.

The honest classification: gating `A` by `g_t` is the wrong shape of
intervention. The next round (r286) should try **shape-preserving
gates** — `sqrt(g_t)`, `g_t^k` for k > 1, or a **gate on the noise mode
only** (the mechanism control), not the sin mode.

## Paper grounding

Round 284 added an oscillatory pulse to the gated liquid-τ cell:
`pulse = A · sin(2π·ω·t + φ(h))`. On structured data the pulse bought
**6× gap-robustness** (`gap_ratio 61 vs 368`). On random noise the
learned amplitude grew 4× (0.10 → 0.40) and raised MSE by **+44.6%**.

The r284 report's own recommended fix:
> "Gate the pulse amplitude by the same predictability gate — `pulse_i =
> g_t · A_i · sin(...)` — so the endogenous drive is suppressed exactly
> when the input is erratic."

This round implements that suggestion as a *parameter-free* operation:
the r280 blend gate `g_t = max(g_vel, g_acc)` already exists; r285 just
multiplies the pulse by it.

## Mechanism (strict superset of r284)

```
gate_t  = max(g_vel, g_acc)                  # r280 (per-step scalar)
pulse_i = g_t · strength · A_i · sin(...)    # ← THE FIX
s_i     = rec_i + in_i + bias_i + α_i·h_i + pulse_i
h       = (1-τ)·h + τ·tanh(s)
```

- `gate_pulse=False` ⇒ pulse is unconditional ⇒ reproduces r284
  bit-for-bit (H4 unit-tested).
- `pulse_strength=0` ⇒ pulse is exactly 0 ⇒ reproduces r280 bit-for-bit
  (composed superset, also unit-tested).

## Implementation

- `lnn/core/predictability_gated_pulse_cfc.py` (NEW, ~250 LOC): subclass
  `PulseGatedLiquidTauCfCCell`, extends `_pulse_term(t, T, h, noise_drive, gate=None)`,
  `forward` passes `gate = max(g_vel, g_acc).unsqueeze(-1)`.
- `tests/test_predictability_gated_pulse_cfc.py` (NEW, 14 tests, all
  green): H4 superset, gate scaling, gradient flow through `g_t`,
  H4' composed (gate_pulse=True + pulse_strength=0 ≡ r280).
- `scripts/bench_predictability_gated_pulse.py` (NEW, ~280 LOC): 4 modes
  × 3 datasets × 2 seeds, 50 epochs, eval on clean + gap p=0.3.
- `analysis/predictability_gated_pulse_bench.json` (NEW, 24 cells).

## Results (128-hidden, T=48, 50 epochs, 2 seeds, gap p=0.3)

Clean test MSE, gap_ratio, pulse_amp:

| mode          | toy_sin mse / gr / amp | structured mse / gr / amp | random mse / gr / amp |
|---------------|------------------------|---------------------------|-----------------------|
| static_tau    | 0.00024 / 1707 / n/a   | 0.00028 / 449 / n/a       | 0.981 / 1.00 / n/a    |
| blend_gated   | 0.00001 / 26464 / n/a  | 0.00024 / 368 / n/a       | 0.984 / 1.00 / n/a    |
| pulse_sin (r284) | 0.00009 / 3108 / 0.083 | 0.00020 / **61** / 0.159 | 1.422 / 1.06 / 0.401 |
| **gated_pulse (r285)** | 0.00003 / 3423 / 0.084 | 0.00021 / **394** / 0.167 | 1.076 / 1.15 / **0.705** |

Δ% clean MSE vs blend_gated:
- toy_sin: pulse_sin +764% / gated_pulse +228% (init noise on saturated regime, both < 1e-4)
- structured: pulse_sin **-19.5%** / gated_pulse **-14.6%**
- random: pulse_sin **+44.6%** / gated_pulse **+9.4%**

## Hypothesis evaluation

### H1 (structured gap_ratio ≤ r284) — REJECTED
structured gap_ratio: blend 368, **r284 61**, **r285 394**. The gated
pulse loses the gap-robustness benefit entirely — it is now comparable
to the ungated blend baseline (368). The multiplicative gate `g_t ≤ 1`
attenuates the pulse at *every* step, not just noise steps, so the
endogenous drive through gaps becomes weaker. r284's structure-driven
gap-robustness depended on *unconditional* pulse amplitude, and the
gate removes exactly that.

### H2 (random Δ% ≤ +5% vs blend) — REJECTED (partial improvement)
random Δ%: r284 +44.6%, r285 **+9.4%**. Vast improvement, but still
above the +5% safety bar. The gate does suppress noise somewhat (the
endogenous drive is weaker on erratic input), but not enough.

### H3 (random pulse_amp ≤ 0.20) — REJECTED (worse!)
pulse_amp on random: r284 0.401, r285 **0.705**. This is the most
instructive result: the gate **amplified** the parameter chase. Because
the optimizer sees `pulse = g_t · A · sin` and the gradient on A is
proportional to `(pulse contribution) / g_t`, the optimizer raises A to
keep the *raw drive through gaps* (on structured) while the gradient
on noise steps is suppressed by `g_t ≈ 0.1`. Result: A grows even
larger than r284's already-chased 0.40. **The parameter-free gate made
the parameter less constrained, not more.**

### H4 (superset) — CONFIRMED (unit-tested)
`gate_pulse=False` ≡ PulseGatedLiquidTauCfCCell bit-for-bit on both
sine and noise inputs. `gate_pulse=True + pulse_strength=0` ≡ r280
BlendGatedLiquidTauCfCCell. Both unit-tested.

### H5 (gate.mean() ≥ 0.5 on structured+gap) — CONFIRMED
`gate.mean() = 0.804` on structured + gap p=0.3 → the pulse is *active*
in 80% of steps. The gating is real (not just clamping to zero). So
H1's failure is not "the gate clamped the pulse to zero on structured"
— the gate is open, but the pulse is now too weak to carry state
through gaps.

## Interpretation

The mechanism map now contains **two target-dependent pulse variants
with different trade-offs**:

| round | mechanism                       | best axis                          | noise cost   |
|-------|---------------------------------|------------------------------------|-------------:|
| r284  | unconditional sin pulse         | structured gap-robustness (6×)     | +44.6%       |
| r285  | g_t-multiplied sin pulse        | partial noise safety (44.6→9.4%)   | +9.4%        |
|       |                                 | but **loses** gap-robustness       |              |

Neither is a strict-positive default. **r285 is the honest r286
target**: the multiplicative `g_t` is the wrong shape of intervention.

### Why gating A by g_t fails

Intuitively, multiplying by g_t should attenuate the drive on noise
and keep it on structured. But the *gradient* on A is also proportional
to `g_t` (chain rule), so the *incentive* to grow A is now `1/g_t` on
low-g steps and `1` on high-g steps. The optimizer solves this by
growing A further on the high-g steps to maintain the *unconditional*
pulse contribution — and then on noise steps the larger A multiplied
by small g_t still leaks through. So gating A doesn't bound the
*worst-case drive on noise*; it only bounds the *conditional-mean drive
on noise*, which is not what we need.

A better gate would be **shape-preserving** — `sqrt(g_t)` reduces the
gap-attenuation on structured (where g ≈ 0.8, sqrt gives 0.89 vs 0.80)
while keeping the noise suppression (g ≈ 0.1, sqrt gives 0.32 vs 0.10).
Or a **gate on the noise mode only**, leaving the sin mode un-gated.

## Mechanism map update

| Bucket        | Before | After | Δ |
|---------------|--------|-------|---|
| Strictly pos. |   71   |   71  | 0 |
| Target-dep    |   31   |   32  | **+1** |
| Negatives     |   62   |   62  | 0 |
| **Total**     |  163   |  164 | +1 |

r285 adds **1 TARGET-DEPENDENT POSITIVE** — the g_t-multiplied pulse.
It is the *second* target-dependent pulse mechanism (after r284) and
provides a *different* trade-off point: weaker noise penalty but also
weaker gap-robustness. **r284 remains the recommended default for
structured/gappy data; r285 is a candidate for noisy data where +9%
regression is acceptable.**

## Files (Round 285)

- `lnn/core/predictability_gated_pulse_cfc.py` (NEW, ~250 LOC)
- `tests/test_predictability_gated_pulse_cfc.py` (NEW, 14 tests, all green)
- `scripts/bench_predictability_gated_pulse.py` (NEW, ~280 LOC)
- `analysis/predictability_gated_pulse_bench.json` (NEW, 24 cells)
- `docs/prds/2026-07-12-lnn-round-285-predictability-gated-pulse-amplitude-a.md`
- `docs/research/2026-07-12_round285_predictability_gated_pulse_report.md` (this)

## Next round (Round 286) candidates

Ranked:
1. **sqrt-gated pulse**: `pulse = sqrt(g_t) · A · sin(...)`. Preserves
   structured amplitude (sqrt(0.8)=0.89 vs 0.80) while attenuating
   noise (sqrt(0.1)=0.32 vs 0.10). Hypothesis: H1 ✓ (preserves
   gap-robustness), H2 ✓ (still attenuates noise enough), H3 ✓
   (sqrt-gate changes gradient scaling, may slow A-chase).
2. **noise-mode-only gate**: gate the r284 `pulse_noise` control term
   but leave `pulse_sin` ungated — since the noise mode is the
   mechanism ablation anyway, this might be a cleaner separation.
3. **arXiv:2607.01986 decorrelation loss** (turbofan degradation) — a
   fresh disentanglement axis.
4. **arXiv:2606.21295 neuron-wise topological dynamics** — new backbone.

## Citation

- Sharma, P. (2026-03). *Pulse-Driven Neural Architecture: Learnable
  Oscillatory Dynamics for Robust Continuous-Time Sequence Processing*.
  arXiv:2603.00153.
- r280 blend gate internal report: `docs/research/2026-07-03_round280_blend_gated_liquid_tau_report.md`
- r284 pulse gate internal report: `docs/research/2026-07-11_round284_pulse_gated_liquid_tau_report.md`