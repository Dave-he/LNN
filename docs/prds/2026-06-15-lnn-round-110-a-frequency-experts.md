# PRD #10-72 — Round 110: MoFE-Time Frequency-Domain Experts (response to arXiv:2507.06502)

**Date**: 2026-06-15
**Round**: 110
**Paper**: arXiv:2507.06502 — *MoFE-Time: Mixture of Frequency Domain Experts for Time-Series Forecasting Models* (Liu et al. Jul 2025)
**Status**: To implement

## Motivation

Our 91-109 audit shows:
- Structural > routing-only: 5 structural winners (99, 102, 105, 107) + 1 target-dep (108) + 1 negative-with-nuance (109)
- All structural mechanisms change the architecture in some way
- No mechanism yet has touched the **frequency domain**

**MoFE-Time proposes a new structural axis**: each expert is a **learnable Fourier reconstructor** with its own harmonic frequencies and amplitudes. This is fundamentally different from prior mechanisms:
- Anchored MoE (108): structural prior on routing
- Dynamic TMoE (109): dynamic expert pool
- **MoFE-Time (110):** expert IS a learnable Fourier basis

The audit predicts this is **strictly positive on periodic data** (sin_irr, structured_irr) and **neutral on random** (no real frequencies).

## What MoFE-Time does (in 60 seconds)

Standard MoE: expert = MLP. MoFE-Time: expert = learnable Fourier reconstructor.

Per expert k:
- Has `h` learnable harmonic frequencies `{ω_i}`
- Has learnable frequencies per dimension
- Output: `x_n = Σ α_i · cos(ω_i · n) + β_i · sin(ω_i · n)` (or complex exponential form)
- Routing weights play the role of Fourier amplitudes

Key innovation: the Fourier transform is **implicit and learnable** (not pre-computed). The network must discover which frequencies matter.

## Hypotheses

**H1 — Frequency expert is structurally meaningful**: sin_irr has a single dominant frequency. Frequency experts should specialize on it (or its harmonics).

**H2 — test_mse on sin_irr improves**: clear frequency signal should be captured well.

**H3 — test_mse on structured_irr improves**: 2 dominant frequencies, both should be captured.

**H4 — test_mse on random_irr preserved or neutral**: no real frequency structure.

## Why this should help (per audit)

- **NEW axis** (frequency domain) not yet explored
- **Structural change**: each expert is a learnable Fourier reconstructor, not an MLP
- **Composes with QuITE** (102) for missing data
- **The learnable frequencies can adapt to data** — unlike fixed FFT

## Architecture

```
input: x (B, T, D)
  │
  ├── TimeDomainBranch: x → Linear → (B, T, H) [standard linear]
  │
  ├── FrequencyExpertPool: K experts, each:
  │   - Linear: x → (B, T, F) where F = #frequencies per expert
  │   - For each freq ω_i, project: cos(ω_i · t) and sin(ω_i · t) over T timesteps
  │   - Output: (B, T, H) = sum over i of learned_weights * basis
  │
  ├── Router: top-K over (B, T, K)
  │
  └── Output: time_concat_freq weighted sum of top-K experts
```

## Test plan

- FrequencyExpert: output shape, learnable frequencies, NaN-aware
- FrequencyExpertPool: K experts, all can specialize
- FrequencyMoECfCCell: end-to-end with both branches
- FrequencyMoECfCNetwork: rolling window

## Bench plan

12 cells:
- 4 conditions: `baseline_mlp` (MLP expert baseline), `freq_fixed` (fixed FFT), `freq_learned` (learnable frequencies), `freq_hybrid` (time + freq)
- 3 datasets: sin_irr, structured_irr, random_irr
- 2 seeds × 100 epochs

Measure: test_mse, expert frequency utilization, amplitude distribution.

## Files to create

- `lnn/core/freq_experts.py` (NEW, ~350 lines)
- `tests/test_freq_experts.py` (NEW, 15+ tests)
- `scripts/bench_freq_experts.py` (NEW, 12-cell bench)
- `docs/research/2026-06-15_freq_experts_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v36.md`
- `README.md` (new section)
- `lnn-round-110-freq-experts.md` (memory)

## Risks

1. **Learnable frequencies may diverge**: if not bounded, ω can grow to infinity. Solution: clamp to [0, 2π].
2. **NaN in cos/sin**: cos/sin of NaN is NaN. Solution: zero-fill before projection.
3. **High computational cost**: K experts × H frequencies × T timesteps. Test with small K.
4. **Random data**: no real frequencies → learned ω may be noise. Audit predicts this is OK.

## References

- arXiv:2507.06502 — Liu et al. (Jul 2025) *MoFE-Time: Mixture of Frequency Domain Experts for Time-Series Forecasting Models*
- arXiv:2605.20678 — round 109 (Dynamic TMoE, complementary structural)
- arXiv:2605.25166 — round 108 (Anchored MoE, complementary)
- arXiv:2606.12240 — round 77 (MR-MoE, related routing)
