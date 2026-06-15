# PRD #10-90 — Round 128 OscillatorCfC (arXiv:2602.12139)

**Date**: 2026-06-15
**Round**: 128
**Status**: Implemented, benched
**Verdict**: **HONEST NEGATIVE-WITH-NUANCE**

## Goal

Test whether the damped harmonic oscillator closed-form
solution from arXiv:2602.12139 (Shende et al. 2026) helps our
CfC cell. The paper claims "orders of magnitude faster"
inference by replacing ContiFormer's NODE backbone with an
oscillator that admits a closed-form matrix exponential.

## Reference

- **Oscillators Are All You Need**: arXiv:2602.12139 (Shende,
  Das, Chauhan, Pathak, Gupta — Ashoka University, February
  2026). Replaces ContiFormer (arXiv:2402.10635) backbone
  with damped harmonic oscillator.

## Design

- 2D state (h, p) where p = dh/dt
- Linear forcing F = W_x·x + b (NO h-term — required for
  closed-form)
- Per-neuron learnable ω (natural frequency) and ζ (damping
  ratio)
- Closed-form step: matrix exponential e^(A Δt) for
  underdamped case
- Tanh activation on force (required for stability)

## Files

- `lnn/core/oscillator_cfc.py` (NEW, ~200 lines)
- `tests/test_oscillator_cfc.py` (NEW, 13/13 pass)
- `scripts/bench_oscillator_cfc.py` (NEW, 12 cells)
- `results/bench_oscillator_cfc.json` (NEW)

## Bench (12 cells, 30 epochs, 2 seeds)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc**             | **0.0094** | **0.0053** | **0.0013** | 2545 |
| oscillator_cfc      | 0.0313 | 0.0607 | 0.0505 | **401** |

**OscillatorCfC loses on ALL 3 datasets** despite 6.3× fewer
parameters (401 vs 2545).

## Verdict

**HONEST NEGATIVE-WITH-NUANCE**:
- 4th ODE family tested (CfC, LTC, MoR, Oscillator)
- 2nd-order damped harmonic oscillator closed-form is
  mathematically elegant
- But the closed-form constraint (F = W_x·x + b, no h-term)
  destroys recurrent nonlinearity
- 13th negative in 91-128 audit

## Why it fails

The paper's setting is **transformer attention** (queries
drive keys/values through external inputs). The recurrent
RNN setting requires F to depend on h, which the closed-form
cannot accommodate. With F = W_x·x + b only, the cell loses
all the recurrent nonlinearity that CfC gets from
W_h·h + sigmoid + tanh.

## Future work

1. Use oscillator as a backbone in MoE (not standalone)
2. Hybrid Oscillator+CfC
3. Critically-damped / overdamped variants
4. Test on PhysioNet 36D
5. Use oscillator for attention (paper's original setting)
