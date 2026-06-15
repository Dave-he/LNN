# PRD #10-91 — Round 129 ELMCfC (arXiv:2605.12049)

**Date**: 2026-06-15
**Round**: 129
**Status**: Implemented, benched
**Verdict**: **HONEST NEGATIVE-WITH-NUANCE**

## Goal

Test whether the Expressive Leaky Memory (ELM) neuron from
arXiv:2605.12049 (Spieler, Martius, Levina, 12 May 2026) helps
our recurrent CfC cell. The paper claims a 3-axis scaling law
(N units × k_e per-unit complexity × k_c per-unit connectivity)
and a Pareto recipe d_m ~ √N.

## Reference

- **Scaling Laws and Tradeoffs in Recurrent Networks of
  Expressive Neurons**: arXiv:2605.12049 (Spieler, Martius,
  Levina, 12 May 2026). Expressive Leaky Memory (ELM) neurons
  with multi-timescale memory units, dendritic branches, MLP
  update, and high-pass filtered output.

## Design

- H logical neurons, each with d_m=4 memory units (Pareto recipe)
- Per-memory-unit learnable κ_m (sigmoid-parameterised)
- Tanh-bounded MLP for update proposal
- EMA readout r with high-pass filtered output a = ReLU(b + w_r^T m - r)
- Skipped dendritic branches (1D setting doesn't need them)

## Files

- `lnn/core/elm_cfc.py` (NEW, ~200 lines)
- `tests/test_elm_cfc.py` (NEW, 11/11 pass)
- `scripts/bench_elm_cfc.py` (NEW, 12 cells)
- `results/bench_elm_cfc.json` (NEW)

## Bench (12 cells, 30 epochs, 2 seeds)

| Cond | sin_irr | structured_irr | random_irr | n_params |
|------|---------|----------------|------------|----------|
| **cfc**     | **0.0094** | **0.0053** | **0.0013** | 2545 |
| elm_cfc     | 0.0916 | 0.1064 | 0.1996 | 3977 |

**ELMCfC loses on ALL 3 datasets** with 1.56× more params.

### Ablation (sin_irr only, 30 epochs, 2 seeds)

| Cond | s0 | s1 | Δ vs cfc |
|------|----|----|----------|
| cfc | 0.0077 | 0.0079 | — |
| elm_cfc (with high-pass) | 0.1064 | 0.0767 | 9-13× worse |
| **elm_no_hp (no high-pass)** | 0.0140 | 0.0203 | 1.8-2.6× worse |

**Removing the high-pass filter recovers ~50% of the gap**.

## Verdict

**HONEST NEGATIVE-WITH-NUANCE**:
- 5th neuron-family tested (1st-order, n_tau, MoR, oscillator, ELM)
- 14th negative in 91-129 audit
- **Main culprit: high-pass filter** destroys DC content
- **Secondary issue**: multi-timescale + MLP update doesn't
  help in 1D, 2-3× worse than CfC even without high-pass

## Why it fails

1. **High-pass filter is task-specific**: designed for
   spike-based / cortical data (SHD, Enwik8, NeuronIO),
   actively hurts continuous-valued regression with DC content
2. **Multi-timescale memory is overkill** in 1D — CfC's
   single τ per neuron is sufficient
3. **MLP-based update is hard to train** in 30 epochs with
   batch_size=8; paper uses 100+ epochs and 700×1000 sequences

## Future work

1. Test ELM on PhysioNet 36D (may match better)
2. Use ELM as a MoE expert (multi-timescale in regime switch)
3. No-highpass + tanh output variant
4. Longer training (100+ epochs)
5. Larger d_m (8, 16) — may be needed for our setting
