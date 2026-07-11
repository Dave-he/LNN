---
title: "PRD #10-125 — Pulse-Augmented Gated Liquid τ (arXiv:2603.00153)"
round: 284
date: 2026-07-11
author: "Claude (r284 /loop session)"
status: "selected"
parent: "r280 blend gate (best gated liquid-τ); r282/r283 gate transfer"
paper: "arXiv:2603.00153 — Pulse-Driven Neural Architecture (Sharma, 2026-03)"
variant: "A"
---

> **Selected** (round 284, 2026-07-11): the r277→r283 liquid-τ gate line
> decides *how liquid* τ is at each step. The freshest LNN-cell paper of
> the harvest — arXiv:2603.00153 (Pulse-Driven Neural Architecture) —
> proposes an orthogonal knob: a learnable **oscillatory pulse**
> `A·sin(ω·t+φ(h))` injected into the state so it keeps evolving with an
> endogenous rhythm through erratic input or gaps. The paper's headline
> control (an equal-magnitude *non-oscillatory* perturbation gives no
> benefit) is a clean, falsifiable mechanism test we can replicate on the
> existing toy gate benchmark plus an eval-time gap condition.

# PRD #10-125 — Pulse-Augmented Gated Liquid τ

## 目标
Test whether grafting the paper's learnable oscillatory pulse onto the
r280 blend-gated liquid-τ cell (a) helps the periodic toy_sin task,
(b) is safe on noise, and (c) improves robustness to eval-time input
gaps (temporal dropout) — and whether that robustness comes from the
oscillatory STRUCTURE (sin) rather than added capacity (RMS-matched
non-oscillatory control).

## 用户故事
- As a gate-line maintainer, I learn whether an endogenous pulse is a
  useful orthogonal addition to the predictability gate, or another
  target-dependent knob.
- As a researcher, I get a direct replication of arXiv:2603.00153's core
  claim (structure > magnitude) on a controlled 1D benchmark.
- As a downstream user with gappy/irregular inputs, I get evidence on
  whether the pulse buys gap-robustness for free on clean data.

## 引擎层职责 (canonical)
- `lnn/core/pulse_gated_liquid_tau_cfc.py` (NEW): `PulseGatedLiquidTauCfCCell`
  subclassing `BlendGatedLiquidTauCfCCell`. Adds per-neuron learnable
  amplitude `A`, angular frequency `ω`, base phase `φ0`, and a state→phase
  projection `W_φ`. `pulse_strength=0` reproduces r280 exactly (superset).
  `pulse_mode='noise'` is the RMS-matched non-oscillatory control.

## 游戏层职责
- `scripts/bench_pulse_gated_liquid_tau.py` (NEW): 4 modes (static_tau,
  gated_blend, pulse_sin, pulse_noise) × 3 datasets × 2 seeds, 50 epochs.
  Each trained model evaluated on clean AND gap-corrupted test
  (temporal dropout p=0.3); report clean MSE, gap MSE, gap_ratio.
- `analysis/pulse_gated_bench.json`, report in `docs/research/`.

## 验收标准
- H1 (headline): pulse_sin ≤ gated_blend on toy_sin (periodic structure
  matches the oscillator). [matrix]
- H2 (safety): pulse_sin random Δ% ≤ +5% vs gated_blend; learned
  amplitude A stays small on noise. [matrix + diag]
- H3 (robustness, paper claim): pulse_sin gap_ratio < gated_blend
  gap_ratio on ≥2 of 3 datasets. [matrix]
- H4 (superset): pulse_strength=0 reproduces r280 bit-for-bit. [unit test]
- H5 (mechanism, paper control): pulse_noise does NOT reproduce the
  pulse_sin gap-robustness — structure, not magnitude, is responsible.
  [matrix]

## 实现难度
**M** (2-4h). ~200 LOC cell (mostly a re-derived forward with a pulse
term) + ~15 unit tests + ~230 LOC bench. Toy grid is small (24 cells,
hidden=128, T=48, 50 epochs) to fit the loop window.

## 风险
- Pulse could destabilize training (large A) → mitigated by small
  `pulse_amp_init=0.1` and grad clipping (already in harness).
- Toy_sin is near-saturated (r280 noted < 1e-4 everywhere) → the clean
  toy_sin signal may be init-noise-dominated; the gap_ratio axis is the
  discriminating measurement this round.
- If pulse helps neither clean nor gap → HONEST NEGATIVE (pulse is a
  vision/gap-heavy mechanism that doesn't transfer to dense 1D), still a
  clean replication result.
