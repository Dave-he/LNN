---
title: "PRD #10-120 — Mixed-Regime Benchmark for the Gate Line (within-sequence regime shifts)"
round: 281
date: 2026-07-03
author: "Claude (r281 /loop session)"
status: "draft"
parent: "r280 blend gate (toy benchmark SATURATED — need harder task)"
variant: "A"
---

> **Selected** (round 281, 2026-07-03): r280 proved the homogeneous toy
> benchmark is saturated (toy_sin <1e-4 everywhere, init noise dominates).
> The gates were designed for WITHIN-sequence regime shifts (natural-gas
> LNN arXiv:2604.24788) — a setting NONE of r277-280's datasets contain.
> A mixed-regime benchmark is the discriminating task the gate line needs,
> and it directly tests each gate's core claim. Zero new engine code.

# PRD #10-120 — Mixed-Regime Benchmark for the Gate Line

## 目标
Discriminate the four liquid-τ gate variants (ungated / velocity /
acceleration / blend) on a HARD task they were actually designed for:
sequences that transition between smooth, structured, and noisy regimes
WITHIN a single sequence — the nonstationary setting the 2026 LTC
literature targets.

## 用户故事
- As an STE-line maintainer, I can finally see which gate is best on
  realistic nonstationary data (not saturated toys), so production gate
  choice is evidenced on the task that matters.
- As a researcher, I can test whether blend's "trust if EITHER signal
  fires" genuinely helps when regimes shift mid-sequence, so I know if
  r280's structured win generalizes.
- As a downstream user, I get gate guidance for real nonstationary
  signals, not just homogeneous toys.

## Mechanism (data, not model — zero new engine code)
```
mixed_regime sequence (T=96, 3 segments of 32):
  seg 0 [0:32]  : smooth sine   (predictable, low accel)
  seg 1 [32:64] : i.i.d. noise  (unpredictable, high accel)
  seg 2 [64:96] : structured    (piecewise-constant levels)
one-step-ahead prediction; loss measured PER-SEGMENT + overall.
```
This is the ONLY setting where the gates can differentiate: a good gate
keeps liquid τ active in segs 0/2 (predictable) and collapses it in seg 1
(noise). Homogeneous toys can't show this because the gate is constant
across the whole sequence.

## 引擎层职责 (canonical)
- No new engine code. Reuse the four gate cells: STEWithEntropy (r267),
  PredictabilityGatedLiquidTauCfCCell (r278), AccelGatedLiquidTauCfCCell
  (r279), BlendGatedLiquidTauCfCCell (r280) + LiquidTauSTECfCCell (r277).

## 游戏层职责
- `scripts/bench_mixed_regime_gates.py` (NEW) — mixed-regime data
  generator + per-segment MSE reporting. Modes: static / liquid /
  gated_vel / gated_accel / gated_blend.
- `analysis/mixed_regime_gates_bench.json`, report.
- Optional: 1-2 sanity unit tests for the data generator (segment
  boundaries, shape).

## 验收标准
- H1: on mixed-regime OVERALL mse, at least one gate beats static
  (the task is now hard enough to separate them). [bench]
- H2 (headline): blend or accel wins the PREDICTABLE segments (0,2)
  while all gates degrade gracefully on the NOISE segment (1). [per-seg]
- H3: ungated liquid is UNSAFE on mixed (noise segment destabilises the
  whole sequence via carried-over state), reproducing r277's failure in
  a within-sequence form. [bench]
- H4: the task is NOT saturated — overall mse > 1e-3 for all modes (so
  results are not init noise). [scale check]
- H5: gate_mean tracks the regime — high in segs 0/2, low in seg 1
  (per-segment gate diagnostics). [diagnostics]

## 实现难度
**S-M** (2-4h). Data generator (~40 LOC) + bench (reuse r280 harness,
add per-segment split) + report. 5 modes × 3 seeds × (1 mixed dataset)
= 15 cells, but T=96 so ~1.5× slower per cell ≈ 20 min. Add 2 homogeneous
controls for calibration = 45 cells ≈ 45 min (trim seeds if needed).

## 风险
- If all gates tie on mixed (task too hard for everyone), the round is a
  NEGATIVE but still resolves "is the toy saturation the whole story?"
- Per-segment loss attribution needs care (carried state means seg-2 loss
  depends on how seg-1 noise corrupted the hidden state). Mitigation:
  report both per-segment AND overall; the per-segment is diagnostic.
- Load-bearing: within-sequence regime shift is where gates differ. If
  even this doesn't separate them, the gate line is done (also useful).
