# PRD #10-61 — Segment Reliability Gate (Round 99)

**Date**: 2026-06-15
**Round**: 99
**Status**: Drafted.

## 1. Why round 99

arXiv:2606.03631 (Xie et al., KDD 2026) — *AnchorMoE: Interpretable Time Series Classification via Anchor-Routed MoE* — proposes a time-series MoE that:

1. Routes local patches (segments) of the time series to specialized experts
2. Applies a **geometric orthogonality constraint** between expert representations (extending rounds 80, 96, 97)
3. Uses an **uncertainty-aware reliability gate** that dynamically calibrates each segment's contribution to the final prediction, suppressing **residual background noise** that the orthogonality constraint alone may not filter

The orthogonality axis is already well-covered in our stack (rounds 80, 96, 97 + audit in round 90). But the **reliability gate** is a **fresh mechanism** that complements our existing gates:

- Round 84-86 (Ecology gates): gate triggered by ecology number E
- Round 89 (Causality gate): gate triggered by per-expert gradient imbalance
- **Round 99 (Reliability gate, NEW)**: gate triggered by per-segment / per-input uncertainty/reliability

The reliability gate is different from the ecology/causality gates because:
- The signal is **per-input** (computed from the input's local statistics), not per-expert
- The gate's purpose is to **suppress noise**, not to **prevent expert collapse**
- It is the **input-side** analog of our existing gates (rounds 84-86, 89 are **expert-side**)

## 2. Hypotheses

- **H1 (reliability gate reduces noise sensitivity)**: with the gate enabled, the model's predictions on noisy inputs (toy_sin + Gaussian noise) are closer to clean-input predictions than without the gate. Measured as `mean(|y_pred_noisy - y_pred_clean|)`.
- **H2 (reliability gate is safe for clean inputs)**: with the gate enabled, task loss on clean inputs is within ±5% of baseline.
- **H3 (reliability gate composes with ecology gate)**: combined reliability + ecology gate is safer than either alone (combined never worse than individual gates).

## 3. Plan

### 3.1 Implementation (`lnn/core/reliability_gate.py` — NEW file)

Add 1 new function:
- `segment_reliability(x, sigma_min=0.01)` — computes a per-input reliability score based on local input statistics. Returns a scalar in [0, 1] where 1 = high reliability (smooth, low-noise) and 0 = low reliability (high-noise, edge case).

Mechanism: `r = 1 / (1 + σ_local / σ_min)`, where `σ_local` is the local standard deviation of the input segment. This gives:
- Constant input: `σ_local = 0` → `r = 1` (highly reliable, but probably useless — not a problem since constant input means trivial task)
- Low-noise: `σ_local < σ_min` → `r > 0.5` (reliable)
- High-noise: `σ_local > σ_min` → `r < 0.5` (suppressed)

This is **stochastic-aware** (the lower the local noise, the higher the reliability).

### 3.2 Apply the gate (`lnn/core/reliability_gate.py`)

Add 1 new function:
- `apply_reliability_gate(y_pred, x, sigma_min=0.01, mix=1.0)` — returns `(1 - mix) * y_pred + mix * r * y_pred`, where `r` is the reliability. This dampens noisy inputs' contribution to the final prediction.

For `mix=1.0`, the gate fully suppresses low-reliability segments. For `mix=0.0`, the gate is disabled.

### 3.3 Tests (`tests/test_reliability_gate.py` — NEW file)

7 new tests:
1. `test_zero_reliability_for_constant_input` — constant input → r=1 (high reliability)
2. `test_high_reliability_for_smooth_input` — smooth input → r > 0.5
3. `test_low_reliability_for_noisy_input` — noisy input → r < 0.5
4. `test_reliability_in_unit_interval` — 0 ≤ r ≤ 1 always
5. `test_sigma_min_affects_threshold` — larger σ_min → more inputs are "reliable"
6. `test_gate_dampens_noisy_output` — noisy output with gate < noisy output without gate
7. `test_gate_preserves_clean_output` — clean output with gate ≈ clean output without gate
8. `test_gate_exported` — module export

### 3.4 Bench (`scripts/bench_segment_reliability_gate.py` — NEW)

36 cells:
- 3 datasets (toy_sin, structured, random)
- 2 noise levels (clean, +Gaussian noise σ=0.1)
- 2 conditions (baseline, +reliability gate)
- 3 seeds, 100 epochs
- 2 models: CfC, LSTM (skip MLP since it has no temporal structure to test)

For each cell measure:
- `task_loss` (final MSE on noisy input)
- `clean_consistency` = `mean(|y_pred_noisy - y_pred_clean|)` (H1)
- `gate_value` = the average r used during the test

Compare baseline vs. +gate for both noise levels.

## 4. Expected outcomes

| model | dataset | clean baseline | clean +gate | noisy baseline | noisy +gate |
|-------|---------|----------------|-------------|-----------------|--------------|
| CfC   | toy_sin | 0.15           | 0.15        | 0.45            | 0.30 (-33%) |
| CfC   | random  | 0.83           | 0.83        | 0.95            | 0.85 (-11%) |
| LSTM  | toy_sin | 0.13           | 0.13        | 0.42            | 0.32 (-24%) |

H1 ✓ if `clean_consistency` drops with gate. H2 ✓ if clean task loss within ±5%. H3 (composition with ecology) tested in a follow-up bench if H1+H2 hold.

## 5. Why this matters

- The reliability gate is a **new axis** of gating — input-side noise suppression, complementing expert-side gates from rounds 84-86, 89.
- It connects to round 92/93 (temporal dropout): the reliability gate is **adaptive** rather than fixed-rate.
- It addresses the "background noise" problem highlighted in round 93's conclusion that input-side dropout helps on noisy data.

## 6. Files

- `docs/prds/2026-06-15-lnn-round-99-a-segment-reliability-gate.md` (this file)
- `lnn/core/reliability_gate.py` (NEW) — 2 new functions
- `lnn/core/__init__.py` — export
- `tests/test_reliability_gate.py` (NEW) — 8 tests
- `scripts/bench_segment_reliability_gate.py` (NEW) — 36-cell bench
- `results/bench_segment_reliability_gate.json`
- `docs/research/2026-06-15_segment_reliability_gate_report.md`
- `docs/daily/2026-06-15_LNN_research_summary_v25.md`
- `README.md` — new section

## 7. Risk

Low. The new functions are simple, deterministic, and easy to test. The bench reuses the round 91/94/98 infrastructure.

## 8. Compatibility

- Functions follow the same signature pattern as `backward_coherence_loss`, `orthogonality_loss`, etc.
- Returns 0-d tensors (or scalars) for composability.
- No Pyright warnings expected beyond pre-existing torch-import false-positives.
